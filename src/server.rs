//! The server: an iroh endpoint in front of the model process (`python -m erisllm.worker`).
//!
//! The worker reads requests on stdin and writes events on stdout, one JSON object per
//! line, each tagged with the request's id. This side gives every incoming request a fresh
//! id, routes the worker's events back to the stream it came from, and cancels a request
//! whose client stops listening.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use erisllm_remote::{ALPN, MAX_REQUEST, home, is_last, secret};
use iroh::endpoint::{Connection, RecvStream, SendStream, presets};
use iroh::{Endpoint, EndpointId};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

pub struct Options {
    /// The ErisLLM checkout: the worker runs as `python -m erisllm.worker` from here.
    pub root: PathBuf,
    pub python: PathBuf,
    pub worker_args: Vec<String>,
    pub allow_any: bool,
    /// A fixed UDP port (IPv4), for port forwarding or `--addr` hints; random if None.
    pub port: Option<u16>,
}

struct Worker {
    input: mpsc::UnboundedSender<String>,
    pending: Mutex<HashMap<u64, mpsc::UnboundedSender<Value>>>,
    next: AtomicU64,
}

impl Worker {
    fn submit(&self, mut request: Map<String, Value>) -> (u64, mpsc::UnboundedReceiver<Value>) {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        request.insert("id".into(), id.into());
        let (tx, rx) = mpsc::unbounded_channel();
        self.pending.lock().unwrap().insert(id, tx);
        let _ = self.input.send(Value::Object(request).to_string());
        (id, rx)
    }

    fn cancel(&self, id: u64) {
        let _ = self.input.send(json!({"op": "cancel", "id": id}).to_string());
    }

    fn forget(&self, id: u64) {
        self.pending.lock().unwrap().remove(&id);
    }
}

/// The allowlist: one client id per line, optionally followed by a name; `#` starts a comment.
pub fn allowlist_path() -> Result<PathBuf> {
    Ok(home()?.join("allowed"))
}

fn allowed(peer: &EndpointId) -> Result<Option<String>> {
    let text = match std::fs::read_to_string(allowlist_path()?) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let mut parts = line.splitn(2, char::is_whitespace);
        if let Some(id) = parts.next().filter(|id| !id.is_empty())
            && id.parse::<EndpointId>().ok().as_ref() == Some(peer)
        {
            let name = parts.next().map(str::trim).filter(|n| !n.is_empty());
            return Ok(Some(name.map_or_else(|| peer.fmt_short().to_string(), str::to_owned)));
        }
    }
    Ok(None)
}

pub async fn serve(options: Options) -> Result<()> {
    let mut child = Command::new(&options.python)
        .arg("-m")
        .arg("erisllm.worker")
        .args(&options.worker_args)
        .current_dir(&options.root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("starting {}", options.python.display()))?;
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    eprintln!("loading the model…");
    let ready: Value = loop {
        let Some(line) = lines.next_line().await? else {
            bail!("the model process exited before it was ready (its errors are above)");
        };
        let event: Value = serde_json::from_str(&line).context("the model process wrote something that is not JSON")?;
        if event["event"] == "ready" {
            break event;
        }
    };
    eprintln!(
        "model ready: {} + {} on {}{}",
        ready["model"].as_str().unwrap_or("?"),
        ready["adapter"].as_str().unwrap_or("?"),
        ready["engine"].as_str().unwrap_or("?"),
        match ready["jlens"]["source"].as_str() {
            Some(source) => format!(", Jacobian lens {source}"),
            None => ", no Jacobian lens".into(),
        }
    );

    let (input, mut input_rx) = mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        while let Some(mut line) = input_rx.recv().await {
            line.push('\n');
            if stdin.write_all(line.as_bytes()).await.is_err() || stdin.flush().await.is_err() {
                break;
            }
        }
    });
    let worker = Arc::new(Worker { input, pending: Mutex::new(HashMap::new()), next: AtomicU64::new(1) });

    let router = {
        let worker = worker.clone();
        tokio::spawn(async move {
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(mut event) = serde_json::from_str::<Value>(&line) else {
                    eprintln!("model process: {line}");
                    continue;
                };
                let id = event.get("id").and_then(Value::as_u64);
                if let Some(fields) = event.as_object_mut() {
                    fields.remove("id");
                }
                let Some(id) = id else {
                    eprintln!("model process: {event}");
                    continue;
                };
                let last = is_last(&event);
                let mut pending = worker.pending.lock().unwrap();
                if let Some(tx) = pending.get(&id) {
                    let _ = tx.send(event);
                }
                if last {
                    pending.remove(&id);
                }
            }
            for (_, tx) in worker.pending.lock().unwrap().drain() {
                let _ = tx.send(json!({"event": "error", "message": "the model process stopped"}));
            }
        })
    };

    let mut builder = Endpoint::builder(presets::N0).secret_key(secret("server")?).alpns(vec![ALPN.to_vec()]);
    if let Some(port) = options.port {
        builder = builder.bind_addr(format!("0.0.0.0:{port}"))?;
    }
    let endpoint = builder.bind().await.context("starting the network")?;
    println!("serving as {}", endpoint.id());
    let ports: Vec<String> = endpoint.bound_sockets().iter().map(|a| a.to_string()).collect();
    eprintln!("listening on UDP {}", ports.join(", "));
    if options.allow_any {
        eprintln!("answering ANY client (--allow-any)");
    } else {
        eprintln!("answering the clients listed in {}", allowlist_path()?.display());
    }

    let result = tokio::select! {
        _ = accept(endpoint.clone(), worker, options.allow_any) => Ok(()),
        _ = router => Err(anyhow::anyhow!("the model process stopped")),
        _ = tokio::signal::ctrl_c() => Ok(()),
    };
    endpoint.close().await;
    let _ = child.kill().await;
    result
}

async fn accept(endpoint: Endpoint, worker: Arc<Worker>, allow_any: bool) {
    while let Some(incoming) = endpoint.accept().await {
        let worker = worker.clone();
        tokio::spawn(async move {
            let Ok(accepting) = incoming.accept() else { return };
            let Ok(connection) = accepting.await else { return };
            let peer = connection.remote_id();
            let name = match allowed(&peer) {
                Ok(Some(name)) => name,
                Ok(None) if allow_any => peer.fmt_short().to_string(),
                Ok(None) => {
                    eprintln!("refused {peer}: not in the allowlist");
                    connection.close(1u32.into(), b"not allowed");
                    return;
                }
                Err(error) => {
                    eprintln!("refused {peer}: reading the allowlist failed: {error}");
                    connection.close(1u32.into(), b"not allowed");
                    return;
                }
            };
            eprintln!("{name}: connected");
            serve_connection(&connection, &worker, &name).await;
            eprintln!("{name}: disconnected");
        });
    }
}

async fn serve_connection(connection: &Connection, worker: &Arc<Worker>, name: &str) {
    while let Ok((send, recv)) = connection.accept_bi().await {
        let (worker, name) = (worker.clone(), name.to_owned());
        tokio::spawn(async move { serve_stream(send, recv, &worker, &name).await });
    }
}

async fn serve_stream(mut send: SendStream, recv: RecvStream, worker: &Worker, name: &str) {
    let mut line = Vec::new();
    let request = match BufReader::new(recv.take(MAX_REQUEST)).read_until(b'\n', &mut line).await {
        Ok(_) => match serde_json::from_slice::<Value>(&line) {
            Ok(Value::Object(request)) if request.get("op").and_then(Value::as_str) != Some("cancel") => Ok(request),
            Ok(Value::Object(_)) => Err("cancel a request by stopping its stream".to_owned()),
            _ => Err("a request is one line of JSON: an object".to_owned()),
        },
        Err(error) => Err(format!("reading the request failed: {error}")),
    };
    let request = match request {
        Ok(request) => request,
        Err(message) => {
            let _ = send.write_all(format!("{}\n", json!({"event": "error", "message": message})).as_bytes()).await;
            let _ = send.finish();
            return;
        }
    };
    let op = request.get("op").and_then(Value::as_str).unwrap_or("generate").to_owned();
    let (id, mut events) = worker.submit(request);
    let stopped = send.stopped();
    tokio::pin!(stopped);
    loop {
        tokio::select! {
            event = events.recv() => {
                let Some(event) = event else { break };
                let last = is_last(&event);
                if last && op == "generate" {
                    log(name, &event);
                }
                let mut text = event.to_string();
                text.push('\n');
                if send.write_all(text.as_bytes()).await.is_err() {
                    worker.cancel(id);
                    break;
                }
                if last {
                    break;
                }
            }
            _ = &mut stopped => {
                worker.cancel(id);
                eprintln!("{name}: cancelled a request");
                break;
            }
        }
    }
    worker.forget(id);
    let _ = send.finish();
}

fn log(name: &str, event: &Value) {
    if event["event"] == "error" {
        eprintln!("{name}: error: {}", event["message"].as_str().unwrap_or("?"));
        return;
    }
    eprintln!(
        "{name}: {} prompt ({} cached) + {} generated, {}, {} tok/s",
        event["prompt_tokens"],
        event["reused_tokens"],
        event["generated_tokens"],
        event["finish_reason"].as_str().unwrap_or("?"),
        event["decode_tokens_per_second"],
    );
}
