//! What an erisllm-remote client and server say to each other, and a client for it.
//!
//! A client keeps one iroh connection to the server and opens one bidirectional stream
//! per request. It writes the request as one line of JSON and finishes its side; the
//! server answers with events, one JSON object per line, and finishes its side after the
//! last one, which is always a "done", "result" or "error" event. Stopping the stream
//! early cancels the request. Requests and events are documented in README.md.
//!
//! Both sides are known by their iroh endpoint ids. The server only answers clients whose
//! ids are in its allowlist.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use iroh::endpoint::{Connection, RecvStream, VarInt, presets};
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader, Lines};
use tokio::sync::Mutex;

pub const ALPN: &[u8] = b"erisllm/remote/1";
/// The longest request line the server reads.
pub const MAX_REQUEST: u64 = 16 << 20;

/// Where keys and the allowlist live: $ERISLLM_REMOTE_HOME, or the system's config folder.
pub fn home() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("ERISLLM_REMOTE_HOME") {
        return Ok(dir.into());
    }
    Ok(dirs::config_dir().context("this system has no config folder; set ERISLLM_REMOTE_HOME")?.join("erisllm-remote"))
}

/// This machine's key for a role ("server" or "client"), made the first time. Keeping it
/// keeps the id the same.
pub fn secret(role: &str) -> Result<SecretKey> {
    let path = home()?.join(format!("{role}.key"));
    match fs::read(&path) {
        Ok(bytes) => {
            let bytes: [u8; 32] = bytes.try_into().map_err(|_| anyhow!("{} is not a key", path.display()))?;
            Ok(SecretKey::from_bytes(&bytes))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let key = SecretKey::generate();
            write_private(&path, &key.to_bytes()).with_context(|| path.display().to_string())?;
            Ok(key)
        }
        Err(error) => Err(error).with_context(|| path.display().to_string()),
    }
}

fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    io::Write::write_all(&mut options.open(path)?, bytes)
}

/// Whether this event is the last one of its request.
pub fn is_last(event: &Value) -> bool {
    matches!(event.get("event").and_then(Value::as_str), Some("done" | "result" | "error"))
}

/// A connection to one server, made again if it drops.
pub struct Client {
    endpoint: Endpoint,
    server: EndpointAddr,
    connection: Mutex<Option<Connection>>,
}

impl Client {
    /// Connects to the server, as this machine's client key. Its id is enough wherever iroh's
    /// address lookup works; direct addresses can be added for where it does not.
    pub async fn connect(server: impl Into<EndpointAddr>) -> Result<Client> {
        let server = server.into();
        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(secret("client")?)
            .bind()
            .await
            .context("starting the network")?;
        let client = Client { endpoint, server, connection: Mutex::new(None) };
        client.connection(false).await?;
        Ok(client)
    }

    /// This machine's client id, which the server's owner adds with `erisllm-remote allow`.
    pub fn id(&self) -> EndpointId {
        self.endpoint.id()
    }

    async fn connection(&self, fresh: bool) -> Result<Connection> {
        let mut slot = self.connection.lock().await;
        if let Some(connection) = slot.as_ref().filter(|c| !fresh && c.close_reason().is_none()) {
            return Ok(connection.clone());
        }
        let id = self.server.id.fmt_short();
        let connection = tokio::time::timeout(Duration::from_secs(30), self.endpoint.connect(self.server.clone(), ALPN))
            .await
            .map_err(|_| anyhow!("the server {id} did not answer within 30 s"))?
            .with_context(|| format!("connecting to {id}"))?;
        *slot = Some(connection.clone());
        Ok(connection)
    }

    /// Sends a request and returns its events as they come.
    pub async fn request(&self, request: &Value) -> Result<Events> {
        let mut line = serde_json::to_string(request)?;
        line.push('\n');
        let mut last_error = None;
        for attempt in 0..2 {
            let connection = self.connection(attempt > 0).await?;
            let sent = async {
                let (mut send, recv) = connection.open_bi().await?;
                send.write_all(line.as_bytes()).await?;
                send.finish()?;
                anyhow::Ok(recv)
            }
            .await;
            match sent {
                Ok(recv) => {
                    let lines = BufReader::new(recv).lines();
                    return Ok(Events { lines, finished: false, connection, me: self.id() });
                }
                Err(error) => match refused(&connection, self.id()) {
                    Some(refusal) => return Err(refusal),
                    None => last_error = Some(error),
                },
            }
        }
        Err(last_error.unwrap())
    }

    pub async fn close(self) {
        self.endpoint.close().await;
    }
}

/// If the server turned this machine away, says what to do about it.
fn refused(connection: &Connection, me: EndpointId) -> Option<anyhow::Error> {
    let reason = connection.close_reason()?;
    reason.to_string().contains("not allowed").then(|| {
        anyhow!("the server does not know this machine; its owner can let it in with\n  erisllm-remote allow {me}")
    })
}

/// The events answering one request. Dropping this before the last event cancels the request.
pub struct Events {
    lines: Lines<BufReader<RecvStream>>,
    finished: bool,
    connection: Connection,
    me: EndpointId,
}

impl Events {
    /// The next event, or None after the last one.
    pub async fn next(&mut self) -> Result<Option<Value>> {
        if self.finished {
            return Ok(None);
        }
        let line = match self.lines.next_line().await {
            Ok(line) => line,
            Err(error) => {
                self.finished = true;
                return Err(refused(&self.connection, self.me).unwrap_or_else(|| anyhow!(error).context("reading from the server")));
            }
        };
        match line {
            Some(line) => {
                let event: Value = serde_json::from_str(&line).context("the server sent something that is not JSON")?;
                self.finished = is_last(&event);
                Ok(Some(event))
            }
            None => {
                self.finished = true;
                Err(anyhow!("the server ended the request without finishing it"))
            }
        }
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.lines.get_mut().get_mut().stop(VarInt::from_u32(0));
        }
    }
}
