//! erisllm-remote: serve ErisLLM from one machine and use it from another, over iroh.
//! See README.md.

mod proxy;
mod server;

use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use erisllm_remote::{Client, home, secret};
use iroh::{EndpointAddr, EndpointId};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "erisllm-remote", version, about = "Serve ErisLLM from one machine and use it from another, over iroh")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the model on this machine (an Apple Silicon Mac with the ErisLLM checkout).
    Serve {
        /// The ErisLLM checkout (default: the one this was built from).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Python with mlx-lm installed (default: ROOT/.venv/bin/python).
        #[arg(long)]
        python: Option<PathBuf>,
        /// Adapter folder (default: ROOT/adapter).
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Base model (default: the one the adapter was trained on).
        #[arg(long)]
        model: Option<String>,
        /// Jacobian lens: "auto" (the pre-fitted one for the base model), "none", or a file.
        #[arg(long, default_value = "auto")]
        jlens: String,
        /// Answer every client, not only those in the allowlist. For testing.
        #[arg(long)]
        allow_any: bool,
        /// Listen on this UDP port (IPv4) instead of a random one.
        #[arg(long)]
        port: Option<u16>,
        /// Where the adapter applies when a request does not say: all, mlp, attention or none.
        #[arg(long, default_value = "all")]
        lora_parts: String,
    },
    /// Let a client in. Its id comes from running `erisllm-remote id` on its machine.
    Allow {
        id: String,
        /// A name to show in the server's log.
        name: Option<String>,
    },
    /// Print this machine's client id, for the server's owner to `allow`.
    Id,
    /// Send one request and print its events, one JSON object per line. The request is a
    /// JSON object read from FILE, or from stdin if no FILE is given.
    Request {
        #[command(flatten)]
        server: Server,
        file: Option<PathBuf>,
    },
    /// Show what the server runs: model, adapter, Jacobian lens, limits.
    Info {
        #[command(flatten)]
        server: Server,
    },
    /// Serve a local HTTP API (POST /v1/generate, ...) that forwards to the server.
    Proxy {
        #[command(flatten)]
        server: Server,
        #[arg(long, default_value = "127.0.0.1:8799")]
        listen: SocketAddr,
    },
}

#[derive(clap::Args)]
struct Server {
    /// The server's id, printed by `erisllm-remote serve`.
    #[arg(long, env = "ERISLLM_SERVER")]
    server: String,
    /// Direct addresses of the server (IP:PORT, comma-separated), for networks where iroh's
    /// address lookup is blocked. Usually not needed.
    #[arg(long = "addr", env = "ERISLLM_SERVER_ADDR", value_delimiter = ',')]
    addrs: Vec<SocketAddr>,
}

impl Server {
    fn addr(&self) -> Result<EndpointAddr> {
        let id: EndpointId = self.server.trim().parse().with_context(|| format!("{} is not a server id", self.server))?;
        Ok(self.addrs.iter().fold(EndpointAddr::new(id), |addr, ip| addr.with_ip_addr(*ip)))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Serve { root, python, adapter, model, jlens, allow_any, port, lora_parts } => {
            let root = match root {
                Some(root) => root,
                None => find_root()?,
            };
            let python = python.unwrap_or_else(|| root.join(".venv/bin/python"));
            let mut worker_args = vec!["--jlens".to_owned(), jlens, "--lora-parts".to_owned(), lora_parts];
            if let Some(adapter) = adapter {
                worker_args.extend(["--adapter".to_owned(), adapter.canonicalize()?.display().to_string()]);
            }
            if let Some(model) = model {
                worker_args.extend(["--model".to_owned(), model]);
            }
            server::serve(server::Options { root, python, worker_args, allow_any, port }).await
        }
        Command::Allow { id, name } => {
            let peer: EndpointId = id.parse().with_context(|| format!("{id} is not a client id"))?;
            let path = server::allowlist_path()?;
            std::fs::create_dir_all(path.parent().unwrap())?;
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
            writeln!(file, "{peer}{}", name.map(|n| format!(" {n}")).unwrap_or_default())?;
            println!("allowed {} (in {}); a running server picks this up on the next connection", peer.fmt_short(), path.display());
            Ok(())
        }
        Command::Id => {
            println!("{}", secret("client")?.public());
            eprintln!("(the key is in {})", home()?.display());
            Ok(())
        }
        Command::Request { server, file } => {
            let mut text = String::new();
            match file {
                Some(file) => text = std::fs::read_to_string(&file).with_context(|| file.display().to_string())?,
                None => {
                    std::io::stdin().read_to_string(&mut text)?;
                }
            }
            let request: Value = serde_json::from_str(&text).context("the request is not JSON")?;
            if !request.is_object() {
                bail!("the request must be a JSON object");
            }
            let client = Client::connect(server.addr()?).await?;
            let failed = print_events(&client, &request).await;
            client.close().await;
            if failed? {
                std::process::exit(1);
            }
            Ok(())
        }
        Command::Info { server } => {
            let client = Client::connect(server.addr()?).await?;
            let failed = print_events(&client, &json!({"op": "info"})).await;
            client.close().await;
            if failed? {
                std::process::exit(1);
            }
            Ok(())
        }
        Command::Proxy { server, listen } => {
            let client = Client::connect(server.addr()?).await?;
            eprintln!("connected to {} as {}", server.server.trim(), client.id().fmt_short());
            proxy::proxy(client, listen).await
        }
    }
}

/// The folder holding the Python side (erisllm/worker.py): the current folder, or the
/// checkout this was built from (whose crate sits at its top or in remote/).
fn find_root() -> Result<PathBuf> {
    let built = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [std::env::current_dir()?, built.clone(), built.join("..")];
    candidates
        .into_iter()
        .find(|dir| dir.join("erisllm/worker.py").exists())
        .map(|dir| dir.canonicalize())
        .transpose()?
        .context("cannot find erisllm/worker.py; run from the checkout or pass --root")
}

/// Prints every event of one request; whether it ended in an error.
async fn print_events(client: &Client, request: &Value) -> Result<bool> {
    let mut events = client.request(request).await?;
    let mut out = std::io::stdout().lock();
    let mut failed = false;
    while let Some(event) = events.next().await? {
        failed = event["event"] == "error";
        writeln!(out, "{event}")?;
        out.flush()?;
    }
    Ok(failed)
}
