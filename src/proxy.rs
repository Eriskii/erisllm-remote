//! A local HTTP API in front of a remote server, so any language can use it.
//!
//!   POST /v1/generate     request JSON in, events out as NDJSON (one JSON object per line),
//!                         streamed as they come. Closing the connection cancels it.
//!   POST /v1/tokenize     {"text": ...}         -> {"tokens": [...]}
//!   POST /v1/detokenize   {"tokens": [...]}     -> {"text": ..., "pieces": [...]}
//!   GET  /v1/info                               -> the model, adapter, lens and limits
//!
//! On a loopback address it refuses browsers (any request with an Origin header) and Host
//! names other than its own, so a web page cannot use it behind your back.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use erisllm_remote::Client;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

struct Proxy {
    client: Client,
    listen: SocketAddr,
}

pub async fn proxy(client: Client, listen: SocketAddr) -> Result<()> {
    let state = Arc::new(Proxy { client, listen });
    let app = Router::new()
        .route("/v1/generate", post(generate))
        .route("/v1/tokenize", post(|s, h, b| simple(s, h, b, "tokenize")))
        .route("/v1/detokenize", post(|s, h, b| simple(s, h, b, "detokenize")))
        .route("/v1/info", get(|s, h| simple(s, h, Bytes::from_static(b"{}"), "info")))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    println!("API on http://{listen}/v1/");
    axum::serve(listener, app).await?;
    Ok(())
}

fn refused(proxy: &Proxy, headers: &HeaderMap) -> Option<Response> {
    if !proxy.listen.ip().is_loopback() {
        return None;
    }
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok());
    let port = proxy.listen.port();
    let host_ok = host.is_none_or(|h| h == format!("127.0.0.1:{port}") || h == format!("localhost:{port}") || h == format!("[::1]:{port}"));
    (headers.contains_key(header::ORIGIN) || !host_ok).then(|| error(StatusCode::FORBIDDEN, "only local programs may use this API"))
}

fn error(status: StatusCode, message: impl std::fmt::Display) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], json!({"event": "error", "message": message.to_string()}).to_string())
        .into_response()
}

fn body_object(body: &Bytes) -> Result<Map<String, Value>, Response> {
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(fields)) => Ok(fields),
        _ => Err(error(StatusCode::BAD_REQUEST, "the body must be a JSON object")),
    }
}

async fn generate(State(proxy): State<Arc<Proxy>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(response) = refused(&proxy, &headers) {
        return response;
    }
    let mut request = match body_object(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    request.insert("op".into(), "generate".into());
    let mut events = match proxy.client.request(&Value::Object(request)).await {
        Ok(events) => events,
        Err(e) => return error(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    };
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(256);
    tokio::spawn(async move {
        loop {
            let event = tokio::select! {
                event = events.next() => event,
                // The HTTP client went away: dropping `events` cancels the request.
                _ = tx.closed() => break,
            };
            let event = match event {
                Ok(Some(event)) => event,
                Ok(None) => break,
                Err(e) => json!({"event": "error", "message": format!("{e:#}")}),
            };
            let last = erisllm_remote::is_last(&event);
            let mut line = event.to_string();
            line.push('\n');
            if tx.send(Ok(Bytes::from(line))).await.is_err() || last {
                break;
            }
        }
    });
    Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

async fn simple(State(proxy): State<Arc<Proxy>>, headers: HeaderMap, body: Bytes, op: &'static str) -> Response {
    if let Some(response) = refused(&proxy, &headers) {
        return response;
    }
    let mut request = match body_object(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    request.insert("op".into(), op.into());
    let answer = async {
        let mut events = proxy.client.request(&Value::Object(request)).await?;
        events.next().await?.ok_or_else(|| anyhow::anyhow!("the server sent no answer"))
    }
    .await;
    match answer {
        Ok(mut event) => {
            let status = if event["event"] == "error" { StatusCode::BAD_REQUEST } else { StatusCode::OK };
            if status == StatusCode::OK
                && let Some(fields) = event.as_object_mut()
            {
                fields.remove("event");
            }
            (status, [(header::CONTENT_TYPE, "application/json")], event.to_string()).into_response()
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    }
}
