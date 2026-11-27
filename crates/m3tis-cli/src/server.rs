//! Zero-dependency OpenAI-compatible HTTP/1.1 + SSE inference server (`m3tis serve`).
//!
//! Endpoints:
//! - `GET  /health`                : Live status, VRAM slots, drafter state
//! - `GET  /v1/models`             : OpenAI model catalog
//! - `POST /v1/chat/completions`   : OpenAI Chat Completions (non-streaming JSON & `stream: true` SSE,
//!                                   including `<think>...</think>` -> `reasoning_content` splitting!)
//! - `POST /v1/completions`        : Raw text completion endpoint

use m3tis_format::tokenizer::{BpeTokenizer, ChatMessage, StreamDecoder};
use m3tis_models::glm53::{ForwardOpts, Glm53Model, Glm53State};
use std::collections::HashMap;
use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
    pub fwd_opts: ForwardOpts,
    pub max_drafts: usize,
    pub draft_conf: f64,
    pub default_thinking: bool,
}

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn run_server(
    model: &mut Glm53Model,
    state: &mut Glm53State,
    tokenizer: &BpeTokenizer,
    cfg: &ServerConfig,
) -> Result<(), Box<dyn Error>> {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr)?;
    println!(
        "[m3tis serve] listening on http://{addr} | model={} | drafter={} | vram_slots={} | ram_slots={} | spec={} | dyn_mass={:.2}/{} | thinking={}",
        cfg.model_name,
        model.drafter.is_some(),
        model.load_summary.vram_slots,
        model.load_summary.ram_slots,
        cfg.fwd_opts.spec_prefetch_p,
        cfg.fwd_opts.dyn_topk_mass,
        cfg.fwd_opts.dyn_topk_min_k,
        cfg.default_thinking,
    );
    println!("[m3tis serve] endpoints: GET /health | GET /v1/models | POST /v1/chat/completions | POST /v1/completions");

    for incoming in listener.incoming() {
        match incoming {
            Ok(mut stream) => {
                if let Err(e) = handle_connection(&mut stream, model, state, tokenizer, cfg) {
                    eprintln!("[serve] connection error: {e}");
                }
            }
            Err(e) => eprintln!("[serve] accept error: {e}"),
        }
    }
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn handle_connection(
    stream: &mut TcpStream,
    model: &mut Glm53Model,
    state: &mut Glm53State,
    tokenizer: &BpeTokenizer,
    cfg: &ServerConfig,
) -> Result<(), Box<dyn Error>> {
    let (method, path, body) = match read_http_request(stream)? {
        Some(r) => r,
        None => return Ok(()),
    };
    let clean_path = path.split('?').next().unwrap_or(&path);

    if method == "OPTIONS" {
        write_response(stream, 204, "No Content", "text/plain", b"")?;
        return Ok(());
    }

    match (method.as_str(), clean_path) {
        ("GET", "/health") => {
            let ls = &model.load_summary;
            let body = format!(
                r#"{{"status":"ok","model":{},"drafter":{},"vram_slots":{},"ram_slots":{},"vram_am_pinned":{},"ram_am_pinned":{}}}"#,
                json_quote(&cfg.model_name),
                model.drafter.is_some(),
                ls.vram_slots,
                ls.ram_slots,
                model.pool.vram_cache.protected_count(),
                model.pool.ram_cache.protected_count(),
            );
            write_response(stream, 200, "OK", "application/json", body.as_bytes())?;
        }
        ("GET", "/v1/models") => {
            let now = unix_now();
            let body = format!(
                r#"{{"object":"list","data":[{{"id":{},"object":"model","created":{},"owned_by":"m3tis"}}]}}"#,
                json_quote(&cfg.model_name),
                now
            );
            write_response(stream, 200, "OK", "application/json", body.as_bytes())?;
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat_completions(stream, &body, model, state, tokenizer, cfg)?;
        }
        ("POST", "/v1/completions") => {
            handle_completions(stream, &body, model, state, tokenizer, cfg)?;
        }
        _ => {
            let err = format!(
                r#"{{"error":{{"message":"unknown route {} {}","type":"invalid_request_error"}}}}"#,
                method, clean_path
            );
            write_response(stream, 404, "Not Found", "application/json", err.as_bytes())?;
        }
    }
    Ok(())
}

fn read_http_request(stream: &mut TcpStream) -> Result<Option<(String, String, Vec<u8>)>, Box<dyn Error>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    let n = reader.read_line(&mut request_line)?;
    if n == 0 {
        return Ok(None);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_owned();
    let path = parts.next().unwrap_or("/").to_owned();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    if content_length > 16 * 1024 * 1024 {
        return Err("request body exceeds 16 MiB".into());
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some((method, path, body)))
}

fn write_response(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let hdr = format!(
        "HTTP/1.1 {code} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(hdr.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn write_sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    let hdr = "HTTP/1.1 200 OK\r\n\
               Content-Type: text/event-stream\r\n\
               Cache-Control: no-cache\r\n\
               X-Accel-Buffering: no\r\n\
               Access-Control-Allow-Origin: *\r\n\
               Access-Control-Allow-Headers: *\r\n\
               Connection: close\r\n\r\n";
    stream.write_all(hdr.as_bytes())?;
    stream.flush()
}

fn handle_chat_completions(
    stream: &mut TcpStream,
    body: &[u8],
    model: &mut Glm53Model,
    state: &mut Glm53State,
    tokenizer: &BpeTokenizer,
    cfg: &ServerConfig,
) -> Result<(), Box<dyn Error>> {
    let req = match parse_json_value(body) {
        Ok(JsonVal::Obj(m)) => m,
        Ok(_) => {
            write_response(
                stream,
                400,
                "Bad Request",
                "application/json",
                br#"{"error":{"message":"JSON body must be an object","type":"invalid_request_error"}}"#,
            )?;
            return Ok(());
        }
        Err(e) => {
            let msg = format!(
                r#"{{"error":{{"message":{},"type":"invalid_request_error"}}}}"#,
                json_quote(&format!("invalid JSON: {e}"))
            );
            write_response(stream, 400, "Bad Request", "application/json", msg.as_bytes())?;
            return Ok(());
        }
    };

    let req_model = get_str(&req, "model").unwrap_or(&cfg.model_name).to_owned();
    let max_tokens = get_i64(&req, "max_completion_tokens")
        .or_else(|| get_i64(&req, "max_tokens"))
        .unwrap_or(512)
        .clamp(1, 4096) as usize;
    let temperature = get_f64(&req, "temperature").unwrap_or(0.0) as f32;
    let top_p = get_f64(&req, "top_p").unwrap_or(0.95) as f32;
    let do_stream = get_bool(&req, "stream").unwrap_or(false);
    let enable_thinking = get_bool(&req, "enable_thinking")
        .or_else(|| get_bool(&req, "thinking"))
        .unwrap_or(cfg.default_thinking);
    let reasoning_effort = get_str(&req, "reasoning_effort");
    let seed = get_i64(&req, "seed").map(|v| v as u64).unwrap_or_else(unix_now);
    let stop_seqs = extract_stops(&req);

    let messages = match req.get("messages") {
        Some(JsonVal::Arr(arr)) => {
            let mut msgs = Vec::with_capacity(arr.len());
            for item in arr {
                if let JsonVal::Obj(mo) = item {
                    let role = get_str(mo, "role").unwrap_or("user").to_owned();
                    let content = extract_message_content(mo.get("content"));
                    msgs.push(ChatMessage { role, content });
                }
            }
            msgs
        }
        _ => {
            write_response(
                stream,
                400,
                "Bad Request",
                "application/json",
                br#"{"error":{"message":"missing 'messages' array","type":"invalid_request_error"}}"#,
            )?;
            return Ok(());
        }
    };

    let prompt_str = tokenizer.format_glm53_chat(&messages, enable_thinking, reasoning_effort);
    let prompt_tokens = tokenizer.encode(&prompt_str);
    if prompt_tokens.len() >= state.max_ctx.saturating_sub(2) {
        let msg = format!(
            r#"{{"error":{{"message":"prompt {} tokens exceeds max_ctx {}","type":"invalid_request_error"}}}}"#,
            prompt_tokens.len(),
            state.max_ctx
        );
        write_response(stream, 400, "Bad Request", "application/json", msg.as_bytes())?;
        return Ok(());
    }

    let req_id = format!("chatcmpl-m3tis-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let created = unix_now();
    let pre_stats = model.pool_stats();

    if do_stream {
        write_sse_headers(stream)?;
        // Initial assistant role delta
        let init_chunk = format!(
            "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}\n\n",
            json_quote(&req_id),
            created,
            json_quote(&req_model)
        );
        if stream.write_all(init_chunk.as_bytes()).is_err() {
            return Ok(());
        }
        let _ = stream.flush();

        let mut dec = StreamDecoder::new();
        let mut in_think = enable_thinking;
        let mut full_text = String::new();
        let mut write_failed = false;

        let res = model.generate_stream(
            state,
            &prompt_tokens,
            max_tokens,
            cfg.max_drafts,
            cfg.draft_conf,
            temperature,
            top_p,
            seed,
            cfg.fwd_opts,
            |tok_id| {
                // Handle GLM-5.3 </think> (154842) / <think> (154841) transitions cleanly!
                if tok_id == 154842 && in_think {
                    in_think = false;
                    return true;
                }
                if tok_id == 154841 && !in_think {
                    in_think = true;
                    return true;
                }
                let piece = dec.push(tokenizer, tok_id, true);
                if piece.is_empty() {
                    return true;
                }
                full_text.push_str(&piece);
                if !stop_seqs.is_empty() && stop_seqs.iter().any(|s| !s.is_empty() && full_text.ends_with(s)) {
                    return false;
                }
                let field = if in_think { "reasoning_content" } else { "content" };
                let chunk = format!(
                    "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{{{}:{}}},\"finish_reason\":null}}]}}\n\n",
                    json_quote(&req_id),
                    created,
                    json_quote(&req_model),
                    json_quote(field),
                    json_quote(&piece)
                );
                if stream.write_all(chunk.as_bytes()).is_err() || stream.flush().is_err() {
                    write_failed = true;
                    return false;
                }
                true
            },
        )?;

        if !write_failed {
            let tail = dec.flush();
            if !tail.is_empty() {
                let field = if in_think { "reasoning_content" } else { "content" };
                let chunk = format!(
                    "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{{{}:{}}},\"finish_reason\":null}}]}}\n\n",
                    json_quote(&req_id),
                    created,
                    json_quote(&req_model),
                    json_quote(field),
                    json_quote(&tail)
                );
                let _ = stream.write_all(chunk.as_bytes());
            }
            let comp_toks = res.generated_tokens.len();
            let done_chunk = format!(
                "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":{}}}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}\n\ndata: [DONE]\n\n",
                json_quote(&req_id),
                created,
                json_quote(&req_model),
                json_quote(res.finish_reason),
                res.prompt_tokens,
                comp_toks,
                res.prompt_tokens + comp_toks
            );
            let _ = stream.write_all(done_chunk.as_bytes());
            let _ = stream.flush();
        }
        log_request_stats(&req_id, true, &res, &pre_stats, &model.pool_stats());
    } else {
        let mut dec = StreamDecoder::new();
        let mut full_text = String::new();
        let res = model.generate_stream(
            state,
            &prompt_tokens,
            max_tokens,
            cfg.max_drafts,
            cfg.draft_conf,
            temperature,
            top_p,
            seed,
            cfg.fwd_opts,
            |tok_id| {
                let piece = dec.push(tokenizer, tok_id, true);
                full_text.push_str(&piece);
                if !stop_seqs.is_empty() && stop_seqs.iter().any(|s| !s.is_empty() && full_text.ends_with(s)) {
                    return false;
                }
                true
            },
        )?;
        full_text.push_str(&dec.flush());
        // Trim matched stop suffix if present
        for s in &stop_seqs {
            if !s.is_empty() && full_text.ends_with(s) {
                let new_len = full_text.len() - s.len();
                full_text.truncate(new_len);
                break;
            }
        }
        let (reasoning, content) = split_thinking(&full_text, enable_thinking);
        let post_stats = model.pool_stats();
        let comp_toks = res.generated_tokens.len();
        let pf_tps = res.prompt_tokens as f64 / res.prefill_secs.max(1e-6);
        let dec_tps = comp_toks.saturating_sub(1) as f64 / res.decode_secs.max(1e-6);
        let vh = post_stats.vram_hits - pre_stats.vram_hits;
        let vm = post_stats.vram_misses - pre_stats.vram_misses;
        let vhit_pct = 100.0 * vh as f64 / (vh + vm).max(1) as f64;
        let tok_rd = if res.spec_stats.rounds > 0 {
            comp_toks.saturating_sub(1) as f64 / res.spec_stats.rounds as f64
        } else {
            1.0
        };

        let reasoning_field = match reasoning {
            Some(r) => format!(",\"reasoning_content\":{}", json_quote(&r)),
            None => String::new(),
        };
        let resp = format!(
            "{{\"id\":{},\"object\":\"chat.completion\",\"created\":{},\"model\":{},\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}{}}},\"finish_reason\":{}}}],\"usage\":{{\"prompt_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{}}},\"completion_tokens\":{},\"total_tokens\":{}}},\"m3tis_stats\":{{\"prefill_secs\":{:.3},\"prefill_tps\":{:.2},\"decode_tps\":{:.2},\"vram_hit_pct\":{:.1},\"tok_per_round\":{:.2},\"prefix_cached_tokens\":{}}}}}",
            json_quote(&req_id),
            created,
            json_quote(&req_model),
            json_quote(&content),
            reasoning_field,
            json_quote(res.finish_reason),
            res.prompt_tokens,
            res.prefix_cached_tokens,
            comp_toks,
            res.prompt_tokens + comp_toks,
            res.prefill_secs,
            pf_tps,
            dec_tps,
            vhit_pct,
            tok_rd,
            res.prefix_cached_tokens
        );
        write_response(stream, 200, "OK", "application/json", resp.as_bytes())?;
        log_request_stats(&req_id, false, &res, &pre_stats, &post_stats);
    }
    Ok(())
}

fn handle_completions(
    stream: &mut TcpStream,
    body: &[u8],
    model: &mut Glm53Model,
    state: &mut Glm53State,
    tokenizer: &BpeTokenizer,
    cfg: &ServerConfig,
) -> Result<(), Box<dyn Error>> {
    let req = match parse_json_value(body) {
        Ok(JsonVal::Obj(m)) => m,
        _ => {
            write_response(
                stream,
                400,
                "Bad Request",
                "application/json",
                br#"{"error":{"message":"invalid JSON object","type":"invalid_request_error"}}"#,
            )?;
            return Ok(());
        }
    };
    let prompt = match get_str(&req, "prompt") {
        Some(p) => p,
        None => {
            write_response(
                stream,
                400,
                "Bad Request",
                "application/json",
                br#"{"error":{"message":"missing 'prompt' string","type":"invalid_request_error"}}"#,
            )?;
            return Ok(());
        }
    };
    let req_model = get_str(&req, "model").unwrap_or(&cfg.model_name).to_owned();
    let max_tokens = get_i64(&req, "max_tokens").unwrap_or(128).clamp(1, 4096) as usize;
    let temperature = get_f64(&req, "temperature").unwrap_or(0.0) as f32;
    let top_p = get_f64(&req, "top_p").unwrap_or(0.95) as f32;
    let seed = get_i64(&req, "seed").map(|v| v as u64).unwrap_or_else(unix_now);
    let prompt_tokens = tokenizer.encode(prompt);
    let req_id = format!("cmpl-m3tis-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let pre_stats = model.pool_stats();

    let res = model.generate_stream(
        state,
        &prompt_tokens,
        max_tokens,
        cfg.max_drafts,
        cfg.draft_conf,
        temperature,
        top_p,
        seed,
        cfg.fwd_opts,
        |_| true,
    )?;
    let text = tokenizer.decode(&res.generated_tokens, true);
    let comp_toks = res.generated_tokens.len();
    let resp = format!(
        "{{\"id\":{},\"object\":\"text_completion\",\"created\":{},\"model\":{},\"choices\":[{{\"index\":0,\"text\":{},\"finish_reason\":{}}}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}",
        json_quote(&req_id),
        unix_now(),
        json_quote(&req_model),
        json_quote(&text),
        json_quote(res.finish_reason),
        res.prompt_tokens,
        comp_toks,
        res.prompt_tokens + comp_toks
    );
    write_response(stream, 200, "OK", "application/json", resp.as_bytes())?;
    log_request_stats(&req_id, false, &res, &pre_stats, &model.pool_stats());
    Ok(())
}

fn split_thinking(text: &str, started_in_think: bool) -> (Option<String>, String) {
    if let Some((before, after)) = text.split_once("</think>") {
        let r = before.strip_prefix("<think>").unwrap_or(before).trim().to_owned();
        let c = after.trim_start_matches('\n').to_owned();
        (if r.is_empty() { None } else { Some(r) }, c)
    } else if started_in_think {
        (Some(text.trim().to_owned()), String::new())
    } else {
        (None, text.to_owned())
    }
}

fn log_request_stats(
    req_id: &str,
    streamed: bool,
    res: &m3tis_models::glm53::GenerateResult,
    pre: &m3tis_cache::PoolStats,
    post: &m3tis_cache::PoolStats,
) {
    let steps = res.generated_tokens.len().saturating_sub(1);
    let pf_tps = res.prompt_tokens as f64 / res.prefill_secs.max(1e-6);
    let dec_tps = steps as f64 / res.decode_secs.max(1e-6);
    let vh = post.vram_hits - pre.vram_hits;
    let vm = post.vram_misses - pre.vram_misses;
    let dm = post.disk_misses - pre.disk_misses;
    let vhit_pct = 100.0 * vh as f64 / (vh + vm).max(1) as f64;
    let dgb = (post.disk_bytes - pre.disk_bytes) as f64 / 1e9;
    let spec_info = if res.spec_stats.rounds > 0 {
        format!(
            " | dflash={:.2} tok/rd ({}/{})",
            steps as f64 / res.spec_stats.rounds as f64,
            res.spec_stats.total_accepted,
            res.spec_stats.total_drafted
        )
    } else {
        String::new()
    };
    println!(
        "[serve {req_id} stream={streamed}] prompt={} (cached={} in {:.3}s, {:.1} tok/s) | gen={} ({:.2} tok/s){spec_info} | vram_hit={:.1}% disk_miss={dm} ({dgb:.2} GB)",
        res.prompt_tokens,
        res.prefix_cached_tokens,
        res.prefill_secs,
        pf_tps,
        res.generated_tokens.len(),
        dec_tps,
        vhit_pct
    );
}

// --- Zero-dep minimal JSON DOM for OpenAI request parsing & safe JSON string escaping ---

#[derive(Debug)]
enum JsonVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(HashMap<String, JsonVal>),
}

fn get_str<'a>(m: &'a HashMap<String, JsonVal>, k: &str) -> Option<&'a str> {
    match m.get(k) {
        Some(JsonVal::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn get_bool(m: &HashMap<String, JsonVal>, k: &str) -> Option<bool> {
    match m.get(k) {
        Some(JsonVal::Bool(b)) => Some(*b),
        _ => None,
    }
}

fn get_i64(m: &HashMap<String, JsonVal>, k: &str) -> Option<i64> {
    match m.get(k) {
        Some(JsonVal::Num(n)) => Some(*n as i64),
        _ => None,
    }
}

fn get_f64(m: &HashMap<String, JsonVal>, k: &str) -> Option<f64> {
    match m.get(k) {
        Some(JsonVal::Num(n)) => Some(*n),
        _ => None,
    }
}

fn extract_stops(m: &HashMap<String, JsonVal>) -> Vec<String> {
    match m.get("stop") {
        Some(JsonVal::Str(s)) => vec![s.clone()],
        Some(JsonVal::Arr(arr)) => arr
            .iter()
            .filter_map(|v| match v {
                JsonVal::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_message_content(v: Option<&JsonVal>) -> String {
    match v {
        Some(JsonVal::Str(s)) => s.clone(),
        Some(JsonVal::Arr(arr)) => {
            let mut out = String::new();
            for item in arr {
                if let JsonVal::Obj(o) = item {
                    if get_str(o, "type") == Some("text") {
                        if let Some(t) = get_str(o, "text") {
                            out.push_str(t);
                        }
                    }
                } else if let JsonVal::Str(s) = item {
                    out.push_str(s);
                }
            }
            out
        }
        _ => String::new(),
    }
}

pub fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn parse_json_value(raw: &[u8]) -> Result<JsonVal, String> {
    let mut pos = 0usize;
    parse_val(raw, &mut pos)
}

fn skip_ws(b: &[u8], pos: &mut usize) {
    while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
}

fn parse_val(b: &[u8], pos: &mut usize) -> Result<JsonVal, String> {
    skip_ws(b, pos);
    let c = *b.get(*pos).ok_or_else(|| "unexpected EOF".to_string())?;
    match c {
        b'{' => {
            *pos += 1;
            let mut m = HashMap::new();
            loop {
                skip_ws(b, pos);
                if b.get(*pos) == Some(&b'}') {
                    *pos += 1;
                    return Ok(JsonVal::Obj(m));
                }
                let k = parse_str(b, pos)?;
                skip_ws(b, pos);
                if b.get(*pos) != Some(&b':') {
                    return Err(format!("expected ':' at {}", *pos));
                }
                *pos += 1;
                let v = parse_val(b, pos)?;
                m.insert(k, v);
                skip_ws(b, pos);
                if b.get(*pos) == Some(&b',') {
                    *pos += 1;
                } else if b.get(*pos) != Some(&b'}') {
                    return Err(format!("expected ',' or '}}' at {}", *pos));
                }
            }
        }
        b'[' => {
            *pos += 1;
            let mut a = Vec::new();
            loop {
                skip_ws(b, pos);
                if b.get(*pos) == Some(&b']') {
                    *pos += 1;
                    return Ok(JsonVal::Arr(a));
                }
                a.push(parse_val(b, pos)?);
                skip_ws(b, pos);
                if b.get(*pos) == Some(&b',') {
                    *pos += 1;
                } else if b.get(*pos) != Some(&b']') {
                    return Err(format!("expected ',' or ']' at {}", *pos));
                }
            }
        }
        b'"' => Ok(JsonVal::Str(parse_str(b, pos)?)),
        b't' if b[*pos..].starts_with(b"true") => {
            *pos += 4;
            Ok(JsonVal::Bool(true))
        }
        b'f' if b[*pos..].starts_with(b"false") => {
            *pos += 5;
            Ok(JsonVal::Bool(false))
        }
        b'n' if b[*pos..].starts_with(b"null") => {
            *pos += 4;
            Ok(JsonVal::Null)
        }
        b'-' | b'0'..=b'9' => {
            let s0 = *pos;
            while *pos < b.len()
                && matches!(b[*pos], b'-' | b'+' | b'0'..=b'9' | b'.' | b'e' | b'E')
            {
                *pos += 1;
            }
            let s = std::str::from_utf8(&b[s0..*pos]).map_err(|e| e.to_string())?;
            let v: f64 = s.parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;
            Ok(JsonVal::Num(v))
        }
        other => Err(format!("unexpected byte '{}' at {}", other as char, *pos)),
    }
}

fn parse_str(b: &[u8], pos: &mut usize) -> Result<String, String> {
    if b.get(*pos) != Some(&b'"') {
        return Err(format!("expected '\"' at {}", *pos));
    }
    *pos += 1;
    let mut out = String::new();
    while *pos < b.len() {
        let c = b[*pos];
        *pos += 1;
        if c == b'"' {
            return Ok(out);
        }
        if c == b'\\' {
            let esc = *b.get(*pos).ok_or_else(|| "truncated escape".to_string())?;
            *pos += 1;
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\u{0008}'),
                b'f' => out.push('\u{000C}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    if *pos + 4 > b.len() {
                        return Err("truncated \\u".into());
                    }
                    let hex = std::str::from_utf8(&b[*pos..*pos + 4]).map_err(|e| e.to_string())?;
                    let u1 = u16::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                    *pos += 4;
                    if (0xD800..=0xDBFF).contains(&u1)
                        && *pos + 6 <= b.len()
                        && b[*pos] == b'\\'
                        && b[*pos + 1] == b'u'
                    {
                        *pos += 2;
                        let hex2 = std::str::from_utf8(&b[*pos..*pos + 4]).map_err(|e| e.to_string())?;
                        let u2 = u16::from_str_radix(hex2, 16).map_err(|e| e.to_string())?;
                        *pos += 4;
                        if (0xDC00..=0xDFFF).contains(&u2) {
                            let cp = 0x10000 + (((u1 as u32 - 0xD800) << 10) | (u2 as u32 - 0xDC00));
                            if let Some(ch) = char::from_u32(cp) {
                                out.push(ch);
                                continue;
                            }
                        }
                    }
                    if let Some(ch) = char::from_u32(u1 as u32) {
                        out.push(ch);
                    }
                }
                other => out.push(other as char),
            }
        } else if c < 0x80 {
            out.push(c as char);
        } else {
            let s0 = *pos - 1;
            let mut len = 1;
            while s0 + len < b.len() && (b[s0 + len] & 0xC0) == 0x80 {
                len += 1;
            }
            let slice = std::str::from_utf8(&b[s0..s0 + len]).map_err(|e| e.to_string())?;
            out.push_str(slice);
            *pos = s0 + len;
        }
    }
    Err("unterminated string".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_json_parser_and_thinking_split() {
        let body = r#"{"model":"glm-5.3-flash","stream":true,"max_tokens":64,"temperature":0.0,"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Hi \"Métis\"!\nLine 2"}]}"#.as_bytes();
        let v = parse_json_value(body).expect("parse openai json");
        let obj = match v {
            JsonVal::Obj(m) => m,
            _ => panic!("expected obj"),
        };
        assert_eq!(get_str(&obj, "model"), Some("glm-5.3-flash"));
        assert_eq!(get_bool(&obj, "stream"), Some(true));
        assert_eq!(get_i64(&obj, "max_tokens"), Some(64));
        assert_eq!(json_quote("a\"b\nc"), "\"a\\\"b\\nc\"");
        let (r, c) = split_thinking("step 1\nstep 2</think>\nFinal answer!", true);
        assert_eq!(r.as_deref(), Some("step 1\nstep 2"));
        assert_eq!(c, "Final answer!");
    }
}
