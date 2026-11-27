//! Zero-dependency Byte-Level BPE Tokenizer & GLM-5.3 Chat Template Formatter.
//!
//! Parses HuggingFace `tokenizer.json` directly (`model.vocab`, `model.merges`, `added_tokens`)
//! with zero external crates, implements the exact cl100k / GLM-5.3 pre-tokenizer split,
//! GPT-2 ByteLevel reversible byte<->unicode mapping, priority-rank BPE merging, streaming
//! incremental UTF-8 decoding, and the GLM-5.3 chat template (`[gMASK]<sop><|user|>...`).

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct BpeTokenizer {
    vocab: HashMap<String, i32>,
    id_to_bytes: Vec<Vec<u8>>,
    id_to_str: Vec<String>,
    is_special: Vec<bool>,
    merges: HashMap<(String, String), usize>,
    added_tokens: Vec<(String, i32, bool)>, // sorted longest-first for greedy matching
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

/// Build the standard GPT-2 / HuggingFace ByteLevel 256-entry reversible byte<->char table.
fn build_byte_level_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::with_capacity(256);
    let mut direct = [false; 256];
    for b in 33u8..=126u8 {
        direct[b as usize] = true;
    }
    for b in 161u8..=172u8 {
        direct[b as usize] = true;
    }
    for b in 174u8..=255u8 {
        direct[b as usize] = true;
    }
    let mut extra = 0u32;
    for b in 0usize..256 {
        let ch = if direct[b] {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
        b2c[b] = ch;
        c2b.insert(ch, b as u8);
    }
    (b2c, c2b)
}

/// Streaming incremental UTF-8 decoder that buffers partial multi-byte UTF-8 sequences across tokens.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// Push one generated token ID and return any complete valid UTF-8 text ready to stream now.
    pub fn push(&mut self, tok: &BpeTokenizer, id: i32, skip_special: bool) -> String {
        if id < 0 || (id as usize) >= tok.id_to_bytes.len() {
            return String::new();
        }
        let idx = id as usize;
        if skip_special && tok.is_special[idx] {
            return String::new();
        }
        self.pending.extend_from_slice(&tok.id_to_bytes[idx]);
        self.drain_valid_utf8(false)
    }

    /// Flush any remaining bytes at end-of-stream (lossy if truncated invalid UTF-8).
    pub fn flush(&mut self) -> String {
        self.drain_valid_utf8(true)
    }

    fn drain_valid_utf8(&mut self, force: bool) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_owned();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if e.error_len().is_none() && !force {
                    // Valid prefix followed by an incomplete trailing UTF-8 sequence — hold tail!
                    let out = std::str::from_utf8(&self.pending[..valid_up_to])
                        .unwrap()
                        .to_owned();
                    self.pending.drain(..valid_up_to);
                    out
                } else {
                    // Truly invalid sequence or forced EOF flush
                    let out = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    out
                }
            }
        }
    }
}

impl BpeTokenizer {
    /// Load a HuggingFace `tokenizer.json` file (either exact path or checkpoint directory containing it).
    pub fn from_path(path: &Path) -> io::Result<Self> {
        let json_path = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let raw = fs::read(&json_path)?;
        Self::from_json_bytes(&raw)
    }

    pub fn from_json_bytes(raw: &[u8]) -> io::Result<Self> {
        let (byte_to_char, char_to_byte) = build_byte_level_tables();
        let mut p = JsonCursor::new(raw);
        p.skip_ws();
        p.expect(b'{')?;

        let mut vocab: HashMap<String, i32> = HashMap::new();
        let mut merges: HashMap<(String, String), usize> = HashMap::new();
        let mut added_tokens: Vec<(String, i32, bool)> = Vec::new();
        let mut max_id: i32 = -1;

        loop {
            p.skip_ws();
            if p.peek() == Some(b'}') {
                break;
            }
            let key = p.parse_string()?;
            p.skip_ws();
            p.expect(b':')?;
            p.skip_ws();
            match key.as_str() {
                "added_tokens" => {
                    p.expect(b'[')?;
                    loop {
                        p.skip_ws();
                        if p.peek() == Some(b']') {
                            p.pos += 1;
                            break;
                        }
                        p.expect(b'{')?;
                        let mut id: i32 = -1;
                        let mut content = String::new();
                        let mut special = false;
                        loop {
                            p.skip_ws();
                            if p.peek() == Some(b'}') {
                                p.pos += 1;
                                break;
                            }
                            let ak = p.parse_string()?;
                            p.skip_ws();
                            p.expect(b':')?;
                            p.skip_ws();
                            match ak.as_str() {
                                "id" => id = p.parse_i64()? as i32,
                                "content" => content = p.parse_string()?,
                                "special" => special = p.parse_bool()?,
                                _ => p.skip_value()?,
                            }
                            p.skip_ws();
                            if p.peek() == Some(b',') {
                                p.pos += 1;
                            }
                        }
                        if id >= 0 && !content.is_empty() {
                            max_id = max_id.max(id);
                            added_tokens.push((content, id, special));
                        }
                        p.skip_ws();
                        if p.peek() == Some(b',') {
                            p.pos += 1;
                        }
                    }
                }
                "model" => {
                    p.expect(b'{')?;
                    loop {
                        p.skip_ws();
                        if p.peek() == Some(b'}') {
                            p.pos += 1;
                            break;
                        }
                        let mk = p.parse_string()?;
                        p.skip_ws();
                        p.expect(b':')?;
                        p.skip_ws();
                        match mk.as_str() {
                            "vocab" => {
                                p.expect(b'{')?;
                                vocab.reserve(160_000);
                                loop {
                                    p.skip_ws();
                                    if p.peek() == Some(b'}') {
                                        p.pos += 1;
                                        break;
                                    }
                                    let tok = p.parse_string()?;
                                    p.skip_ws();
                                    p.expect(b':')?;
                                    p.skip_ws();
                                    let id = p.parse_i64()? as i32;
                                    max_id = max_id.max(id);
                                    vocab.insert(tok, id);
                                    p.skip_ws();
                                    if p.peek() == Some(b',') {
                                        p.pos += 1;
                                    }
                                }
                            }
                            "merges" => {
                                p.expect(b'[')?;
                                merges.reserve(330_000);
                                let mut rank = 0usize;
                                loop {
                                    p.skip_ws();
                                    if p.peek() == Some(b']') {
                                        p.pos += 1;
                                        break;
                                    }
                                    if p.peek() == Some(b'[') {
                                        p.pos += 1;
                                        p.skip_ws();
                                        let a = p.parse_string()?;
                                        p.skip_ws();
                                        p.expect(b',')?;
                                        p.skip_ws();
                                        let b = p.parse_string()?;
                                        p.skip_ws();
                                        p.expect(b']')?;
                                        merges.insert((a, b), rank);
                                    } else {
                                        let pair = p.parse_string()?;
                                        if let Some((a, b)) = pair.split_once(' ') {
                                            merges.insert((a.to_owned(), b.to_owned()), rank);
                                        }
                                    }
                                    rank += 1;
                                    p.skip_ws();
                                    if p.peek() == Some(b',') {
                                        p.pos += 1;
                                    }
                                }
                            }
                            _ => p.skip_value()?,
                        }
                        p.skip_ws();
                        if p.peek() == Some(b',') {
                            p.pos += 1;
                        }
                    }
                }
                _ => p.skip_value()?,
            }
            p.skip_ws();
            if p.peek() == Some(b',') {
                p.pos += 1;
            }
        }

        if vocab.is_empty() || max_id < 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "tokenizer.json missing model.vocab"));
        }

        let size = (max_id as usize) + 1;
        let mut id_to_bytes = vec![Vec::new(); size];
        let mut id_to_str = vec![String::new(); size];
        let mut is_special = vec![false; size];

        for (tok, &id) in &vocab {
            if id >= 0 {
                let idx = id as usize;
                let mut bytes = Vec::with_capacity(tok.len());
                for ch in tok.chars() {
                    if let Some(&b) = char_to_byte.get(&ch) {
                        bytes.push(b);
                    } else {
                        let mut tmp = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                    }
                }
                id_to_bytes[idx] = bytes;
                id_to_str[idx] = tok.clone();
            }
        }
        for (content, id, special) in &added_tokens {
            if *id >= 0 {
                let idx = *id as usize;
                vocab.insert(content.clone(), *id);
                id_to_bytes[idx] = content.as_bytes().to_vec();
                id_to_str[idx] = content.clone();
                is_special[idx] = *special;
            }
        }
        // Longest added token first so e.g. `<|endoftext|>` matches before any shorter prefix
        added_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(Self {
            vocab,
            id_to_bytes,
            id_to_str,
            is_special,
            merges,
            added_tokens,
            byte_to_char,
            char_to_byte,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    pub fn token_id(&self, token: &str) -> Option<i32> {
        self.vocab.get(token).copied()
    }

    pub fn id_str(&self, id: i32) -> Option<&str> {
        if id >= 0 && (id as usize) < self.id_to_str.len() {
            Some(&self.id_to_str[id as usize])
        } else {
            None
        }
    }

    /// Encode text into token IDs, matching HuggingFace `tokenizer.encode(text, add_special_tokens=false)`
    /// while recognizing all `added_tokens` (e.g., `[gMASK]`, `<sop>`, `<|user|>`, `<think>`).
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 8);
        let mut pos = 0usize;
        let mut plain_start = 0usize;
        let bytes = text.as_bytes();

        while pos < bytes.len() {
            let mut matched: Option<(usize, i32)> = None;
            // Fast check: all GLM-5.3 added tokens begin with '<', '[', or '/'
            let b0 = bytes[pos];
            if b0 == b'<' || b0 == b'[' || b0 == b'/' {
                let tail = &text[pos..];
                for (tok_s, id, _) in &self.added_tokens {
                    if tail.starts_with(tok_s.as_str()) {
                        matched = Some((tok_s.len(), *id));
                        break;
                    }
                }
            }
            if let Some((len, id)) = matched {
                if plain_start < pos {
                    self.encode_plain(&text[plain_start..pos], &mut out);
                }
                out.push(id);
                pos += len;
                plain_start = pos;
            } else {
                // Advance one UTF-8 character
                let ch = text[pos..].chars().next().unwrap();
                pos += ch.len_utf8();
            }
        }
        if plain_start < text.len() {
            self.encode_plain(&text[plain_start..], &mut out);
        }
        out
    }

    /// Decode a slice of token IDs back into a UTF-8 string.
    pub fn decode(&self, ids: &[i32], skip_special: bool) -> String {
        let mut dec = StreamDecoder::new();
        let mut out = String::new();
        for &id in ids {
            out.push_str(&dec.push(self, id, skip_special));
        }
        out.push_str(&dec.flush());
        out
    }

    /// Format OpenAI-style chat messages using GLM-5.3's chat template (`chat_template.jinja`).
    pub fn format_glm53_chat(
        &self,
        messages: &[ChatMessage],
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> String {
        let mut s = String::with_capacity(256);
        s.push_str("[gMASK]<sop>");
        if let Some(eff) = reasoning_effort {
            s.push_str("<|system|>Reasoning Effort: ");
            s.push_str(eff);
        }
        for m in messages {
            match m.role.as_str() {
                "system" => {
                    s.push_str("<|system|>");
                    s.push_str(&m.content);
                }
                "user" => {
                    s.push_str("<|user|>");
                    s.push_str(&m.content);
                }
                "assistant" => {
                    s.push_str("<|assistant|><think></think>");
                    s.push_str(m.content.trim());
                }
                "tool" => {
                    s.push_str("<|observation|><tool_response>");
                    s.push_str(&m.content);
                    s.push_str("</tool_response>");
                }
                _ => {
                    s.push_str("<|user|>");
                    s.push_str(&m.content);
                }
            }
        }
        s.push_str("<|assistant|>");
        if enable_thinking {
            s.push_str("<think>");
        } else {
            s.push_str("<think></think>");
        }
        s
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<i32>) {
        let mut pos = 0usize;
        while pos < text.len() {
            let next = next_cl100k_piece(text, pos);
            let piece = &text[pos..next];
            self.encode_piece(piece, out);
            pos = next;
        }
    }

    fn encode_piece(&self, piece: &str, out: &mut Vec<i32>) {
        // Convert raw UTF-8 bytes to GPT-2 ByteLevel unicode characters
        let mut encoded = String::with_capacity(piece.len() * 2);
        for &b in piece.as_bytes() {
            encoded.push(self.byte_to_char[b as usize]);
        }
        // Fast path: whole piece already in vocab
        if let Some(&id) = self.vocab.get(&encoded) {
            out.push(id);
            return;
        }
        let mut parts: Vec<String> = encoded.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..parts.len() - 1 {
                if let Some(&rank) = self.merges.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            let right = parts.remove(best_idx + 1);
            parts[best_idx].push_str(&right);
        }
        for tok in parts {
            if let Some(&id) = self.vocab.get(&tok) {
                out.push(id);
            } else {
                // Fallback to individual byte tokens (always present in ByteLevel vocab)
                for ch in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        let s = self.byte_to_char[b as usize].to_string();
                        if let Some(&id) = self.vocab.get(&s) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }
}

/// Zero-dep exact implementation of the cl100k / GLM-5.3 pre-tokenizer regex:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
fn next_cl100k_piece(s: &str, start: usize) -> usize {
    let tail = &s[start..];
    let bytes = tail.as_bytes();
    if bytes.is_empty() {
        return start;
    }

    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if bytes[0] == b'\'' {
        if bytes.len() >= 3 {
            let a = bytes[1].to_ascii_lowercase();
            let b = bytes[2].to_ascii_lowercase();
            if matches!((a, b), (b'r', b'e') | (b'v', b'e') | (b'l', b'l')) {
                return start + 3;
            }
        }
        if bytes.len() >= 2 {
            let a = bytes[1].to_ascii_lowercase();
            if matches!(a, b's' | b't' | b'm' | b'd') {
                return start + 2;
            }
        }
    }

    let mut it = tail.char_indices().peekable();
    let (_, c0) = *it.peek().unwrap();

    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    if c0.is_alphabetic() {
        let mut end = 0usize;
        for (i, c) in tail.char_indices() {
            if c.is_alphabetic() {
                end = i + c.len_utf8();
            } else {
                break;
            }
        }
        return start + end;
    }
    if c0 != '\r' && c0 != '\n' && !c0.is_numeric() {
        let after0 = c0.len_utf8();
        if let Some(c1) = tail[after0..].chars().next() {
            if c1.is_alphabetic() {
                let mut end = after0;
                for (i, c) in tail[after0..].char_indices() {
                    if c.is_alphabetic() {
                        end = after0 + i + c.len_utf8();
                    } else {
                        break;
                    }
                }
                return start + end;
            }
        }
    }

    // 3. \p{N}{1,3}
    if c0.is_numeric() {
        let mut end = 0usize;
        let mut cnt = 0usize;
        for (i, c) in tail.char_indices() {
            if c.is_numeric() && cnt < 3 {
                end = i + c.len_utf8();
                cnt += 1;
            } else {
                break;
            }
        }
        return start + end;
    }

    // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    let is_punct = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
    let punct_start = if c0 == ' ' {
        if tail[1..].chars().next().is_some_and(is_punct) {
            Some(1usize)
        } else {
            None
        }
    } else if is_punct(c0) {
        Some(0usize)
    } else {
        None
    };
    if let Some(p0) = punct_start {
        let mut end = p0;
        let mut rest_start = p0;
        for (i, c) in tail[p0..].char_indices() {
            if is_punct(c) {
                end = p0 + i + c.len_utf8();
                rest_start = end;
            } else {
                break;
            }
        }
        for (i, c) in tail[rest_start..].char_indices() {
            if c == '\r' || c == '\n' {
                end = rest_start + i + c.len_utf8();
            } else {
                break;
            }
        }
        return start + end;
    }

    // 5. \s*[\r\n]+ | 6. \s+(?!\S) | 7. \s+
    if c0.is_whitespace() {
        let mut ws_end = 0usize;
        let mut last_nl_end = 0usize;
        let mut prev_ws_end = 0usize;
        for (i, c) in tail.char_indices() {
            if c.is_whitespace() {
                prev_ws_end = ws_end;
                ws_end = i + c.len_utf8();
                if c == '\r' || c == '\n' {
                    last_nl_end = ws_end;
                }
            } else {
                break;
            }
        }
        // Branch 5: \s*[\r\n]+
        if last_nl_end > 0 {
            return start + last_nl_end;
        }
        // Branch 6: \s+(?!\S)
        if ws_end == tail.len() {
            return start + ws_end;
        }
        if prev_ws_end > 0 {
            return start + prev_ws_end;
        }
        // Branch 7: \s+
        return start + ws_end;
    }

    start + c0.len_utf8()
}

// --- Zero-dependency streaming JSON parser for tokenizer.json ---

struct JsonCursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> JsonCursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    #[inline]
    fn skip_ws(&mut self) {
        while self.pos < self.b.len() && matches!(self.b[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn expect(&mut self, ch: u8) -> io::Result<()> {
        if self.peek() == Some(ch) {
            self.pos += 1;
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("JSON syntax error at byte {}: expected '{}'", self.pos, ch as char),
            ))
        }
    }

    fn parse_string(&mut self) -> io::Result<String> {
        self.expect(b'"')?;
        let start = self.pos;
        // Fast path: scan for unescaped quote
        while self.pos < self.b.len() {
            let c = self.b[self.pos];
            if c == b'"' {
                let s = std::str::from_utf8(&self.b[start..self.pos])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                    .to_owned();
                self.pos += 1;
                return Ok(s);
            }
            if c == b'\\' {
                break;
            }
            self.pos += 1;
        }
        // Slow path with escape sequences
        let mut out = String::from_utf8(self.b[start..self.pos].to_vec())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        while self.pos < self.b.len() {
            let c = self.b[self.pos];
            self.pos += 1;
            if c == b'"' {
                return Ok(out);
            }
            if c == b'\\' {
                if self.pos >= self.b.len() {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated escape"));
                }
                let esc = self.b[self.pos];
                self.pos += 1;
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
                        let u1 = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&u1)
                            && self.pos + 6 <= self.b.len()
                            && self.b[self.pos] == b'\\'
                            && self.b[self.pos + 1] == b'u'
                        {
                            self.pos += 2;
                            let u2 = self.parse_hex4()?;
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
                // Multi-byte UTF-8 sequence inside escaped string
                let s0 = self.pos - 1;
                let mut len = 1;
                while s0 + len < self.b.len() && (self.b[s0 + len] & 0xC0) == 0x80 {
                    len += 1;
                }
                let slice = std::str::from_utf8(&self.b[s0..s0 + len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                out.push_str(slice);
                self.pos = s0 + len;
            }
        }
        Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unterminated JSON string"))
    }

    fn parse_hex4(&mut self) -> io::Result<u16> {
        if self.pos + 4 > self.b.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated \\u escape"));
        }
        let s = std::str::from_utf8(&self.b[self.pos..self.pos + 4])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.pos += 4;
        u16::from_str_radix(s, 16).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    fn parse_i64(&mut self) -> io::Result<i64> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.pos])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        s.parse::<i64>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    fn parse_bool(&mut self) -> io::Result<bool> {
        if self.b[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(true)
        } else if self.b[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(false)
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "expected JSON boolean"))
        }
    }

    fn skip_value(&mut self) -> io::Result<()> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => {
                self.pos += 1;
                while self.pos < self.b.len() {
                    let c = self.b[self.pos];
                    self.pos += 1;
                    if c == b'\\' {
                        self.pos += 1;
                    } else if c == b'"' {
                        return Ok(());
                    }
                }
                Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unterminated string"))
            }
            Some(b'{') => {
                self.pos += 1;
                loop {
                    self.skip_ws();
                    if self.peek() == Some(b'}') {
                        self.pos += 1;
                        return Ok(());
                    }
                    self.skip_value()?;
                    self.skip_ws();
                    self.expect(b':')?;
                    self.skip_value()?;
                    self.skip_ws();
                    if self.peek() == Some(b',') {
                        self.pos += 1;
                    }
                }
            }
            Some(b'[') => {
                self.pos += 1;
                loop {
                    self.skip_ws();
                    if self.peek() == Some(b']') {
                        self.pos += 1;
                        return Ok(());
                    }
                    self.skip_value()?;
                    self.skip_ws();
                    if self.peek() == Some(b',') {
                        self.pos += 1;
                    }
                }
            }
            Some(_) => {
                while self.pos < self.b.len()
                    && !matches!(self.b[self.pos], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
                {
                    self.pos += 1;
                }
                Ok(())
            }
            None => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected EOF in JSON")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_encode_decode_roundtrip_and_streaming_utf8() {
        let sample = r#"{
            "added_tokens": [
                {"id": 10, "content": "<|user|>", "special": true},
                {"id": 11, "content": "<think>", "special": false}
            ],
            "model": {
                "vocab": {
                    "h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6,
                    "Ġ": 7, "ð": 8, "Ł": 9, "ĺ": 12, "Ģ": 13
                },
                "merges": [["h", "e"], ["l", "l"], ["he", "ll"], ["hell", "o"]]
            }
        }"#;
        // Manually insert 'hell' for merge chain
        let sample = sample.replace(r#""hello": 6,"#, r#""hell": 14, "hello": 6,"#);
        let tok = BpeTokenizer::from_json_bytes(sample.as_bytes()).unwrap();
        let ids = tok.encode("<|user|>hello<think>");
        assert_eq!(ids, vec![10, 6, 11]);
        assert_eq!(tok.decode(&ids, true), "hello<think>");
        assert_eq!(tok.decode(&ids, false), "<|user|>hello<think>");

        // Test streaming incremental 4-byte UTF-8 emoji (😀 = F0 9F 98 80 -> ð Ł ĺ Ģ in ByteLevel!)
        // Wait: verify exact byte mapping for F0 9F 98 80
        let mut dec = StreamDecoder::new();
        let emoji_bytes = "😀".as_bytes();
        assert_eq!(emoji_bytes.len(), 4);
        // Push bytes directly via helper check
        dec.pending.push(emoji_bytes[0]);
        assert_eq!(dec.drain_valid_utf8(false), "");
        dec.pending.push(emoji_bytes[1]);
        assert_eq!(dec.drain_valid_utf8(false), "");
        dec.pending.push(emoji_bytes[2]);
        assert_eq!(dec.drain_valid_utf8(false), "");
        dec.pending.push(emoji_bytes[3]);
        assert_eq!(dec.drain_valid_utf8(false), "😀");
    }

    #[test]
    fn real_glm53_tokenizer_parity_if_present() {
        let Ok(path_str) = std::env::var("M3TIS_TEST_TOKENIZER") else {
            return;
        };
        let p = Path::new(&path_str);
        if !p.exists() {
            return;
        }
        let tok = BpeTokenizer::from_path(p).expect("load real GLM-5.3 tokenizer.json");
        assert!(tok.vocab_size() >= 154856);
        // Verify exact roundtrip against HuggingFace reference token IDs + tricky code/unicode/emoji!
        let ref_ids = vec![154822, 154824, 154827, 198, 3838, 374, 264, 53882, 53503, 5267, 67121, 4749, 30];
        let decoded = tok.decode(&ref_ids, false);
        assert_eq!(tok.encode(&decoded), ref_ids, "must match HF reference IDs on decoded={decoded:?}");

        for sample in [
            "[gMASK]<sop><|user|>\nWhat is a Red River jig?\n<|assistant|><think></think>",
            "def fibonacci(n: int) -> int:\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)\n",
            "Hello, world! Don't stop; we're 1234567890 == 0xFF + 3.14159 🚀🔥 Métis!",
            "   leading spaces   \n\n\ttabs and   multiple   spaces  ",
        ] {
            let ids = tok.encode(sample);
            assert_eq!(tok.decode(&ids, false), sample, "exact decode mismatch on {sample:?}");
            let mut sd = StreamDecoder::new();
            let mut streamed = String::new();
            for &id in &ids {
                streamed.push_str(&sd.push(&tok, id, false));
            }
            streamed.push_str(&sd.flush());
            assert_eq!(streamed, sample, "streaming decode mismatch on {sample:?}");
        }
    }
}
