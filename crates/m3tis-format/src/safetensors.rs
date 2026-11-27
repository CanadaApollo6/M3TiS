//! Minimal zero-dependency Safetensors multi-shard index + reader.
//! Parses every `*.safetensors` header in a checkpoint directory in milliseconds without touching
//! payload pages, enabling direct pread of scales (`suh`/`svh`), routers, and dense tensors.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}

/// Ask the Linux kernel to drop cached file pages immediately so multi-gigabyte sequential
/// model/pack reads never bloat cgroup memory or trigger `systemd-oomd`.
pub fn drop_page_cache(file: &File) {
    #[cfg(target_os = "linux")]
    unsafe {
        const POSIX_FADV_DONTNEED: i32 = 4;
        let _ = posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = file;
}

#[derive(Debug, Clone)]
pub struct TensorLoc {
    pub shard_idx: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file_offset: u64,
    pub nbytes: usize,
}

pub struct SafetensorsIndex {
    pub shards: Vec<PathBuf>,
    files: Vec<File>,
    pub tensors: HashMap<String, TensorLoc>,
}

impl SafetensorsIndex {
    pub fn open_dir(dir: &Path) -> io::Result<Self> {
        let mut shards: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|x| x.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no *.safetensors found in {}", dir.display()),
            ));
        }

        let mut files = Vec::with_capacity(shards.len());
        let mut tensors = HashMap::new();
        for (shard_idx, path) in shards.iter().enumerate() {
            let mut f = File::open(path)?;
            let mut len_buf = [0u8; 8];
            f.read_exact(&mut len_buf)?;
            let hdr_len = u64::from_le_bytes(len_buf) as usize;
            if hdr_len > 256 * 1024 * 1024 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "safetensors header too large"));
            }
            let mut hdr_bytes = vec![0u8; hdr_len];
            f.read_exact(&mut hdr_bytes)?;
            let hdr_str = std::str::from_utf8(&hdr_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let data_base = 8u64 + hdr_len as u64;
            parse_safetensors_header(hdr_str, shard_idx, data_base, &mut tensors)?;
            files.push(f);
        }
        Ok(Self {
            shards,
            files,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> io::Result<&TensorLoc> {
        self.tensors.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("missing tensor: {name}"))
        })
    }

    pub fn read_into(&self, name: &str, dst: &mut [u8]) -> io::Result<()> {
        let loc = self.get(name)?;
        if dst.len() != loc.nbytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("size mismatch for {name}: dst={} tensor={}", dst.len(), loc.nbytes),
            ));
        }
        self.files[loc.shard_idx].read_exact_at(dst, loc.file_offset)
    }

    pub fn read_range(&self, name: &str, byte_offset: usize, dst: &mut [u8]) -> io::Result<()> {
        let loc = self.get(name)?;
        if byte_offset.saturating_add(dst.len()) > loc.nbytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "range out of bounds for {name}: off={byte_offset} len={} tensor={}",
                    dst.len(),
                    loc.nbytes
                ),
            ));
        }
        self.files[loc.shard_idx].read_exact_at(dst, loc.file_offset + byte_offset as u64)
    }

    pub fn read_vec(&self, name: &str) -> io::Result<Vec<u8>> {
        let loc = self.get(name)?;
        let mut out = vec![0u8; loc.nbytes];
        self.files[loc.shard_idx].read_exact_at(&mut out, loc.file_offset)?;
        Ok(out)
    }

    /// Release all read-once safetensors shard pages from the Linux page cache.
    pub fn release_page_cache(&self) {
        for f in &self.files {
            drop_page_cache(f);
        }
    }
}

impl Drop for SafetensorsIndex {
    fn drop(&mut self) {
        self.release_page_cache();
    }
}

// Fast minimal JSON scanner tailored specifically for safetensors header schema:
// { "<name>": { "dtype": "F16", "shape": [a, b], "data_offsets": [b0, b1] }, ... }
fn parse_safetensors_header(
    s: &str,
    shard_idx: usize,
    data_base: u64,
    out: &mut HashMap<String, TensorLoc>,
) -> io::Result<()> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    skip_ws(bytes, &mut i);
    expect_byte(bytes, &mut i, b'{')?;
    loop {
        skip_ws(bytes, &mut i);
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        let key = parse_json_str(bytes, &mut i)?;
        skip_ws(bytes, &mut i);
        expect_byte(bytes, &mut i, b':')?;
        skip_ws(bytes, &mut i);
        if key == "__metadata__" {
            skip_json_value(bytes, &mut i)?;
        } else {
            expect_byte(bytes, &mut i, b'{')?;
            let mut dtype = String::new();
            let mut shape = Vec::new();
            let mut off_start = 0u64;
            let mut off_end = 0u64;
            loop {
                skip_ws(bytes, &mut i);
                if i >= bytes.len() || bytes[i] == b'}' {
                    i += 1;
                    break;
                }
                let field = parse_json_str(bytes, &mut i)?;
                skip_ws(bytes, &mut i);
                expect_byte(bytes, &mut i, b':')?;
                skip_ws(bytes, &mut i);
                match field.as_str() {
                    "dtype" => dtype = parse_json_str(bytes, &mut i)?,
                    "shape" => {
                        let vals = parse_u64_array(bytes, &mut i)?;
                        shape = vals.into_iter().map(|v| v as usize).collect();
                    }
                    "data_offsets" => {
                        let vals = parse_u64_array(bytes, &mut i)?;
                        if vals.len() != 2 || vals[1] < vals[0] {
                            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad data_offsets"));
                        }
                        off_start = vals[0];
                        off_end = vals[1];
                    }
                    _ => skip_json_value(bytes, &mut i)?,
                }
                skip_ws(bytes, &mut i);
                if i < bytes.len() && bytes[i] == b',' {
                    i += 1;
                }
            }
            out.insert(
                key,
                TensorLoc {
                    shard_idx,
                    dtype,
                    shape,
                    file_offset: data_base + off_start,
                    nbytes: (off_end - off_start) as usize,
                },
            );
        }
        skip_ws(bytes, &mut i);
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        }
    }
    Ok(())
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\n' | b'\r' | b'\t') {
        *i += 1;
    }
}

fn expect_byte(b: &[u8], i: &mut usize, want: u8) -> io::Result<()> {
    if *i < b.len() && b[*i] == want {
        *i += 1;
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "malformed safetensors json"))
    }
}

fn parse_json_str(b: &[u8], i: &mut usize) -> io::Result<String> {
    expect_byte(b, i, b'"')?;
    let start = *i;
    let mut out = String::new();
    let mut escaped = false;
    while *i < b.len() {
        let ch = b[*i];
        *i += 1;
        if escaped {
            out.push(ch as char);
            escaped = false;
        } else if ch == b'\\' {
            if out.is_empty() && *i - 1 > start {
                out.push_str(std::str::from_utf8(&b[start..*i - 1]).unwrap_or(""));
            }
            escaped = true;
        } else if ch == b'"' {
            if out.is_empty() {
                return Ok(std::str::from_utf8(&b[start..*i - 1])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                    .to_string());
            }
            return Ok(out);
        } else if !out.is_empty() {
            out.push(ch as char);
        }
    }
    Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unterminated json string"))
}

fn parse_u64_array(b: &[u8], i: &mut usize) -> io::Result<Vec<u64>> {
    expect_byte(b, i, b'[')?;
    let mut out = Vec::new();
    loop {
        skip_ws(b, i);
        if *i < b.len() && b[*i] == b']' {
            *i += 1;
            break;
        }
        let mut n = 0u64;
        let mut any = false;
        while *i < b.len() && b[*i].is_ascii_digit() {
            n = n * 10 + (b[*i] - b'0') as u64;
            *i += 1;
            any = true;
        }
        if !any {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "expected int in array"));
        }
        out.push(n);
        skip_ws(b, i);
        if *i < b.len() && b[*i] == b',' {
            *i += 1;
        }
    }
    Ok(out)
}

fn skip_json_value(b: &[u8], i: &mut usize) -> io::Result<()> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof skipping json"));
    }
    match b[*i] {
        b'"' => {
            let _ = parse_json_str(b, i)?;
            Ok(())
        }
        b'{' | b'[' => {
            let mut depth = 0usize;
            while *i < b.len() {
                match b[*i] {
                    b'"' => {
                        let _ = parse_json_str(b, i)?;
                    }
                    b'{' | b'[' => {
                        depth += 1;
                        *i += 1;
                    }
                    b'}' | b']' => {
                        depth -= 1;
                        *i += 1;
                        if depth == 0 {
                            return Ok(());
                        }
                    }
                    _ => *i += 1,
                }
            }
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unclosed json container"))
        }
        _ => {
            while *i < b.len() && !matches!(b[*i], b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t') {
                *i += 1;
            }
            Ok(())
        }
    }
}
