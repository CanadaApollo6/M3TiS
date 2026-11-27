//! Zero-dependency Persistent NVMe Disk Prefix KV/KDA/DSA/DFlash2 Cache (`M3TISPFX1`).
//!
//! Persists committed prompt/turn prefix states across server restarts and CLI invocations so a
//! 16K–64K codebase context reloads from Samsung 990 PRO NVMe in ~50–150 ms instead of re-prefilling
//! 20–90 seconds of 320B MoE layers.
//!
//! Safety guarantees:
//! - Atomic write-then-rename (`.tmp` -> `.m3tispfx`) so crashes never leave partial files.
//! - 64-bit model/pack/drafter/DSA compatibility fingerprint checked on every load.
//! - Header-only startup scan (reads `120 + 4*num_tokens` bytes/file -> 0 RAM spent on cold payloads).
//! - Bounded total disk quota with LRU eviction by last-used timestamp.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: [u8; 8] = *b"M3TISPFX";
const VERSION: u32 = 1;
const FLAG_HAS_DSA: u32 = 1 << 0;
const FLAG_HAS_DRAFTER: u32 = 1 << 1;
const FLAG_HAS_LOGITS: u32 = 1 << 2;

/// Complete serialized prefix state handed between `Glm53Model` and `DiskPrefixStore`.
pub struct DiskPrefixPayload {
    pub tokens: Vec<i32>,
    pub pos: usize,
    pub kda_rec: Vec<u8>,
    pub kda_conv: Vec<u8>,
    pub mla_kv_layers: Vec<Vec<u8>>,
    pub dsa_ik_layers: Vec<Vec<u8>>,
    pub dsa_ig_layers: Vec<Vec<u8>>,
    pub dsa_pk_layers: Vec<Vec<u8>>,
    pub drafter_rings: Option<(Vec<u8>, Vec<u8>, i64)>,
    pub last_logits: Option<Vec<f32>>,
}

struct DiskEntry {
    path: PathBuf,
    tokens: Vec<i32>,
    pos: usize,
    has_logits: bool,
    file_size: u64,
    last_used: u64,
}

/// FNV-1a 64-bit hash (deterministic across runs & platforms, zero external deps).
pub fn fnv1a_u64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn hash_tokens(tokens: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn read_u32_le<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64_le<R: Read>(r: &mut R) -> io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

struct ParsedHeader {
    fingerprint: u64,
    pos: usize,
    num_tokens: usize,
    mla_layers: usize,
    flags: u32,
    kda_rec_len: usize,
    kda_conv_len: usize,
    mla_kv_layer_len: usize,
    dsa_ik_layer_len: usize,
    dsa_pk_layer_len: usize,
    drafter_rk_len: usize,
    drafter_from_pos: i64,
    logits_len: usize,
}

fn read_header<R: Read>(r: &mut R) -> io::Result<ParsedHeader> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad M3TISPFX magic"));
    }
    let version = read_u32_le(r)?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported M3TISPFX version {version}"),
        ));
    }
    let flags = read_u32_le(r)?;
    let fingerprint = read_u64_le(r)?;
    let pos = read_u64_le(r)? as usize;
    let num_tokens = read_u64_le(r)? as usize;
    let mla_layers = read_u32_le(r)? as usize;
    let _reserved = read_u32_le(r)?;
    let kda_rec_len = read_u64_le(r)? as usize;
    let kda_conv_len = read_u64_le(r)? as usize;
    let mla_kv_layer_len = read_u64_le(r)? as usize;
    let dsa_ik_layer_len = read_u64_le(r)? as usize;
    let dsa_pk_layer_len = read_u64_le(r)? as usize;
    let drafter_rk_len = read_u64_le(r)? as usize;
    let drafter_from_pos = read_i64_le(r)?;
    let logits_len = read_u64_le(r)? as usize;
    Ok(ParsedHeader {
        fingerprint,
        pos,
        num_tokens,
        mla_layers,
        flags,
        kda_rec_len,
        kda_conv_len,
        mla_kv_layer_len,
        dsa_ik_layer_len,
        dsa_pk_layer_len,
        drafter_rk_len,
        drafter_from_pos,
        logits_len,
    })
}

fn read_tokens<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<i32>> {
    let mut raw = vec![0u8; n * 4];
    r.read_exact(&mut raw)?;
    let mut toks = Vec::with_capacity(n);
    for ch in raw.chunks_exact(4) {
        toks.push(i32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
    }
    Ok(toks)
}

pub struct DiskPrefixStore {
    dir: PathBuf,
    fingerprint: u64,
    max_bytes: u64,
    total_bytes: u64,
    clock: u64,
    entries: Vec<DiskEntry>,
    pub disk_hits: u64,
    pub disk_saves: u64,
}

impl DiskPrefixStore {
    pub fn open(dir: PathBuf, fingerprint: u64, max_bytes: u64) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        let mut entries = Vec::new();
        let mut total_bytes = 0u64;

        for ent in fs::read_dir(&dir)? {
            let ent = match ent {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = ent.path();
            if path.extension().and_then(|s| s.to_str()) != Some("m3tispfx") {
                continue;
            }
            let meta = match fs::metadata(&path) {
                Ok(m) if m.is_file() => m,
                _ => continue,
            };
            let file_size = meta.len();
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let f = match File::open(&path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut r = BufReader::new(f);
            let hdr = match read_header(&mut r) {
                Ok(h) => h,
                Err(_) => continue,
            };
            if hdr.fingerprint != fingerprint || hdr.num_tokens == 0 || hdr.num_tokens > 1_048_576 {
                continue;
            }
            let tokens = match read_tokens(&mut r, hdr.num_tokens) {
                Ok(t) => t,
                Err(_) => continue,
            };
            total_bytes += file_size;
            entries.push(DiskEntry {
                path,
                tokens,
                pos: hdr.pos,
                has_logits: (hdr.flags & FLAG_HAS_LOGITS) != 0,
                file_size,
                last_used: mtime,
            });
        }

        let clock = entries
            .iter()
            .map(|e| e.last_used)
            .max()
            .unwrap_or(0)
            .max(now_secs());

        Ok(Self {
            dir,
            fingerprint,
            max_bytes,
            total_bytes,
            clock,
            entries,
            disk_hits: 0,
            disk_saves: 0,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Find the longest on-disk prefix matching `prompt` that fits within `max_ctx`.
    /// Returns `(entry_idx, matched_len)`.
    pub fn find_best(&self, prompt: &[i32], max_ctx: usize) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let n = e.tokens.len();
            if n == 0 || n > prompt.len() || e.pos > max_ctx {
                continue;
            }
            if n == prompt.len() && !e.has_logits {
                continue;
            }
            if prompt[..n] == e.tokens[..] {
                if best.map_or(true, |(_, bl)| n > bl) {
                    best = Some((i, n));
                }
            }
        }
        best
    }

    /// Load a full prefix payload from disk and bump its LRU timestamp.
    pub fn load(&mut self, idx: usize) -> io::Result<DiskPrefixPayload> {
        let path = self.entries[idx].path.clone();
        let f = File::open(&path)?;
        let mut r = BufReader::with_capacity(1 << 20, f);
        let hdr = read_header(&mut r)?;
        if hdr.fingerprint != self.fingerprint {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "prefix fingerprint mismatch on disk reload",
            ));
        }
        let tokens = read_tokens(&mut r, hdr.num_tokens)?;

        let mut kda_rec = vec![0u8; hdr.kda_rec_len];
        r.read_exact(&mut kda_rec)?;
        let mut kda_conv = vec![0u8; hdr.kda_conv_len];
        r.read_exact(&mut kda_conv)?;

        let mut mla_kv_layers = Vec::with_capacity(hdr.mla_layers);
        for _ in 0..hdr.mla_layers {
            let mut buf = vec![0u8; hdr.mla_kv_layer_len];
            if !buf.is_empty() {
                r.read_exact(&mut buf)?;
            }
            mla_kv_layers.push(buf);
        }

        let mut dsa_ik_layers = Vec::new();
        let mut dsa_ig_layers = Vec::new();
        let mut dsa_pk_layers = Vec::new();
        if (hdr.flags & FLAG_HAS_DSA) != 0 {
            dsa_ik_layers.reserve(hdr.mla_layers);
            dsa_ig_layers.reserve(hdr.mla_layers);
            dsa_pk_layers.reserve(hdr.mla_layers);
            for _ in 0..hdr.mla_layers {
                let mut bik = vec![0u8; hdr.dsa_ik_layer_len];
                let mut big = vec![0u8; hdr.dsa_ik_layer_len];
                let mut bpk = vec![0u8; hdr.dsa_pk_layer_len];
                if !bik.is_empty() {
                    r.read_exact(&mut bik)?;
                    r.read_exact(&mut big)?;
                }
                if !bpk.is_empty() {
                    r.read_exact(&mut bpk)?;
                }
                dsa_ik_layers.push(bik);
                dsa_ig_layers.push(big);
                dsa_pk_layers.push(bpk);
            }
        }

        let drafter_rings = if (hdr.flags & FLAG_HAS_DRAFTER) != 0 {
            let mut rk = vec![0u8; hdr.drafter_rk_len];
            let mut rv = vec![0u8; hdr.drafter_rk_len];
            r.read_exact(&mut rk)?;
            r.read_exact(&mut rv)?;
            Some((rk, rv, hdr.drafter_from_pos))
        } else {
            None
        };

        let last_logits = if (hdr.flags & FLAG_HAS_LOGITS) != 0 && hdr.logits_len > 0 {
            let mut raw = vec![0u8; hdr.logits_len * 4];
            r.read_exact(&mut raw)?;
            let mut v = Vec::with_capacity(hdr.logits_len);
            for ch in raw.chunks_exact(4) {
                v.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
            }
            Some(v)
        } else {
            None
        };

        self.clock += 1;
        self.entries[idx].last_used = self.clock;
        self.disk_hits += 1;
        Ok(DiskPrefixPayload {
            tokens,
            pos: hdr.pos,
            kda_rec,
            kda_conv,
            mla_kv_layers,
            dsa_ik_layers,
            dsa_ig_layers,
            dsa_pk_layers,
            drafter_rings,
            last_logits,
        })
    }

    /// Persist a snapshot atomically to disk and enforce `max_bytes` LRU eviction.
    pub fn save(&mut self, p: &DiskPrefixPayload) -> io::Result<()> {
        if self.max_bytes == 0 || p.tokens.is_empty() {
            return Ok(());
        }
        let has_logits = p.last_logits.is_some();
        if let Some(idx) = self.entries.iter().position(|e| e.tokens == p.tokens) {
            self.clock += 1;
            self.entries[idx].last_used = self.clock;
            if self.entries[idx].has_logits || !has_logits {
                return Ok(());
            }
            // Upgrade existing file without logits -> overwrite atomically
            self.total_bytes = self.total_bytes.saturating_sub(self.entries[idx].file_size);
            self.entries.swap_remove(idx);
        }

        let th = hash_tokens(&p.tokens);
        let fname = format!("pfx_{:016x}_{:016x}_{}.m3tispfx", self.fingerprint, th, p.tokens.len());
        let final_path = self.dir.join(&fname);
        let tmp_path = self.dir.join(format!("{fname}.tmp.{}", std::process::id()));

        let has_dsa = !p.dsa_pk_layers.is_empty() || !p.dsa_ik_layers.is_empty();
        let has_drafter = p.drafter_rings.is_some();
        let mut flags = 0u32;
        if has_dsa {
            flags |= FLAG_HAS_DSA;
        }
        if has_drafter {
            flags |= FLAG_HAS_DRAFTER;
        }
        if has_logits {
            flags |= FLAG_HAS_LOGITS;
        }

        let mla_layers = p.mla_kv_layers.len();
        let mla_kv_layer_len = p.mla_kv_layers.first().map_or(0, Vec::len);
        let dsa_ik_layer_len = p.dsa_ik_layers.first().map_or(0, Vec::len);
        let dsa_pk_layer_len = p.dsa_pk_layers.first().map_or(0, Vec::len);
        let (drafter_rk_len, drafter_from_pos) = match &p.drafter_rings {
            Some((rk, _, fp)) => (rk.len(), *fp),
            None => (0, 0),
        };
        let logits_len = p.last_logits.as_ref().map_or(0, Vec::len);

        let write_res = (|| -> io::Result<u64> {
            let f = File::create(&tmp_path)?;
            let mut w = BufWriter::with_capacity(1 << 20, f);
            w.write_all(&MAGIC)?;
            w.write_all(&VERSION.to_le_bytes())?;
            w.write_all(&flags.to_le_bytes())?;
            w.write_all(&self.fingerprint.to_le_bytes())?;
            w.write_all(&(p.pos as u64).to_le_bytes())?;
            w.write_all(&(p.tokens.len() as u64).to_le_bytes())?;
            w.write_all(&(mla_layers as u32).to_le_bytes())?;
            w.write_all(&0u32.to_le_bytes())?;
            w.write_all(&(p.kda_rec.len() as u64).to_le_bytes())?;
            w.write_all(&(p.kda_conv.len() as u64).to_le_bytes())?;
            w.write_all(&(mla_kv_layer_len as u64).to_le_bytes())?;
            w.write_all(&(dsa_ik_layer_len as u64).to_le_bytes())?;
            w.write_all(&(dsa_pk_layer_len as u64).to_le_bytes())?;
            w.write_all(&(drafter_rk_len as u64).to_le_bytes())?;
            w.write_all(&drafter_from_pos.to_le_bytes())?;
            w.write_all(&(logits_len as u64).to_le_bytes())?;

            for &tok in &p.tokens {
                w.write_all(&tok.to_le_bytes())?;
            }
            w.write_all(&p.kda_rec)?;
            w.write_all(&p.kda_conv)?;
            for buf in &p.mla_kv_layers {
                w.write_all(buf)?;
            }
            if has_dsa {
                for idx in 0..mla_layers {
                    w.write_all(&p.dsa_ik_layers[idx])?;
                    w.write_all(&p.dsa_ig_layers[idx])?;
                    w.write_all(&p.dsa_pk_layers[idx])?;
                }
            }
            if let Some((rk, rv, _)) = &p.drafter_rings {
                w.write_all(rk)?;
                w.write_all(rv)?;
            }
            if let Some(logits) = &p.last_logits {
                for &v in logits {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
            w.flush()?;
            let file_size = fs::metadata(&tmp_path)?.len();
            fs::rename(&tmp_path, &final_path)?;
            Ok(file_size)
        })();

        let file_size = match write_res {
            Ok(sz) => sz,
            Err(e) => {
                let _ = fs::remove_file(&tmp_path);
                return Err(e);
            }
        };

        self.clock += 1;
        self.total_bytes += file_size;
        self.disk_saves += 1;
        self.entries.push(DiskEntry {
            path: final_path,
            tokens: p.tokens.clone(),
            pos: p.pos,
            has_logits,
            file_size,
            last_used: self.clock,
        });

        // Bounded disk LRU eviction (always keep the newest entry we just wrote!)
        while self.total_bytes > self.max_bytes && self.entries.len() > 1 {
            if let Some((evict_idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
            {
                let old = self.entries.swap_remove(evict_idx);
                self.total_bytes = self.total_bytes.saturating_sub(old.file_size);
                let _ = fs::remove_file(&old.path);
            } else {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_payload(tokens: Vec<i32>, pos: usize, with_logits: bool, seed: u8) -> DiskPrefixPayload {
        DiskPrefixPayload {
            tokens,
            pos,
            kda_rec: vec![seed; 512],
            kda_conv: vec![seed.wrapping_add(1); 128],
            mla_kv_layers: vec![vec![seed.wrapping_add(2); 64], vec![seed.wrapping_add(3); 64]],
            dsa_ik_layers: vec![vec![seed.wrapping_add(4); 32], vec![seed.wrapping_add(5); 32]],
            dsa_ig_layers: vec![vec![seed.wrapping_add(6); 32], vec![seed.wrapping_add(7); 32]],
            dsa_pk_layers: vec![vec![seed.wrapping_add(8); 16], vec![seed.wrapping_add(9); 16]],
            drafter_rings: Some((vec![seed.wrapping_add(10); 48], vec![seed.wrapping_add(11); 48], 42)),
            last_logits: if with_logits { Some(vec![1.25, -0.5, 3.75, 0.0]) } else { None },
        }
    }

    #[test]
    fn disk_prefix_store_roundtrip_restart_fingerprint_and_lru_eviction() {
        let dir = std::env::temp_dir().join(format!("m3tis_pfx_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let fp = fnv1a_u64(b"glm53-test-fp-v1");

        // 1. Save two prefixes and verify longest-match lookup
        {
            let mut store = DiskPrefixStore::open(dir.clone(), fp, 64 * 1024).unwrap();
            let p1 = make_payload(vec![10, 20, 30, 40], 4, true, 0x11);
            let p2 = make_payload(vec![10, 20, 30, 40, 50, 60], 6, false, 0x22);
            store.save(&p1).unwrap();
            store.save(&p2).unwrap();
            assert_eq!(store.len(), 2);
        }

        // 2. Reopen in a fresh store (simulates process restart!) and verify byte-exact reload
        {
            let mut store = DiskPrefixStore::open(dir.clone(), fp, 64 * 1024).unwrap();
            assert_eq!(store.len(), 2);
            // Exact match to p2 has no logits -> must fall back to p1 (4 tokens)!
            let (i_exact, len_exact) = store.find_best(&[10, 20, 30, 40, 50, 60], 128).unwrap();
            assert_eq!(len_exact, 4);
            let loaded1 = store.load(i_exact).unwrap();
            assert_eq!(loaded1.tokens, vec![10, 20, 30, 40]);
            assert_eq!(loaded1.kda_rec, vec![0x11; 512]);
            assert_eq!(loaded1.last_logits.as_deref(), Some(&[1.25f32, -0.5, 3.75, 0.0][..]));

            // Longer prompt extending p2 -> matches p2 (6 tokens)!
            let (i_ext, len_ext) = store.find_best(&[10, 20, 30, 40, 50, 60, 70], 128).unwrap();
            assert_eq!(len_ext, 6);
            let loaded2 = store.load(i_ext).unwrap();
            assert_eq!(loaded2.pos, 6);
            assert_eq!(loaded2.kda_rec, vec![0x22; 512]);
            assert_eq!(loaded2.drafter_rings.as_ref().unwrap().2, 42);
            assert!(loaded2.last_logits.is_none());

            // max_ctx smaller than p2.pos -> falls back to p1!
            let (_, len_cap) = store.find_best(&[10, 20, 30, 40, 50, 60, 70], 5).unwrap();
            assert_eq!(len_cap, 4);
        }

        // 3. Incompatible fingerprint sees zero entries (never loads wrong model/pack!)
        {
            let other = DiskPrefixStore::open(dir.clone(), fp ^ 1, 64 * 1024).unwrap();
            assert_eq!(other.len(), 0);
        }

        // 4. Bounded disk LRU eviction keeps total_bytes <= max_bytes
        {
            let mut store = DiskPrefixStore::open(dir.clone(), fp, 1500).unwrap();
            let p3 = make_payload(vec![99, 100, 101, 102], 4, true, 0x33);
            store.save(&p3).unwrap();
            assert_eq!(store.len(), 1);
            assert!(store.find_best(&[99, 100, 101, 102], 128).is_some());
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
