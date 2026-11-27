//! 3-Tier Residency Manager for M3TiS: VRAM <- Scan-Resistant 2Q/LFU Pinned Host RAM <- Dual-Ring NVMe.
//!
//! Key algorithmic upgrade over plain CLOCK:
//! - Tier-2 Pinned Host RAM splits slots logically (zero data copying) into:
//!   - **A1 Probation (`20%` of slots):** all cold misses, prefill scans, and speculative prefetches
//!     enter here first. A 747-token prefill scan cycles strictly through A1 and NEVER evicts the
//!     80% protected hot super-expert working set!
//!   - **Am Protected Hot Pool (`80%` of slots):** experts promote `A1 -> Am` on their 2nd demand hit
//!     (or optional Zipf seeding). Eviction inside `Am` uses frequency-decaying LFU-CLOCK.
//! - Every slot has one owner (`TieredExpertPool` on the reactor thread) and explicit `Ready` vs
//!   `Reading` state so in-flight speculative prefetches are seamlessly joined by `demand()` with
//!   ZERO duplicate NVMe reads.

use m3tis_cuda::{CudaContext, CuDevicePtr, DeviceBuf, Event, PinnedHostBuf, RouterRemapKernels, Stream};
use m3tis_format::PackCatalog;
use m3tis_io::{DirectFile, DualRingIo, ReadOp};
use std::collections::VecDeque;
use std::io;
use std::path::Path;
use std::sync::Arc;

pub mod cpu_exl3;
pub mod layer_stream;
pub mod scales;

pub const ABSENT: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Scan-resistant 2Q + LFU-CLOCK (20% A1 probation, 80% Am protected hot set).
    TwoQLfu,
    /// Classic single-queue CLOCK (kept for A/B benchmarking against 2Q/LFU).
    PlainClock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueueKind {
    None,
    Probation, // A1
    Protected, // Am
}

#[derive(Debug, Clone, Copy)]
pub struct Ticket {
    pub key: usize,
    pub slot: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    key: Option<usize>,
    generation: u64,
    reading: bool,
    ready: bool,
    leased: bool,
    freq: u8, // saturating demand frequency (0 for untouched prefetch, 1 on 1st demand, >=2 -> Am)
    queue: QueueKind,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            key: None,
            generation: 0,
            reading: false,
            ready: false,
            leased: false,
            freq: 0,
            queue: QueueKind::None,
        }
    }
}

/// Bounded slot cache supporting both Scan-Resistant W-TinyLFU 2Q and Plain CLOCK, with explicit
/// `Ready` / `Reading` / `Leased` invariants, persistent per-key frequency history (`hist_freq`),
/// and in-flight join.
pub struct SlotCache {
    policy: EvictionPolicy,
    map: Vec<i32>,      // [num_keys] -> ready slot or ABSENT
    inflight: Vec<i32>, // [num_keys] -> reading slot or ABSENT
    slots: Vec<Slot>,
    demand_mark: Vec<bool>,
    current: Vec<usize>,
    // Persistent W-TinyLFU frequency history across ALL keys (tiny: 24 KiB for 12,096 experts!).
    // Survives A1 evictions so cyclic wide-verify scans (`rows = 4..8`) still promote super-experts into Am!
    hist_freq: Vec<u16>,
    hist_ops: u64,
    hist_decay_period: u64,
    // 2Q queues (slot indices)
    a1_fifo: VecDeque<usize>,
    am_count: usize,
    am_cap: usize,
    hand: usize,
    pub hits: u64,
    pub evictions: u64,
    pub promotions_am: u64,
}

impl SlotCache {
    pub fn new(num_keys: usize, capacity: usize, policy: EvictionPolicy) -> Self {
        assert!(num_keys > 0 && capacity > 0);
        // 15% A1 probation scratch (covers >3x max single-layer active union) + 85% Am protected super-experts.
        let a1_target = ((capacity as f64) * 0.15).round().max(1.0) as usize;
        let am_cap = capacity.saturating_sub(a1_target);
        let hist_decay_period = (capacity as u64).saturating_mul(32).max(2048);
        Self {
            policy,
            map: vec![ABSENT; num_keys],
            inflight: vec![ABSENT; num_keys],
            slots: vec![Slot::default(); capacity],
            demand_mark: vec![false; num_keys],
            current: Vec::with_capacity(capacity.min(1024)),
            hist_freq: vec![0u16; num_keys],
            hist_ops: 0,
            hist_decay_period,
            a1_fifo: VecDeque::with_capacity(capacity),
            am_count: 0,
            am_cap,
            hand: 0,
            hits: 0,
            evictions: 0,
            promotions_am: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn map_slice(&self) -> &[i32] {
        &self.map
    }

    #[inline]
    pub fn peek(&self, key: usize) -> Option<usize> {
        let s = self.map[key];
        if s < 0 {
            return None;
        }
        let idx = s as usize;
        debug_assert!(self.slots[idx].ready && !self.slots[idx].reading && self.slots[idx].key == Some(key));
        Some(idx)
    }

    #[inline]
    pub fn pending(&self, key: usize) -> Option<Ticket> {
        let s = self.inflight[key];
        if s < 0 {
            return None;
        }
        let idx = s as usize;
        let slot = &self.slots[idx];
        debug_assert!(slot.reading && slot.key == Some(key));
        Some(Ticket {
            key,
            slot: idx,
            generation: slot.generation,
        })
    }

    /// Release leases from the previous layer's active union.
    pub fn release(&mut self) {
        for &key in &self.current {
            self.demand_mark[key] = false;
            let idx = if self.map[key] >= 0 {
                Some(self.map[key] as usize)
            } else if self.inflight[key] >= 0 {
                Some(self.inflight[key] as usize)
            } else {
                None
            };
            if let Some(i) = idx {
                self.slots[i].leased = false;
            }
        }
        self.current.clear();
    }

    /// Protect every key in the incoming active union BEFORE selecting any eviction victim.
    /// Also promotes A1 -> Am on 2nd demand hit under `TwoQLfu`, and leases in-flight prefetches!
    pub fn protect(&mut self, keys: &[usize]) -> io::Result<()> {
        debug_assert!(self.current.is_empty(), "must call release() before protect()");
        for &key in keys {
            if key >= self.map.len() {
                self.release();
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "expert key out of bounds"));
            }
            if !self.demand_mark[key] {
                if self.current.len() >= self.slots.len() {
                    self.release();
                    return Err(io::Error::new(
                        io::ErrorKind::OutOfMemory,
                        "active expert union exceeds cache capacity",
                    ));
                }
                self.demand_mark[key] = true;
                self.current.push(key);
            }
        }
        for i in 0..self.current.len() {
            let key = self.current[i];
            let hf = self.record_demand_freq(key);
            let slot_idx = if self.map[key] >= 0 {
                Some(self.map[key] as usize)
            } else if self.inflight[key] >= 0 {
                Some(self.inflight[key] as usize)
            } else {
                None
            };
            if let Some(idx) = slot_idx {
                self.slots[idx].leased = true;
                self.slots[idx].freq = self.slots[idx].freq.saturating_add(1);
                self.hits += 1;
                if self.policy == EvictionPolicy::TwoQLfu
                    && self.slots[idx].queue == QueueKind::Probation
                    && hf >= 2
                {
                    self.try_promote_to_protected(idx, hf);
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn record_demand_freq(&mut self, key: usize) -> u16 {
        let v = self.hist_freq[key].saturating_add(1);
        self.hist_freq[key] = v;
        self.hist_ops += 1;
        if self.hist_ops >= self.hist_decay_period {
            self.hist_ops = 0;
            for f in &mut self.hist_freq {
                *f >>= 1;
            }
        }
        self.hist_freq[key]
    }

    /// Sample up to `SAMPLE_K` unleased Protected slots starting from `self.hand` and return the
    /// coldest one `(slot_idx, hist_freq)`. Classic Sampled-LFU / W-TinyLFU admission test!
    fn sample_coldest_protected(&mut self) -> Option<(usize, u16)> {
        const SAMPLE_K: usize = 24;
        let cap = self.slots.len();
        let mut best: Option<(usize, u16)> = None;
        let mut sampled = 0usize;
        for _ in 0..cap {
            let i = self.hand;
            self.hand = (i + 1) % cap;
            let s = &self.slots[i];
            if s.queue != QueueKind::Protected || s.reading || s.leased {
                continue;
            }
            let f = s.key.map(|k| self.hist_freq[k]).unwrap_or(0);
            if best.map_or(true, |(_, bf)| f < bf) {
                best = Some((i, f));
                if f == 0 {
                    break;
                }
            }
            sampled += 1;
            if sampled >= SAMPLE_K {
                break;
            }
        }
        best
    }

    /// W-TinyLFU promotion gate: admit `slot_idx` (`cand_freq >= 2`) into `Am` if `Am` has room OR
    /// if `cand_freq` strictly beats the coldest sampled Protected victim (demoting that victim to `A1`).
    fn try_promote_to_protected(&mut self, slot_idx: usize, cand_freq: u16) -> bool {
        if self.am_cap == 0 {
            return false;
        }
        if self.am_count >= self.am_cap {
            if let Some((demote, victim_freq)) = self.sample_coldest_protected() {
                if cand_freq <= victim_freq {
                    return false;
                }
                self.slots[demote].queue = QueueKind::Probation;
                self.am_count -= 1;
                self.a1_fifo.push_back(demote);
            } else {
                return false;
            }
        }
        self.slots[slot_idx].queue = QueueKind::Protected;
        self.am_count += 1;
        self.promotions_am += 1;
        true
    }

    #[inline]
    pub fn current_keys(&self) -> &[usize] {
        &self.current
    }

    /// Clear all cached slot entries while preserving `hist_freq` (Zipf prior) and counters.
    /// Used between Step-1 VRAM prewarm and Step-2 Exclusive Tier-2 RAM prewarm so Step-1 VRAM
    /// duplicates never lock `A_m` slots in Tier-2 RAM!
    pub fn clear_entries_keep_freq(&mut self) {
        for s in &mut self.slots {
            *s = Slot::default();
        }
        self.map.fill(ABSENT);
        self.inflight.fill(ABSENT);
        self.demand_mark.fill(false);
        self.current.clear();
        self.a1_fifo.clear();
        self.am_count = 0;
        self.hand = 0;
    }

    /// Reserve a slot for `key`. Returns `Ok(None)` if `key` is already `Ready` OR already
    /// in-flight (`Reading`) — preventing duplicate NVMe reads!
    pub fn reserve(&mut self, key: usize, is_prefetch: bool) -> io::Result<Option<Ticket>> {
        Ok(self.reserve_evict(key, is_prefetch)?.map(|(t, _)| t))
    }

    /// Same as `reserve`, also returning any evicted `old_key` so outer split caches stay synchronized.
    pub fn reserve_evict(&mut self, key: usize, is_prefetch: bool) -> io::Result<Option<(Ticket, Option<usize>)>> {
        if key >= self.map.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid key"));
        }
        if !is_prefetch && !self.demand_mark[key] {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "unprotected demand"));
        }
        if self.peek(key).is_some() || self.pending(key).is_some() {
            return Ok(None);
        }

        let victim = self.select_victim(is_prefetch).ok_or_else(|| {
            io::Error::new(io::ErrorKind::WouldBlock, "no evictable cache slot available")
        })?;

        let old_key = self.slots[victim].key;
        let old_queue = self.slots[victim].queue;
        if let Some(ok) = old_key {
            self.map[ok] = ABSENT;
            self.evictions += 1;
        }
        if old_queue == QueueKind::Protected {
            self.am_count = self.am_count.saturating_sub(1);
        }

        let gen_next = self.slots[victim].generation.wrapping_add(1);
        let init_freq = if is_prefetch { 0 } else { 1 };
        let queue = match self.policy {
            EvictionPolicy::TwoQLfu => QueueKind::Probation,
            EvictionPolicy::PlainClock => QueueKind::None,
        };
        self.slots[victim] = Slot {
            key: Some(key),
            generation: gen_next,
            reading: true,
            ready: false,
            leased: self.demand_mark[key],
            freq: init_freq,
            queue,
        };
        if queue == QueueKind::Probation {
            let hf = self.hist_freq[key];
            if !is_prefetch && hf >= 2 && self.try_promote_to_protected(victim, hf) {
                // Directly admitted to Am via W-TinyLFU persistent frequency history!
            } else {
                self.a1_fifo.push_back(victim);
            }
        }
        self.inflight[key] = victim as i32;
        Ok(Some((
            Ticket {
                key,
                slot: victim,
                generation: gen_next,
            },
            old_key,
        )))
    }

    fn select_victim(&mut self, is_prefetch: bool) -> Option<usize> {
        let cap = self.slots.len();
        // 1. Always grab any empty free slot first.
        for _ in 0..cap {
            let i = self.hand;
            self.hand = (i + 1) % cap;
            let s = &self.slots[i];
            if s.key.is_none() && !s.reading && !s.leased {
                return Some(i);
            }
        }

        match self.policy {
            EvictionPolicy::PlainClock => {
                let sweeps = if is_prefetch { 1 } else { 4 };
                for _ in 0..(cap * sweeps) {
                    let i = self.hand;
                    self.hand = (i + 1) % cap;
                    let s = &mut self.slots[i];
                    if s.reading || s.leased {
                        continue;
                    }
                    if s.freq > 0 {
                        if !is_prefetch {
                            s.freq -= 1;
                        }
                        continue;
                    }
                    return Some(i);
                }
                None
            }
            EvictionPolicy::TwoQLfu => {
                // 2. Evict strictly from A1 Probation FIFO first! Keeps Am hot super-experts 100% shielded
                // against one-shot prefill scans and speculative prefetch misses.
                let qlen = self.a1_fifo.len();
                for _ in 0..qlen {
                    if let Some(idx) = self.a1_fifo.pop_front() {
                        let s = &self.slots[idx];
                        if s.queue != QueueKind::Probation {
                            // Already promoted to Am; drop stale FIFO handle.
                            continue;
                        }
                        if s.reading || s.leased {
                            self.a1_fifo.push_back(idx);
                            continue;
                        }
                        return Some(idx);
                    }
                }
                // 3. Fallback: only if A1 has no evictable slot (and never let speculative prefetch evict hot Am!)
                if is_prefetch {
                    return None;
                }
                self.pick_protected_victim(false)
            }
        }
    }

    fn pick_protected_victim(&mut self, only_protected: bool) -> Option<usize> {
        let cap = self.slots.len();
        // LFU-CLOCK over Protected slots: decay freq by half per pass so hot super-experts survive.
        for pass in 0..9 {
            for _ in 0..cap {
                let i = self.hand;
                self.hand = (i + 1) % cap;
                let s = &mut self.slots[i];
                if s.reading || s.leased {
                    continue;
                }
                if only_protected && s.queue != QueueKind::Protected {
                    continue;
                }
                if s.freq == 0 || pass == 8 {
                    return Some(i);
                }
                s.freq >>= 1;
            }
        }
        None
    }

    pub fn publish(&mut self, ticket: Ticket) -> io::Result<()> {
        let s = &mut self.slots[ticket.slot];
        if s.key != Some(ticket.key) || s.generation != ticket.generation || !s.reading {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "stale cache ticket"));
        }
        self.inflight[ticket.key] = ABSENT;
        s.reading = false;
        s.ready = true;
        self.map[ticket.key] = ticket.slot as i32;
        Ok(())
    }

    pub fn abort(&mut self, ticket: Ticket) -> io::Result<()> {
        let s = &mut self.slots[ticket.slot];
        if s.key != Some(ticket.key) || s.generation != ticket.generation || !s.reading {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "stale cache ticket"));
        }
        self.inflight[ticket.key] = ABSENT;
        if s.queue == QueueKind::Protected {
            self.am_count = self.am_count.saturating_sub(1);
        }
        *s = Slot {
            generation: s.generation,
            ..Slot::default()
        };
        Ok(())
    }

    pub fn protected_count(&self) -> usize {
        self.am_count
    }

    /// Seed persistent W-TinyLFU `hist_freq` prior for keys matching `mask` (or all keys if `None`).
    pub fn seed_zipf_prior_filtered(&mut self, ranks: &[u32], mask: Option<&[bool]>, want_mask: bool) {
        if self.policy != EvictionPolicy::TwoQLfu || ranks.len() != self.hist_freq.len() {
            return;
        }
        let mut eligible: Vec<(u32, usize)> = ranks
            .iter()
            .enumerate()
            .filter(|&(k, _)| mask.map_or(true, |m| m[k] == want_mask))
            .map(|(k, &r)| (r, k))
            .collect();
        eligible.sort_unstable_by_key(|&(r, k)| (r, k));
        let take = self.am_cap.min(eligible.len());
        for &(_, key) in &eligible[..take] {
            self.hist_freq[key] = self.hist_freq[key].max(2);
        }
    }

    pub fn seed_zipf_prior(&mut self, ranks: &[u32]) {
        self.seed_zipf_prior_filtered(ranks, None, false);
    }

    #[inline]
    pub fn hist_freq(&self, key: usize) -> u16 {
        self.hist_freq[key]
    }

    /// Promote a pre-warmed resident key into Protected (`Am`) at soft baseline frequency `2`
    /// (NEVER inflating to 4!) so active session experts touched >=3 times immediately win `Am` admission!
    pub fn promote_prewarm_soft(&mut self, key: usize) {
        if key >= self.map.len() {
            return;
        }
        let slot = self.map[key];
        if slot < 0 {
            return;
        }
        let sidx = slot as usize;
        if self.hist_freq[key] < 2 {
            self.hist_freq[key] = 2;
        }
        let s = &mut self.slots[sidx];
        s.freq = s.freq.max(1);
        if self.policy == EvictionPolicy::TwoQLfu && s.queue == QueueKind::Probation {
            self.try_promote_to_protected(sidx, 2);
        }
    }

    /// Record a VRAM hit without acquiring a lease (used by zero-sync WarpZeroSync to keep W-TinyLFU hot!).
    pub fn touch_hit(&mut self, key: usize) {
        if key >= self.map.len() {
            return;
        }
        let slot = self.map[key];
        if slot < 0 {
            return;
        }
        let sidx = slot as usize;
        self.hist_freq[key] = self.hist_freq[key].saturating_add(1).min(255);
        let hf = self.hist_freq[key];
        let s = &mut self.slots[sidx];
        s.freq = s.freq.saturating_add(1).min(255);
        if self.policy == EvictionPolicy::TwoQLfu && s.queue == QueueKind::Probation && hf >= 2 {
            self.try_promote_to_protected(sidx, hf);
        }
    }

    #[cfg(debug_assertions)]
    pub fn assert_invariants(&self) {
        let mut ready_cnt = 0usize;
        let mut reading_cnt = 0usize;
        for (idx, s) in self.slots.iter().enumerate() {
            assert!(!(s.ready && s.reading));
            if let Some(k) = s.key {
                if s.ready {
                    assert_eq!(self.map[k], idx as i32);
                    assert_eq!(self.inflight[k], ABSENT);
                    ready_cnt += 1;
                } else if s.reading {
                    assert_eq!(self.inflight[k], idx as i32);
                    assert_eq!(self.map[k], ABSENT);
                    reading_cnt += 1;
                }
            }
        }
        let map_cnt = self.map.iter().filter(|&&v| v >= 0).count();
        let inf_cnt = self.inflight.iter().filter(|&&v| v >= 0).count();
        assert_eq!(ready_cnt, map_cnt);
        assert_eq!(reading_cnt, inf_cnt);
    }
}

/// Dual-partition compact slot cache for mixed-bitwidth `.m3tis` v2 packs (`hot` 3b slots `0..hot_cap`
/// + `cold` 2b slots `hot_cap..hot_cap+cold_cap`), eliminating 33% dead padding on 2b experts AND
/// physically isolating hot super-experts from cold tail evictions! For uniform v1 packs, `cold` is `None`.
pub struct SplitSlotCache {
    hot: SlotCache,
    cold: Option<SlotCache>,
    is_cold: Vec<bool>,
    hot_cap: usize,
    map: Vec<i32>,
    current: Vec<usize>,
    hot_tmp: Vec<usize>,
    cold_tmp: Vec<usize>,
}

impl SplitSlotCache {
    pub fn new(
        num_keys: usize,
        hot_cap: usize,
        cold_cap: usize,
        is_cold: Vec<bool>,
        policy: EvictionPolicy,
    ) -> Self {
        assert_eq!(is_cold.len(), num_keys);
        let hot = SlotCache::new(num_keys, hot_cap, policy);
        let cold = if cold_cap > 0 {
            Some(SlotCache::new(num_keys, cold_cap, policy))
        } else {
            None
        };
        Self {
            hot,
            cold,
            is_cold,
            hot_cap,
            map: vec![ABSENT; num_keys],
            current: Vec::with_capacity((hot_cap + cold_cap).min(1024)),
            hot_tmp: Vec::with_capacity(hot_cap.min(512)),
            cold_tmp: Vec::with_capacity(cold_cap.min(512)),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.hot_cap + self.cold.as_ref().map_or(0, |c| c.capacity())
    }

    #[inline]
    pub fn hot_capacity(&self) -> usize {
        self.hot_cap
    }

    #[inline]
    pub fn cold_capacity(&self) -> usize {
        self.cold.as_ref().map_or(0, |c| c.capacity())
    }

    #[inline]
    pub fn map_slice(&self) -> &[i32] {
        &self.map
    }

    #[inline]
    pub fn current_keys(&self) -> &[usize] {
        &self.current
    }

    #[inline]
    pub fn protected_count(&self) -> usize {
        self.hot.protected_count() + self.cold.as_ref().map_or(0, |c| c.protected_count())
    }

    #[inline]
    pub fn hist_freq(&self, key: usize) -> u16 {
        if self.cold.is_some() && self.is_cold[key] {
            self.cold.as_ref().unwrap().hist_freq(key)
        } else {
            self.hot.hist_freq(key)
        }
    }

    #[inline]
    pub fn peek(&self, key: usize) -> Option<usize> {
        let s = self.map[key];
        if s < 0 { None } else { Some(s as usize) }
    }

    #[inline]
    pub fn promote_prewarm_soft(&mut self, key: usize) {
        if self.cold.is_some() && self.is_cold[key] {
            self.cold.as_mut().unwrap().promote_prewarm_soft(key);
        } else {
            self.hot.promote_prewarm_soft(key);
        }
    }

    pub fn touch_hit(&mut self, key: usize) {
        if self.cold.is_some() && self.is_cold[key] {
            self.cold.as_mut().unwrap().touch_hit(key);
        } else {
            self.hot.touch_hit(key);
        }
    }

    #[inline]
    pub fn pending(&self, key: usize) -> Option<Ticket> {
        if self.cold.is_some() && self.is_cold[key] {
            self.cold.as_ref().unwrap().pending(key).map(|t| Ticket {
                key: t.key,
                slot: self.hot_cap + t.slot,
                generation: t.generation,
            })
        } else {
            self.hot.pending(key)
        }
    }

    pub fn release(&mut self) {
        self.hot.release();
        if let Some(c) = self.cold.as_mut() {
            c.release();
        }
        self.current.clear();
    }

    pub fn protect(&mut self, keys: &[usize]) -> io::Result<()> {
        self.hot_tmp.clear();
        self.cold_tmp.clear();
        self.current.clear();
        let has_cold = self.cold.is_some();
        for &k in keys {
            if has_cold && self.is_cold[k] {
                self.cold_tmp.push(k);
            } else {
                self.hot_tmp.push(k);
            }
        }
        if !self.hot_tmp.is_empty() {
            self.hot.protect(&self.hot_tmp)?;
            self.current.extend_from_slice(self.hot.current_keys());
        }
        if let Some(c) = self.cold.as_mut() {
            if !self.cold_tmp.is_empty() {
                if let Err(e) = c.protect(&self.cold_tmp) {
                    self.hot.release();
                    self.current.clear();
                    return Err(e);
                }
                self.current.extend_from_slice(c.current_keys());
            }
        }
        Ok(())
    }

    pub fn reserve(&mut self, key: usize, is_prefetch: bool) -> io::Result<Option<Ticket>> {
        if self.cold.is_some() && self.is_cold[key] {
            let c = self.cold.as_mut().unwrap();
            if let Some((t, evicted)) = c.reserve_evict(key, is_prefetch)? {
                if let Some(ok) = evicted {
                    self.map[ok] = ABSENT;
                }
                Ok(Some(Ticket {
                    key: t.key,
                    slot: self.hot_cap + t.slot,
                    generation: t.generation,
                }))
            } else {
                Ok(None)
            }
        } else if let Some((t, evicted)) = self.hot.reserve_evict(key, is_prefetch)? {
            if let Some(ok) = evicted {
                self.map[ok] = ABSENT;
            }
            Ok(Some(t))
        } else {
            Ok(None)
        }
    }

    pub fn publish(&mut self, ticket: Ticket) -> io::Result<()> {
        if self.cold.is_some() && self.is_cold[ticket.key] {
            let local = Ticket {
                key: ticket.key,
                slot: ticket.slot - self.hot_cap,
                generation: ticket.generation,
            };
            self.cold.as_mut().unwrap().publish(local)?;
        } else {
            self.hot.publish(ticket)?;
        }
        self.map[ticket.key] = ticket.slot as i32;
        Ok(())
    }

    pub fn abort(&mut self, ticket: Ticket) -> io::Result<()> {
        if self.cold.is_some() && self.is_cold[ticket.key] {
            let local = Ticket {
                key: ticket.key,
                slot: ticket.slot - self.hot_cap,
                generation: ticket.generation,
            };
            self.cold.as_mut().unwrap().abort(local)?;
        } else {
            self.hot.abort(ticket)?;
        }
        Ok(())
    }

    pub fn seed_zipf_prior(&mut self, ranks: &[u32]) {
        if self.cold.is_some() {
            self.hot.seed_zipf_prior_filtered(ranks, Some(&self.is_cold), false);
            self.cold
                .as_mut()
                .unwrap()
                .seed_zipf_prior_filtered(ranks, Some(&self.is_cold), true);
        } else {
            self.hot.seed_zipf_prior(ranks);
        }
    }

    pub fn clear_entries_keep_freq(&mut self) {
        self.hot.clear_entries_keep_freq();
        if let Some(c) = self.cold.as_mut() {
            c.clear_entries_keep_freq();
        }
        self.map.fill(ABSENT);
        self.current.clear();
    }

    #[cfg(debug_assertions)]
    pub fn assert_invariants(&self) {
        self.hot.assert_invariants();
        if let Some(c) = self.cold.as_ref() {
            c.assert_invariants();
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PoolStats {
    pub vram_hits: u64,
    pub vram_misses: u64,
    pub ram_hits: u64,
    pub cpu_hits: u64,
    pub disk_misses: u64,
    pub nvme_tail_skipped: u64,
    pub spec_submitted: u64,
    pub spec_hits: u64,
    pub spec_ready_early: u64,
    pub htod_bytes: u64,
    pub disk_bytes: u64,
}

const MAX_SPEC_INFLIGHT: usize = 8; // 8 experts * 3 projs = 24 SQEs
/// Double-buffered streaming prefill ring (`2 x 8 = 16` experts): zero cache eviction, full NVMe+PCIe+GPU overlap!
pub const PF_WAVE_SLOTS: usize = 8;
pub const PF_RING_BUFS: usize = 2;
pub const PF_STAGE_SLOTS: usize = PF_WAVE_SLOTS * PF_RING_BUFS;

struct InflightSpec {
    tickets: [Option<Ticket>; MAX_SPEC_INFLIGHT],
    count: usize,
    done_mask: u64,
    published_mask: u16,
    submitted_sqes: usize,
    completed_sqes: usize,
}

impl Default for InflightSpec {
    fn default() -> Self {
        Self {
            tickets: [None; MAX_SPEC_INFLIGHT],
            count: 0,
            done_mask: 0,
            published_mask: 0,
            submitted_sqes: 0,
            completed_sqes: 0,
        }
    }
}

/// Complete 3-tier CUDA + Dual-Ring `io_uring` (`read_fixed`) + Split-Tier Compact W-TinyLFU 2Q Pool.
/// Supports both uniform v1 (`M3TISPK1`) and mixed-bitwidth v2 (`M3TISPK2`, e.g. 3b hot / 2b cold) packs
/// with tightly packed per-bitwidth VRAM & RAM slabs (zero 2b padding!) and registered fixed I/O buffers.
pub struct TieredExpertPool {
    pub catalog: PackCatalog,
    file: DirectFile,
    pub io: DualRingIo,
    pub vram_cache: SplitSlotCache,
    pub ram_cache: SplitSlotCache,
    hot_strides: [usize; 3],
    cold_strides: [usize; 3],
    vram_hot_slots: usize,
    vram_cold_slots: usize,
    ram_hot_slots: usize,
    ram_cold_slots: usize,
    vram_slabs: [DeviceBuf; 3],
    ram_slabs: [PinnedHostBuf; 3],
    // Registered io_uring fixed-buffer chunk maps (`-1` if unregistered)
    ram_hot_buf_base: [i32; 3],
    ram_hot_chunk_slots: [usize; 3],
    ram_cold_buf_base: [i32; 3],
    ram_cold_chunk_slots: [usize; 3],
    host_map: PinnedHostBuf,
    dev_map: DeviceBuf,
    h_k2: [PinnedHostBuf; 3],
    d_k2: [DeviceBuf; 3],
    is_mixed: bool,
    k2_dirty: bool,
    dev_map_dirty: bool,
    pub copy_stream: Stream,
    copy_done: Event,
    prev_compute_done: Event,
    has_prev_compute: bool,
    bg_copy_pending: bool,
    spec: InflightSpec,
    miss_buf: Vec<usize>,
    cold_buf: Vec<usize>,
    budget_buf: Vec<usize>,
    demand_jobs: Vec<Ticket>,
    demand_ops: Vec<ReadOp>,
    // Milestone 5B: Double-buffered Expert-Major Streaming Prefill Ring (zero cache eviction!)
    pf_stage_ram: [PinnedHostBuf; 3],
    pf_stage_ram_buf_idx: [i32; 3],
    pf_stage_vram: [DeviceBuf; 3],
    pf_stage_ptrs: [DeviceBuf; 3],
    pf_h_k2: [[PinnedHostBuf; 3]; PF_RING_BUFS],
    pf_d_k2: [DeviceBuf; 3],
    pf_copy_done: [Event; PF_RING_BUFS],
    pf_compute_done: [Event; PF_RING_BUFS],
    pf_copy_armed: [bool; PF_RING_BUFS],
    pf_compute_armed: [bool; PF_RING_BUFS],
    pf_io_ops: Vec<ReadOp>,
    pf_io_inflight_sqes: usize,
    pf_io_inflight_experts: usize,
    // Milestone 6: Lock-free asynchronous Layer-Ahead NVMe streamer (overlaps 100% of NVMe with Attention + Dense!)
    pub ahead: Option<layer_stream::LayerAheadStreamer>,
    ahead_disk_bytes: u64,
    pub expert_hits: Vec<u64>,
    pub stats: PoolStats,
}

impl TieredExpertPool {
    pub fn open(
        ctx: &Arc<CudaContext>,
        pack_path: &Path,
        vram_gib: f64,
        ram_gib: f64,
        policy: EvictionPolicy,
        want_sqpoll: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let catalog = PackCatalog::open(pack_path)?;
        let file = DirectFile::open(pack_path)?;
        let mut io = DualRingIo::new(64, want_sqpoll)?;

        let total = catalog.total_experts();
        let hot_strides = catalog.max_strides;
        let hot_per_exp: usize = hot_strides.iter().sum();

        // Detect compact cold class (e.g. 2b experts with aligned_len < 3b max_strides)
        let mut is_cold = vec![false; total];
        let mut cold_strides = [0usize; 3];
        let mut n_cold_keys = 0usize;
        for (k, e) in catalog.entries.iter().enumerate() {
            let smaller = (0..3).all(|p| e.components[p].aligned_len < hot_strides[p]);
            if smaller {
                is_cold[k] = true;
                n_cold_keys += 1;
                for p in 0..3 {
                    cold_strides[p] = cold_strides[p].max(e.components[p].aligned_len);
                }
            }
        }
        let n_hot_keys = total - n_cold_keys;
        let use_split = n_cold_keys > 0 && n_hot_keys > 0 && cold_strides.iter().sum::<usize>() < hot_per_exp;
        let cold_per_exp: usize = if use_split { cold_strides.iter().sum() } else { 0 };

        // Compute per-layer max hot/cold counts so wide prefill chunks never overflow either sub-pool
        let mut max_hot_layer = 0usize;
        let mut max_cold_layer = 0usize;
        if use_split {
            let e_per = catalog.experts_per_layer;
            for l in 0..catalog.layers {
                let mut lh = 0usize;
                let mut lc = 0usize;
                for e in 0..e_per {
                    if is_cold[l * e_per + e] { lc += 1; } else { lh += 1; }
                }
                max_hot_layer = max_hot_layer.max(lh);
                max_cold_layer = max_cold_layer.max(lc);
            }
        } else {
            is_cold.fill(false);
        }

        let pf_stage_bytes = PF_STAGE_SLOTS * hot_per_exp;
        let raw_vram_budget = (vram_gib * (1u64 << 30) as f64) as usize;
        let vram_bytes_budget = if raw_vram_budget > pf_stage_bytes * 2 {
            raw_vram_budget - pf_stage_bytes
        } else {
            raw_vram_budget
        };
        let ram_bytes_budget = (ram_gib * (1u64 << 30) as f64) as usize;

        let (vram_hot_slots, vram_cold_slots, ram_hot_slots, ram_cold_slots) = if use_split {
            // VRAM: 65% hot 3b (super-expert core) / 35% compact 2b (prevents 2b thrash & maximizes total slots!)
            let min_vh = max_hot_layer.max(16).min(n_hot_keys);
            let min_vc = max_cold_layer.max(16).min(n_cold_keys);
            let vh = (((vram_bytes_budget as f64 * 0.65) as usize) / hot_per_exp).clamp(min_vh, n_hot_keys);
            let rem_v = vram_bytes_budget.saturating_sub(vh * hot_per_exp);
            let vc = (rem_v / cold_per_exp).clamp(min_vc, n_cold_keys);

            // RAM: 65% hot 3b / 35% compact 2b (keeps ~1,400+ compact 2b RAM slots to eliminate 2b SSD misses!)
            let min_rh = (max_hot_layer + MAX_SPEC_INFLIGHT).max(32).min(n_hot_keys);
            let min_rc = (max_cold_layer + MAX_SPEC_INFLIGHT).max(32).min(n_cold_keys);
            let max_useful_rh = (n_hot_keys.saturating_sub(vh) + max_hot_layer).clamp(min_rh, n_hot_keys);
            let rh = (((ram_bytes_budget as f64 * 0.65) as usize) / hot_per_exp).clamp(min_rh, max_useful_rh);
            let rem_r = ram_bytes_budget.saturating_sub(rh * hot_per_exp);
            let rc = (rem_r / cold_per_exp).clamp(min_rc, n_cold_keys);
            (vh, vc, rh, rc)
        } else {
            let v_slots = (vram_bytes_budget / hot_per_exp).clamp(16, total);
            let min_ram_slots = (catalog.experts_per_layer + MAX_SPEC_INFLIGHT).min(total).max(32);
            let r_slots = (ram_bytes_budget / hot_per_exp).clamp(min_ram_slots, total);
            (v_slots, 0, r_slots, 0)
        };
        let vram_slots = vram_hot_slots + vram_cold_slots;

        let mut vram_slabs_v = Vec::with_capacity(3);
        let mut ram_slabs_v = Vec::with_capacity(3);
        for p in 0..3 {
            let vb = vram_hot_slots * hot_strides[p] + vram_cold_slots * cold_strides[p];
            let rb = ram_hot_slots * hot_strides[p] + ram_cold_slots * cold_strides[p];
            vram_slabs_v.push(ctx.alloc_device(vb)?);
            ram_slabs_v.push(ctx.alloc_pinned(rb)?);
        }
        let vram_slabs: [DeviceBuf; 3] = vram_slabs_v.try_into().ok().unwrap();
        let mut ram_slabs: [PinnedHostBuf; 3] = ram_slabs_v.try_into().ok().unwrap();

        // Allocate Milestone 5B double-buffered Expert-Major Streaming Prefill slabs (2 x 8 = 16 hot-sized slots)
        let mut pf_stage_ram_v = Vec::with_capacity(3);
        let mut pf_stage_vram_v = Vec::with_capacity(3);
        for p in 0..3 {
            let sb = PF_STAGE_SLOTS * hot_strides[p];
            pf_stage_ram_v.push(ctx.alloc_pinned(sb)?);
            pf_stage_vram_v.push(ctx.alloc_device(sb)?);
        }
        let mut pf_stage_ram: [PinnedHostBuf; 3] = pf_stage_ram_v.try_into().ok().unwrap();
        let pf_stage_vram: [DeviceBuf; 3] = pf_stage_vram_v.try_into().ok().unwrap();

        // Register pinned RAM slabs + streaming prefill bounce slabs in <= 1 GiB chunks for zero-copy io_uring read_fixed!
        const MAX_IOVEC_BYTES: usize = 1008 * 1024 * 1024; // strictly < 1 GiB, multiple of 6 MiB
        let mut reg_slices: Vec<(*mut u8, usize)> = Vec::new();
        let mut ram_hot_buf_base = [-1i32; 3];
        let mut ram_hot_chunk_slots = [1usize; 3];
        let mut ram_cold_buf_base = [-1i32; 3];
        let mut ram_cold_chunk_slots = [1usize; 3];
        let mut pf_stage_ram_buf_idx = [-1i32; 3];
        for p in 0..3 {
            let base_ptr = ram_slabs[p].as_mut_ptr();
            let h_chunk = (MAX_IOVEC_BYTES / hot_strides[p]).max(1);
            ram_hot_chunk_slots[p] = h_chunk;
            ram_hot_buf_base[p] = reg_slices.len() as i32;
            let mut s0 = 0usize;
            while s0 < ram_hot_slots {
                let ns = h_chunk.min(ram_hot_slots - s0);
                let ptr = unsafe { base_ptr.add(s0 * hot_strides[p]) };
                reg_slices.push((ptr, ns * hot_strides[p]));
                s0 += ns;
            }
            if ram_cold_slots > 0 {
                let cold_off = ram_hot_slots * hot_strides[p];
                let c_chunk = (MAX_IOVEC_BYTES / cold_strides[p]).max(1);
                ram_cold_chunk_slots[p] = c_chunk;
                ram_cold_buf_base[p] = reg_slices.len() as i32;
                let mut c0 = 0usize;
                while c0 < ram_cold_slots {
                    let ns = c_chunk.min(ram_cold_slots - c0);
                    let ptr = unsafe { base_ptr.add(cold_off + c0 * cold_strides[p]) };
                    reg_slices.push((ptr, ns * cold_strides[p]));
                    c0 += ns;
                }
            }
            pf_stage_ram_buf_idx[p] = reg_slices.len() as i32;
            reg_slices.push((pf_stage_ram[p].as_mut_ptr(), pf_stage_ram[p].len()));
        }
        let reg_ok = io.demand.register_buffers(&reg_slices).is_ok()
            && io.prefetch.register_buffers(&reg_slices).is_ok();
        if !reg_ok {
            ram_hot_buf_base = [-1; 3];
            ram_cold_buf_base = [-1; 3];
            pf_stage_ram_buf_idx = [-1; 3];
        }

        let map_bytes = total * std::mem::size_of::<i32>();
        let mut host_map = ctx.alloc_pinned(map_bytes)?;
        host_map.as_i32_mut().fill(ABSENT);
        let dev_map = ctx.alloc_device(map_bytes)?;

        // Per-slot EXL3 k2 tables (2 * bits) for gate/up/down -> hot slots prefilled with hot bits, cold with cold bits!
        let first_hot_bits = catalog
            .entries
            .iter()
            .enumerate()
            .find(|&(k, _)| !is_cold[k])
            .map(|(_, e)| e.components.map(|c| c.bits))
            .unwrap_or_else(|| catalog.entries[0].components.map(|c| c.bits));
        let first_cold_bits = catalog
            .entries
            .iter()
            .enumerate()
            .find(|&(k, _)| is_cold[k])
            .map(|(_, e)| e.components.map(|c| c.bits))
            .unwrap_or(first_hot_bits);
        let is_mixed = catalog.entries.iter().any(|e| {
            e.components[0].bits != first_hot_bits[0]
                || e.components[1].bits != first_hot_bits[1]
                || e.components[2].bits != first_hot_bits[2]
        });
        let k2_bytes = vram_slots * std::mem::size_of::<i32>();
        let mut h_k2_vec = Vec::with_capacity(3);
        let mut d_k2_vec = Vec::with_capacity(3);
        let copy_stream = ctx.stream()?;
        for p in 0..3 {
            let mut hk = ctx.alloc_pinned(k2_bytes)?;
            {
                let sl = hk.as_i32_mut();
                sl[..vram_hot_slots].fill((first_hot_bits[p] as i32) * 2);
                sl[vram_hot_slots..vram_slots].fill((first_cold_bits[p] as i32) * 2);
            }
            let dk = ctx.alloc_device(k2_bytes)?;
            unsafe {
                copy_stream.htod_async(dk.ptr(), hk.as_ptr(), k2_bytes)?;
            }
            h_k2_vec.push(hk);
            d_k2_vec.push(dk);
        }
        let h_k2: [PinnedHostBuf; 3] = h_k2_vec.try_into().ok().unwrap();
        let d_k2: [DeviceBuf; 3] = d_k2_vec.try_into().ok().unwrap();

        copy_stream.memset_d32_async(dev_map.ptr(), ABSENT as u32, total)?;
        copy_stream.synchronize()?;

        let copy_done = ctx.event()?;
        let prev_compute_done = ctx.event()?;

        // Pre-populate streaming prefill VRAM pointer tables & per-buffer k2 metadata
        let mut pf_stage_ptrs_v = Vec::with_capacity(3);
        let mut pf_d_k2_v = Vec::with_capacity(3);
        let mut pf_h_k2_0 = Vec::with_capacity(3);
        let mut pf_h_k2_1 = Vec::with_capacity(3);
        for p in 0..3 {
            let mut h_ptrs = vec![0u64; PF_STAGE_SLOTS];
            let base = pf_stage_vram[p].ptr();
            for s in 0..PF_STAGE_SLOTS {
                h_ptrs[s] = base + (s * hot_strides[p]) as u64;
            }
            let d_ptrs = ctx.alloc_device(PF_STAGE_SLOTS * 8)?;
            unsafe {
                copy_stream.htod_async(d_ptrs.ptr(), h_ptrs.as_ptr() as *const u8, PF_STAGE_SLOTS * 8)?;
            }
            pf_stage_ptrs_v.push(d_ptrs);
            pf_d_k2_v.push(ctx.alloc_device(PF_STAGE_SLOTS * 4)?);
            pf_h_k2_0.push(ctx.alloc_pinned(PF_WAVE_SLOTS * 4)?);
            pf_h_k2_1.push(ctx.alloc_pinned(PF_WAVE_SLOTS * 4)?);
        }
        copy_stream.synchronize()?;
        let pf_stage_ptrs: [DeviceBuf; 3] = pf_stage_ptrs_v.try_into().ok().unwrap();
        let pf_d_k2: [DeviceBuf; 3] = pf_d_k2_v.try_into().ok().unwrap();
        let pf_h_k2: [[PinnedHostBuf; 3]; PF_RING_BUFS] = [
            pf_h_k2_0.try_into().ok().unwrap(),
            pf_h_k2_1.try_into().ok().unwrap(),
        ];
        let pf_copy_done = [ctx.event()?, ctx.event()?];
        let pf_compute_done = [ctx.event()?, ctx.event()?];
        let ahead = if ram_bytes_budget >= (4u64 << 30) as usize {
            Some(layer_stream::LayerAheadStreamer::open(
                ctx,
                pack_path,
                hot_strides,
                layer_stream::AHEAD_RING_SLOTS,
            )?)
        } else {
            None
        };

        let mut vram_cache = SplitSlotCache::new(total, vram_hot_slots, vram_cold_slots, is_cold.clone(), policy);
        let mut ram_cache = SplitSlotCache::new(total, ram_hot_slots, ram_cold_slots, is_cold, policy);
        if catalog.version == m3tis_format::PackVersion::V2MixedZipf {
            let ranks: Vec<u32> = catalog.entries.iter().map(|e| e.zipf_rank).collect();
            vram_cache.seed_zipf_prior(&ranks);
            ram_cache.seed_zipf_prior(&ranks);
        }

        Ok(Self {
            vram_cache,
            ram_cache,
            catalog,
            file,
            io,
            hot_strides,
            cold_strides,
            vram_hot_slots,
            vram_cold_slots,
            ram_hot_slots,
            ram_cold_slots,
            vram_slabs,
            ram_slabs,
            ram_hot_buf_base,
            ram_hot_chunk_slots,
            ram_cold_buf_base,
            ram_cold_chunk_slots,
            host_map,
            dev_map,
            h_k2,
            d_k2,
            is_mixed,
            k2_dirty: false,
            dev_map_dirty: false,
            copy_stream,
            copy_done,
            prev_compute_done,
            has_prev_compute: false,
            bg_copy_pending: false,
            spec: InflightSpec::default(),
            miss_buf: Vec::with_capacity(vram_slots),
            cold_buf: Vec::with_capacity(vram_slots),
            budget_buf: Vec::with_capacity(512),
            demand_jobs: Vec::with_capacity(vram_slots),
            demand_ops: Vec::with_capacity(vram_slots * 3),
            pf_stage_ram,
            pf_stage_ram_buf_idx,
            pf_stage_vram,
            pf_stage_ptrs,
            pf_h_k2,
            pf_d_k2,
            pf_copy_done,
            pf_compute_done,
            pf_copy_armed: [false; PF_RING_BUFS],
            pf_compute_armed: [false; PF_RING_BUFS],
            pf_io_ops: Vec::with_capacity(PF_WAVE_SLOTS * 3),
            pf_io_inflight_sqes: 0,
            pf_io_inflight_experts: 0,
            ahead,
            ahead_disk_bytes: 0,
            expert_hits: vec![0u64; total],
            stats: PoolStats::default(),
        })
    }

    #[inline]
    fn vram_slot_byte_offset(&self, comp: usize, vslot: usize) -> usize {
        if vslot < self.vram_hot_slots {
            vslot * self.hot_strides[comp]
        } else {
            self.vram_hot_slots * self.hot_strides[comp]
                + (vslot - self.vram_hot_slots) * self.cold_strides[comp]
        }
    }

    #[inline]
    fn ram_slot_byte_offset(&self, comp: usize, rslot: usize) -> usize {
        if rslot < self.ram_hot_slots {
            rslot * self.hot_strides[comp]
        } else {
            self.ram_hot_slots * self.hot_strides[comp]
                + (rslot - self.ram_hot_slots) * self.cold_strides[comp]
        }
    }

    #[inline]
    fn ram_slot_fixed_buf_idx(&self, comp: usize, rslot: usize) -> i32 {
        if rslot < self.ram_hot_slots {
            let base = self.ram_hot_buf_base[comp];
            if base < 0 { -1 } else { base + (rslot / self.ram_hot_chunk_slots[comp]) as i32 }
        } else {
            let base = self.ram_cold_buf_base[comp];
            if base < 0 {
                -1
            } else {
                base + ((rslot - self.ram_hot_slots) / self.ram_cold_chunk_slots[comp]) as i32
            }
        }
    }

    #[inline]
    pub fn expert_vram_bytes(&self) -> usize {
        self.vram_slabs[0].len()
            + self.vram_slabs[1].len()
            + self.vram_slabs[2].len()
            + self.pf_stage_vram[0].len()
            + self.pf_stage_vram[1].len()
            + self.pf_stage_vram[2].len()
    }

    #[inline]
    pub fn vram_hot_cold_slots(&self) -> (usize, usize) {
        (self.vram_hot_slots, self.vram_cold_slots)
    }

    #[inline]
    pub fn ram_hot_cold_slots(&self) -> (usize, usize) {
        (self.ram_hot_slots, self.ram_cold_slots)
    }

    #[inline]
    pub fn is_registered_bufs(&self) -> bool {
        self.io.demand.registered_bufs && self.io.prefetch.registered_bufs
    }

    #[inline]
    pub fn vram_k2_ptr(&self, comp: usize) -> CuDevicePtr {
        self.d_k2[comp].ptr()
    }

    #[inline]
    pub fn is_mixed_bitwidth(&self) -> bool {
        self.is_mixed
    }

    #[inline]
    pub fn vram_slab_ptr(&self, comp: usize) -> CuDevicePtr {
        self.vram_slabs[comp].ptr()
    }

    #[inline]
    pub fn dev_map_ptr(&self) -> CuDevicePtr {
        self.dev_map.ptr()
    }

    #[inline]
    pub fn is_direct(&self) -> bool {
        self.file.direct
    }

    #[inline]
    pub fn vram_slots(&self) -> usize {
        self.vram_cache.capacity()
    }

    /// Return zero-copy pinned-RAM pointers `([gate, up, down], [k2_g, k2_u, k2_d])` if `key` is
    /// currently ready in Tier-2 Pinned RAM (used by Hybrid CPU-GPU EXL3 execution!).
    #[inline]
    pub fn ram_expert_ptrs(&self, key: usize) -> Option<([*const u32; 3], [i32; 3])> {
        let rslot = self.ram_cache.peek(key)?;
        let entry = &self.catalog.entries[key];
        let mut ptrs = [std::ptr::null::<u32>(); 3];
        let mut k2s = [0i32; 3];
        for p in 0..3 {
            let off = self.ram_slot_byte_offset(p, rslot);
            ptrs[p] = unsafe { self.ram_slabs[p].as_ptr().add(off) as *const u32 };
            k2s[p] = (entry.components[p].bits as i32) * 2;
        }
        Some((ptrs, k2s))
    }

    /// Protect `cpu_keys` in Tier-2 Pinned RAM (leasing their RAM slots so neither cold NVMe reads
    /// nor speculative prefetches can evict them while background CPU SIMD workers execute them!).
    pub fn protect_ram_only(&mut self, cpu_keys: &[usize]) -> io::Result<()> {
        if cpu_keys.is_empty() {
            return Ok(());
        }
        self.ram_cache.release();
        self.ram_cache.protect(cpu_keys)
    }

    /// Build `[vram_slots]` u64 device pointer tables for gate/up/down VRAM slabs + uniform k2 table.
    pub fn build_exl3_slot_tables(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Stream,
        default_k2: i32,
    ) -> Result<([DeviceBuf; 3], DeviceBuf), m3tis_cuda::CudaError> {
        let slots = self.vram_slots();
        let mut ptr_bufs = Vec::with_capacity(3);
        for p in 0..3 {
            let base = self.vram_slabs[p].ptr();
            let ptrs: Vec<u64> = (0..slots)
                .map(|s| base + self.vram_slot_byte_offset(p, s) as u64)
                .collect();
            let d = ctx.alloc_device(slots * 8)?;
            unsafe {
                stream.htod_async(d.ptr(), ptrs.as_ptr() as *const u8, slots * 8)?;
            }
            ptr_bufs.push(d);
        }
        let k2s = vec![default_k2; slots];
        let d_k2 = ctx.alloc_device(slots * 4)?;
        unsafe {
            stream.htod_async(d_k2.ptr(), k2s.as_ptr() as *const u8, slots * 4)?;
        }
        let [g, u, d]: [DeviceBuf; 3] = ptr_bufs.try_into().ok().unwrap();
        Ok(([g, u, d], d_k2))
    }

    fn poll_spec_prefetch(&mut self, wait_all: bool) -> io::Result<()> {
        if self.spec.count == 0 {
            return Ok(());
        }
        let mut ud = [0u64; 64];
        let mut res = [0i32; 64];
        while self.spec.completed_sqes < self.spec.submitted_sqes {
            let rem = self.spec.submitted_sqes - self.spec.completed_sqes;
            let wait_nr = if wait_all { rem as u32 } else { 0 };
            let got = self
                .io
                .prefetch
                .reap_async(&mut ud[..rem], &mut res[..rem], wait_nr, &mut self.io.prefetch_stats)?;
            if got == 0 {
                break;
            }
            for i in 0..got {
                let sqe_idx = ud[i] as usize;
                if res[i] < 0 {
                    return Err(io::Error::from_raw_os_error(-res[i]));
                }
                self.spec.done_mask |= 1u64 << sqe_idx;
                self.spec.completed_sqes += 1;
            }
            if !wait_all {
                break;
            }
        }
        // Publish any expert whose 3 component SQEs have all completed.
        for i in 0..self.spec.count {
            let bit = 1u16 << i;
            if (self.spec.published_mask & bit) != 0 {
                continue;
            }
            let need = 0b111u64 << (i * 3);
            if (self.spec.done_mask & need) == need {
                if let Some(t) = self.spec.tickets[i] {
                    self.ram_cache.publish(t)?;
                    self.spec.published_mask |= bit;
                    if !wait_all {
                        self.stats.spec_ready_early += 1;
                    }
                }
            }
        }
        if wait_all || self.spec.completed_sqes >= self.spec.submitted_sqes {
            self.spec = InflightSpec::default();
        }
        Ok(())
    }

    /// Non-blocking Lever 2 speculative next-layer prefetch (~5 µs return).
    pub fn start_spec_prefetch(&mut self, ranked_keys: &[usize], max_experts: usize) -> io::Result<usize> {
        if ranked_keys.is_empty() || max_experts == 0 {
            return Ok(0);
        }
        if self.spec.count > 0 {
            self.poll_spec_prefetch(false)?;
            if self.spec.count > 0 {
                // Previous speculative prefetch still in flight — NEVER block compute on a mispredicted read!
                return Ok(0);
            }
        }
        let limit = max_experts.min(MAX_SPEC_INFLIGHT);
        let mut ops = [ReadOp {
            fd: 0,
            offset: 0,
            dst: std::ptr::null_mut(),
            len: 0,
            user_data: 0,
            fixed_buf_idx: -1,
        }; MAX_SPEC_INFLIGHT * 3];
        let mut count = 0usize;
        let fd = self.file.fd();

        for &key in ranked_keys {
            if count >= limit {
                break;
            }
            if self.vram_cache.peek(key).is_some()
                || self.ram_cache.peek(key).is_some()
                || self.ram_cache.pending(key).is_some()
            {
                continue;
            }
            let ticket = match self.ram_cache.reserve(key, true) {
                Ok(Some(t)) => t,
                Ok(None) => continue,
                Err(_) => break,
            };
            let entry = &self.catalog.entries[key];
            for p in 0..3 {
                let comp = entry.components[p];
                let off = self.ram_slot_byte_offset(p, ticket.slot);
                let fidx = self.ram_slot_fixed_buf_idx(p, ticket.slot);
                let dst = unsafe { self.ram_slabs[p].as_mut_ptr().add(off) };
                ops[count * 3 + p] = ReadOp {
                    fd,
                    offset: comp.offset,
                    dst,
                    len: comp.aligned_len,
                    user_data: (count * 3 + p) as u64,
                    fixed_buf_idx: fidx,
                };
            }
            self.spec.tickets[count] = Some(ticket);
            count += 1;
        }
        if count == 0 {
            return Ok(0);
        }
        let sqes = count * 3;
        self.io
            .prefetch
            .submit_async(&ops[..sqes], &mut self.io.prefetch_stats)?;
        self.spec.count = count;
        self.spec.submitted_sqes = sqes;
        self.spec.completed_sqes = 0;
        self.spec.done_mask = 0;
        self.spec.published_mask = 0;
        self.stats.spec_submitted += count as u64;
        Ok(count)
    }

    /// Fill `host_map` with 3-tier residency encoding:
    /// `>= 0` = Tier-1 VRAM slot, `-2` = Tier-2 Pinned RAM resident (0.25 ms), `-1` = Cold NVMe (1.5 ms).
    /// All remap kernels test `slot >= 0`, so `-2` is 100% transparent while letting the router prefer RAM over NVMe!
    #[inline]
    fn fill_host_map_tiered(&mut self) {
        let vmap = self.vram_cache.map_slice();
        let rmap = self.ram_cache.map_slice();
        let dst = self.host_map.as_i32_mut();
        for i in 0..vmap.len() {
            let v = vmap[i];
            dst[i] = if v >= 0 {
                v
            } else if rmap[i] >= 0 {
                -2
            } else {
                ABSENT
            };
        }
    }

    /// Milestone 8 Split-Wave Resident-First Async MoE (Phase 1):
    /// Classifies `all_keys` into kept Tier-1 VRAM hits (immediately leased in `vram_cache`!) vs missing keys
    /// (`miss_buf`, in RAM or NVMe). Returns `(kept_vram_hits_count, miss_count)`.
    /// When `miss_count > 0 && kept_vram_hits_count > 0`, caller launches Wave 1 EXL3 MoE on `compute_stream`
    /// IMMEDIATELY and then calls `finish_split_demand()` to stream RAM PCIe + NVMe 100% concurrently with Wave 1!
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_split_demand(
        &mut self,
        compute_stream: &Stream,
        all_keys: &[usize],
        strict_keys: &[usize],
        use_budget: bool,
        promote_all_ram: bool,
        scales: &mut scales::Exl3ScaleBank,
        skipped_cold_out: &mut Vec<usize>,
    ) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        if self.spec.count > 0 {
            self.poll_spec_prefetch(false)?;
        }
        skipped_cold_out.clear();
        self.budget_buf.clear();
        if use_budget {
            for &k in all_keys {
                let keep = strict_keys.contains(&k)
                    || self.vram_cache.peek(k).is_some()
                    || (promote_all_ram
                        && (self.ram_cache.peek(k).is_some() || self.ram_cache.pending(k).is_some()));
                if keep {
                    self.budget_buf.push(k);
                } else {
                    self.stats.nvme_tail_skipped += 1;
                    skipped_cold_out.push(k);
                }
            }
        } else {
            self.budget_buf.extend_from_slice(all_keys);
        }
        if self.budget_buf.is_empty() {
            return Ok((0, 0));
        }

        // Score speculative prefetch hits on the kept set
        if self.spec.count > 0 {
            for i in 0..self.spec.count {
                if let Some(t) = self.spec.tickets[i] {
                    if self.budget_buf.contains(&t.key) {
                        self.stats.spec_hits += 1;
                    }
                }
            }
        }

        // Protect ALL kept keys in VRAM: all ready VRAM hits are now leased and immune from eviction!
        self.vram_cache.release();
        let keys_copy = std::mem::take(&mut self.budget_buf);
        let prot_res = self.vram_cache.protect(&keys_copy);
        self.budget_buf = keys_copy;
        prot_res?;

        self.miss_buf.clear();
        for &key in self.vram_cache.current_keys() {
            if self.vram_cache.peek(key).is_none() {
                self.miss_buf.push(key);
            }
        }
        let uniq_len = self.vram_cache.current_keys().len();
        let miss_len = self.miss_buf.len();
        let hit_len = uniq_len - miss_len;
        self.stats.vram_hits += hit_len as u64;
        self.stats.vram_misses += miss_len as u64;

        // Fence any prior background L+1 RAM->VRAM promotion & ensure all Wave-1 hit scales are visible on compute_stream!
        let mut need_fence = self.bg_copy_pending;
        if hit_len > 0 {
            for &key in self.vram_cache.current_keys() {
                if let Some(vslot) = self.vram_cache.peek(key) {
                    if scales.sync_slot_async(&self.copy_stream, key, vslot)? {
                        need_fence = true;
                    }
                }
            }
        }
        if miss_len == 0 && (self.dev_map_dirty || self.k2_dirty) {
            if self.has_prev_compute {
                self.copy_stream.wait(&self.prev_compute_done)?;
            }
            if self.dev_map_dirty {
                self.fill_host_map_tiered();
                let total_bytes = self.vram_cache.map_slice().len() * std::mem::size_of::<i32>();
                unsafe {
                    self.copy_stream.htod_async(self.dev_map.ptr(), self.host_map.as_ptr(), total_bytes)?;
                }
                self.dev_map_dirty = false;
            }
            if self.k2_dirty {
                let k2_bytes = self.vram_slots() * std::mem::size_of::<i32>();
                for p in 0..3 {
                    unsafe {
                        self.copy_stream.htod_async(self.d_k2[p].ptr(), self.h_k2[p].as_ptr(), k2_bytes)?;
                    }
                }
                self.k2_dirty = false;
            }
            need_fence = true;
        }
        if need_fence {
            self.copy_done.record(&self.copy_stream)?;
            compute_stream.wait(&self.copy_done)?;
            self.bg_copy_pending = false;
        }
        Ok((hit_len, miss_len))
    }

    /// Milestone 8 Split-Wave Resident-First Async MoE (Phase 2):
    /// Called IMMEDIATELY after Wave 1 EXL3 MoE is enqueued on `compute_stream`!
    /// Streams all `miss_buf` experts from Tier-2 Pinned RAM (23.5 GB/s PCIe) and cold NVMe (6.75 GB/s io_uring)
    /// into disjoint unleased VRAM slots CONCURRENTLY with Wave 1, then fences `compute_stream` for Wave 2!
    pub fn finish_split_demand(
        &mut self,
        compute_stream: &Stream,
        scales: &mut scales::Exl3ScaleBank,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let miss_len = self.miss_buf.len();
        if miss_len == 0 {
            return Ok(());
        }
        if self.bg_copy_pending {
            self.copy_done.synchronize()?;
            self.bg_copy_pending = false;
        }
        self.ram_cache.release();
        self.ram_cache.protect(&self.miss_buf)?;

        // Wait ONLY on Layer L-1 compute completion (already done!), NOT on Layer L Wave 1!
        // All Wave-1 VRAM slots are leased in `vram_cache` and CANNOT be evicted by `reserve()`!
        if self.has_prev_compute {
            self.copy_stream.wait(&self.prev_compute_done)?;
        }

        self.cold_buf.clear();
        self.demand_jobs.clear();
        self.demand_ops.clear();
        let fd = self.file.fd();

        let mut ready_hits = 0u64;
        for i in 0..self.miss_buf.len() {
            let key = self.miss_buf[i];
            if let Some(ram_slot) = self.ram_cache.peek(key) {
                ready_hits += 1;
                if let Some(vticket) = self.vram_cache.reserve(key, false)? {
                    self.enqueue_htod(key, ram_slot, vticket.slot)?;
                    scales.sync_slot_async(&self.copy_stream, key, vticket.slot)?;
                    self.vram_cache.publish(vticket)?;
                }
            } else {
                self.cold_buf.push(key);
                if let Some(rticket) = self.ram_cache.reserve(key, false)? {
                    let entry = &self.catalog.entries[key];
                    for p in 0..3 {
                        let comp = entry.components[p];
                        let off = self.ram_slot_byte_offset(p, rticket.slot);
                        let fidx = self.ram_slot_fixed_buf_idx(p, rticket.slot);
                        let dst = unsafe { self.ram_slabs[p].as_mut_ptr().add(off) };
                        self.demand_ops.push(ReadOp {
                            fd,
                            offset: comp.offset,
                            dst,
                            len: comp.aligned_len,
                            user_data: (self.demand_jobs.len() * 3 + p) as u64,
                            fixed_buf_idx: fidx,
                        });
                    }
                    self.demand_jobs.push(rticket);
                }
            }
        }

        let new_disk = self.demand_jobs.len() as u64;
        self.stats.disk_misses += new_disk;
        self.stats.ram_hits += (miss_len as u64).saturating_sub(new_disk).max(ready_hits);

        if !self.demand_ops.is_empty() {
            self.io.read_demand(&self.demand_ops)?;
            for &t in &self.demand_jobs {
                self.ram_cache.publish(t)?;
            }
        }
        if self.spec.count > 0 {
            let mut wait_needed = self.cold_buf.iter().any(|&k| self.ram_cache.pending(k).is_some());
            if !wait_needed {
                self.poll_spec_prefetch(false)?;
            } else {
                while wait_needed && self.spec.completed_sqes < self.spec.submitted_sqes {
                    let rem = self.spec.submitted_sqes - self.spec.completed_sqes;
                    let mut ud = [0u64; 64];
                    let mut res = [0i32; 64];
                    let got = self.io.prefetch.reap_async(
                        &mut ud[..rem],
                        &mut res[..rem],
                        1,
                        &mut self.io.prefetch_stats,
                    )?;
                    for i in 0..got {
                        let sqe_idx = ud[i] as usize;
                        if res[i] < 0 {
                            return Err(Box::new(io::Error::from_raw_os_error(-res[i])));
                        }
                        self.spec.done_mask |= 1u64 << sqe_idx;
                        self.spec.completed_sqes += 1;
                    }
                    for i in 0..self.spec.count {
                        let bit = 1u16 << i;
                        if (self.spec.published_mask & bit) != 0 {
                            continue;
                        }
                        let need = 0b111u64 << (i * 3);
                        if (self.spec.done_mask & need) == need {
                            if let Some(t) = self.spec.tickets[i] {
                                self.ram_cache.publish(t)?;
                                self.spec.published_mask |= bit;
                            }
                        }
                    }
                    if self.spec.completed_sqes >= self.spec.submitted_sqes {
                        self.spec = InflightSpec::default();
                        break;
                    }
                    wait_needed = self.cold_buf.iter().any(|&k| self.ram_cache.pending(k).is_some());
                }
            }
        }

        for i in 0..self.cold_buf.len() {
            let key = self.cold_buf[i];
            if let Some(vticket) = self.vram_cache.reserve(key, false)? {
                let ram_slot = self.ram_cache.peek(key).ok_or("expert not ready in RAM")?;
                self.enqueue_htod(key, ram_slot, vticket.slot)?;
                scales.sync_slot_async(&self.copy_stream, key, vticket.slot)?;
                self.vram_cache.publish(vticket)?;
            }
        }

        // Mirror updated 3-tier dev_map (Wave 1 uses explicit d_mapped, so dev_map update is 100% race-free!)
        self.fill_host_map_tiered();
        let total_bytes = self.vram_cache.map_slice().len() * std::mem::size_of::<i32>();
        unsafe {
            self.copy_stream.htod_async(self.dev_map.ptr(), self.host_map.as_ptr(), total_bytes)?;
        }
        if self.k2_dirty {
            for &key in &self.miss_buf {
                if let Some(vslot) = self.vram_cache.peek(key) {
                    for p in 0..3 {
                        unsafe {
                            self.copy_stream.htod_async(
                                self.d_k2[p].at(vslot * 4),
                                self.h_k2[p].as_ptr().add(vslot * 4),
                                4,
                            )?;
                        }
                    }
                }
            }
            self.k2_dirty = false;
        }
        self.dev_map_dirty = false;
        self.copy_done.record(&self.copy_stream)?;
        compute_stream.wait(&self.copy_done)?;
        self.stats.disk_bytes = self.io.demand_stats.bytes + self.io.prefetch_stats.bytes + self.ahead_disk_bytes;
        Ok(())
    }

    /// Return the kept keys slice from the last `prepare_split_demand` call.
    #[inline]
    pub fn split_kept_keys(&self) -> &[usize] {
        &self.budget_buf
    }

    /// Never-Stall Tail demand: loads all `strict_keys` AND every key in `all_keys` already resident
    /// in VRAM (plus optionally Tier-2 RAM / in-flight prefetch when `promote_all_ram` is true),
    /// skipping unneeded tail misses without stalling compute!
    /// When `promote_all_ram == false` (True VRAM-Mass + RAM-First Gate), Tier-2 RAM tail hits are ONLY
    /// promoted if the caller explicitly included them in `strict_keys`, preventing PCIe/VRAM thrashing!
    #[allow(clippy::too_many_arguments)]
    pub fn demand_with_scales_budget(
        &mut self,
        compute_stream: &Stream,
        layer_base: usize,
        layer_experts: usize,
        all_keys: &[usize],
        strict_keys: &[usize],
        promote_all_ram: bool,
        scales: &mut scales::Exl3ScaleBank,
        skipped_cold_out: &mut Vec<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.spec.count > 0 {
            self.poll_spec_prefetch(false)?;
        }
        skipped_cold_out.clear();
        self.budget_buf.clear();
        for &k in all_keys {
            let keep = strict_keys.contains(&k)
                || self.vram_cache.peek(k).is_some()
                || (promote_all_ram
                    && (self.ram_cache.peek(k).is_some() || self.ram_cache.pending(k).is_some()));
            if keep {
                self.budget_buf.push(k);
            } else {
                self.stats.nvme_tail_skipped += 1;
                skipped_cold_out.push(k);
            }
        }
        if self.budget_buf.is_empty() {
            return Ok(());
        }
        let keys_copy = std::mem::take(&mut self.budget_buf);
        let res = self.demand_inner(compute_stream, layer_base, layer_experts, &keys_copy, Some(scales), &[]);
        self.budget_buf = keys_copy;
        res
    }

    /// Hybrid CPU-GPU demand: ensures `gpu_keys` are resident in Tier-1 VRAM (via RAM->VRAM HtoD + NVMe)
    /// while simultaneously leasing `cpu_keys` in Tier-2 Pinned RAM so background CPU EXL3 workers can read
    /// `cpu_keys` directly from DDR5 with ZERO PCIe transfers, ZERO VRAM evictions, and ZERO data races!
    #[allow(clippy::too_many_arguments)]
    pub fn demand_with_scales_hybrid(
        &mut self,
        compute_stream: &Stream,
        layer_base: usize,
        layer_experts: usize,
        gpu_keys: &[usize],
        cpu_keys: &[usize],
        scales: &mut scales::Exl3ScaleBank,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.stats.cpu_hits += cpu_keys.len() as u64;
        if gpu_keys.is_empty() {
            if !cpu_keys.is_empty() {
                self.ram_cache.release();
                self.ram_cache.protect(cpu_keys)?;
            }
            return Ok(());
        }
        self.demand_inner(compute_stream, layer_base, layer_experts, gpu_keys, Some(scales), cpu_keys)
    }

    /// Ensure every expert in `keys` is resident in Tier-1 VRAM before `compute_stream` runs.
    /// Overlaps Tier-2 RAM-hit PCIe HtoD with `demand_ring` + in-flight `prefetch_ring` NVMe I/O!
    pub fn demand(
        &mut self,
        compute_stream: &Stream,
        layer_base: usize,
        layer_experts: usize,
        keys: &[usize],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.demand_inner(compute_stream, layer_base, layer_experts, keys, None, &[])
    }

    /// Same as `demand`, and also mirrors any missing `Exl3ScaleBank` (`suh`/`svh`) scales into the
    /// assigned VRAM slots on `copy_stream` before recording `copy_done` for `compute_stream`.
    pub fn demand_with_scales(
        &mut self,
        compute_stream: &Stream,
        layer_base: usize,
        layer_experts: usize,
        keys: &[usize],
        scales: &mut scales::Exl3ScaleBank,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.demand_inner(compute_stream, layer_base, layer_experts, keys, Some(scales), &[])
    }

    fn demand_inner(
        &mut self,
        compute_stream: &Stream,
        _layer_base: usize,
        _layer_experts: usize,
        keys: &[usize],
        mut scales: Option<&mut scales::Exl3ScaleBank>,
        extra_ram_protect: &[usize],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if keys.is_empty() {
            return Ok(());
        }
        // 1. Score speculative prefetch hits and non-blockingly reap early completions!
        if self.spec.count > 0 {
            for i in 0..self.spec.count {
                if let Some(t) = self.spec.tickets[i] {
                    if keys.contains(&t.key) {
                        self.stats.spec_hits += 1;
                    }
                }
            }
            self.poll_spec_prefetch(false)?;
        }

        // 2. Protect Tier-1 VRAM hits
        self.vram_cache.release();
        self.vram_cache.protect(keys)?;

        self.miss_buf.clear();
        for &key in self.vram_cache.current_keys() {
            if self.vram_cache.peek(key).is_none() {
                self.miss_buf.push(key);
            }
        }
        let uniq_len = self.vram_cache.current_keys().len();
        let miss_len = self.miss_buf.len();
        self.stats.vram_hits += (uniq_len - miss_len) as u64;
        self.stats.vram_misses += miss_len as u64;

        if miss_len > 0 {
            // Host-sync any prior async L+1 RAM->VRAM promotion BEFORE releasing/evicting ram_cache slots!
            if self.bg_copy_pending {
                self.copy_done.synchronize()?;
                self.bg_copy_pending = false;
            }
            // 3. Protect all VRAM misses + any active CPU-offloaded RAM experts in Tier-2 2Q/LFU RAM!
            self.ram_cache.release();
            if extra_ram_protect.is_empty() {
                self.ram_cache.protect(&self.miss_buf)?;
            } else {
                let mut prot = Vec::with_capacity(miss_len + extra_ram_protect.len());
                prot.extend_from_slice(&self.miss_buf);
                prot.extend_from_slice(extra_ram_protect);
                self.ram_cache.protect(&prot)?;
            }

            if self.has_prev_compute {
                self.copy_stream.wait(&self.prev_compute_done)?;
            }

            // 4. Separate ready Tier-2 RAM hits from cold/in-flight misses; start RAM-hit HtoD immediately!
            self.cold_buf.clear();
            self.demand_jobs.clear();
            self.demand_ops.clear();
            let fd = self.file.fd();

            let mut ready_hits = 0u64;
            for i in 0..self.miss_buf.len() {
                let key = self.miss_buf[i];
                if let Some(ram_slot) = self.ram_cache.peek(key) {
                    ready_hits += 1;
                    if let Some(vticket) = self.vram_cache.reserve(key, false)? {
                        self.enqueue_htod(key, ram_slot, vticket.slot)?;
                        self.vram_cache.publish(vticket)?;
                        self.host_map.as_i32_mut()[key] = vticket.slot as i32;
                    }
                } else {
                    self.cold_buf.push(key);
                    if let Some(rticket) = self.ram_cache.reserve(key, false)? {
                        let entry = &self.catalog.entries[key];
                        for p in 0..3 {
                            let comp = entry.components[p];
                            let off = self.ram_slot_byte_offset(p, rticket.slot);
                            let fidx = self.ram_slot_fixed_buf_idx(p, rticket.slot);
                            let dst = unsafe { self.ram_slabs[p].as_mut_ptr().add(off) };
                            self.demand_ops.push(ReadOp {
                                fd,
                                offset: comp.offset,
                                dst,
                                len: comp.aligned_len,
                                user_data: (self.demand_jobs.len() * 3 + p) as u64,
                                fixed_buf_idx: fidx,
                            });
                        }
                        self.demand_jobs.push(rticket);
                    }
                }
            }

            let new_disk = self.demand_jobs.len() as u64;
            self.stats.disk_misses += new_disk;
            self.stats.ram_hits += (miss_len as u64).saturating_sub(new_disk).max(ready_hits);

            // 5. Execute unpredicted cold NVMe reads concurrently with RAM-hit PCIe HtoD & in-flight prefetch!
            if !self.demand_ops.is_empty() {
                self.io.read_demand(&self.demand_ops)?;
                for &t in &self.demand_jobs {
                    self.ram_cache.publish(t)?;
                }
            }
            // 6. Drain ONLY still-pending speculative prefetches that are actually needed by `cold_buf`!
            // Unneeded speculative misses stay in-flight asynchronously and never stall this layer's critical path!
            if self.spec.count > 0 {
                let mut wait_needed = self.cold_buf.iter().any(|&k| self.ram_cache.pending(k).is_some());
                if !wait_needed {
                    self.poll_spec_prefetch(false)?;
                } else {
                    while wait_needed && self.spec.completed_sqes < self.spec.submitted_sqes {
                        let rem = self.spec.submitted_sqes - self.spec.completed_sqes;
                        let mut ud = [0u64; 64];
                        let mut res = [0i32; 64];
                        let got = self.io.prefetch.reap_async(
                            &mut ud[..rem],
                            &mut res[..rem],
                            1,
                            &mut self.io.prefetch_stats,
                        )?;
                        for i in 0..got {
                            let sqe_idx = ud[i] as usize;
                            if res[i] < 0 {
                                return Err(Box::new(io::Error::from_raw_os_error(-res[i])));
                            }
                            self.spec.done_mask |= 1u64 << sqe_idx;
                            self.spec.completed_sqes += 1;
                        }
                        for i in 0..self.spec.count {
                            let bit = 1u16 << i;
                            if (self.spec.published_mask & bit) != 0 {
                                continue;
                            }
                            let need = 0b111u64 << (i * 3);
                            if (self.spec.done_mask & need) == need {
                                if let Some(t) = self.spec.tickets[i] {
                                    self.ram_cache.publish(t)?;
                                    self.spec.published_mask |= bit;
                                }
                            }
                        }
                        if self.spec.completed_sqes >= self.spec.submitted_sqes {
                            self.spec = InflightSpec::default();
                            break;
                        }
                        wait_needed = self.cold_buf.iter().any(|&k| self.ram_cache.pending(k).is_some());
                    }
                }
            }

            // 7. Promote freshly loaded cold misses from Pinned RAM -> VRAM
            for i in 0..self.cold_buf.len() {
                let key = self.cold_buf[i];
                if let Some(vticket) = self.vram_cache.reserve(key, false)? {
                    let ram_slot = self.ram_cache.peek(key).ok_or("expert not ready in RAM")?;
                    self.enqueue_htod(key, ram_slot, vticket.slot)?;
                    self.vram_cache.publish(vticket)?;
                    self.host_map.as_i32_mut()[key] = vticket.slot as i32;
                }
            }

            // Mirror ALL VRAM evictions & insertions globally into dev_map (48 KiB = ~2 us over PCIe Gen4!)
            // with 3-tier encoding (>=0 VRAM, -2 Tier-2 RAM, -1 Cold NVMe)!
            self.fill_host_map_tiered();
            let total_bytes = self.vram_cache.map_slice().len() * std::mem::size_of::<i32>();
            unsafe {
                self.copy_stream.htod_async(
                    self.dev_map.ptr(),
                    self.host_map.as_ptr(),
                    total_bytes,
                )?;
            }
            if let Some(sb) = scales.as_deref_mut() {
                for &key in self.vram_cache.current_keys() {
                    if let Some(vslot) = self.vram_cache.peek(key) {
                        sb.sync_slot_async(&self.copy_stream, key, vslot)?;
                    }
                }
            }
            if self.k2_dirty {
                let k2_bytes = self.vram_slots() * std::mem::size_of::<i32>();
                for p in 0..3 {
                    unsafe {
                        self.copy_stream.htod_async(
                            self.d_k2[p].ptr(),
                            self.h_k2[p].as_ptr(),
                            k2_bytes,
                        )?;
                    }
                }
                self.k2_dirty = false;
            }
            self.dev_map_dirty = false;
            self.copy_done.record(&self.copy_stream)?;
            compute_stream.wait(&self.copy_done)?;
        } else {
            // Protect any active CPU-offloaded RAM keys so subsequent `start_spec_prefetch` cannot evict them!
            if !extra_ram_protect.is_empty() {
                if self.bg_copy_pending {
                    self.copy_done.synchronize()?;
                    self.bg_copy_pending = false;
                }
                self.ram_cache.release();
                self.ram_cache.protect(extra_ram_protect)?;
            }
            let mut need_fence = self.bg_copy_pending;
            if self.dev_map_dirty || self.k2_dirty {
                if self.has_prev_compute {
                    self.copy_stream.wait(&self.prev_compute_done)?;
                }
                if self.dev_map_dirty {
                    self.fill_host_map_tiered();
                    let total_bytes = self.vram_cache.map_slice().len() * std::mem::size_of::<i32>();
                    unsafe {
                        self.copy_stream.htod_async(
                            self.dev_map.ptr(),
                            self.host_map.as_ptr(),
                            total_bytes,
                        )?;
                    }
                    self.dev_map_dirty = false;
                }
                if self.k2_dirty {
                    let k2_bytes = self.vram_slots() * std::mem::size_of::<i32>();
                    for p in 0..3 {
                        unsafe {
                            self.copy_stream.htod_async(
                                self.d_k2[p].ptr(),
                                self.h_k2[p].as_ptr(),
                                k2_bytes,
                            )?;
                        }
                    }
                    self.k2_dirty = false;
                }
                need_fence = true;
            }
            if let Some(sb) = scales.as_deref_mut() {
                // Even on 100% VRAM hit, ensure scales for all active slots are mirrored (no-op once cached).
                for &key in self.vram_cache.current_keys() {
                    if let Some(vslot) = self.vram_cache.peek(key) {
                        if sb.sync_slot_async(&self.copy_stream, key, vslot)? {
                            need_fence = true;
                        }
                    }
                }
            }
            if need_fence {
                self.copy_done.record(&self.copy_stream)?;
                compute_stream.wait(&self.copy_done)?;
            }
        }

        self.stats.disk_bytes = self.io.demand_stats.bytes + self.io.prefetch_stats.bytes + self.ahead_disk_bytes;
        #[cfg(debug_assertions)]
        {
            self.vram_cache.assert_invariants();
            self.ram_cache.assert_invariants();
        }
        Ok(())
    }

    /// Milestone 7: Asynchronous L+1 Tier-2 RAM -> Tier-1 VRAM Pre-Promotion!
    /// Called right AFTER Layer L's `exl3.run_moe()` is enqueued and BEFORE `mark_compute_done()`
    /// (so `prev_compute_done` still points to finished Layer L-1 and Layer L's active VRAM slots remain
    /// `leased = true` and CANNOT be evicted!).
    /// Overlaps L+1's 23.5 GB/s PCIe HtoD concurrently with Layer L MoE + Layer L+1 Attention!
    /// Never modifies GPU `dev_map` or `d_k2` while Layer L runs (`dev_map_dirty = true` flushes at L+1).
    pub fn promote_next_layer_ram_async(
        &mut self,
        ranked_keys: &[usize],
        max_promote: usize,
        scales: &mut scales::Exl3ScaleBank,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if ranked_keys.is_empty() || max_promote == 0 {
            return Ok(0);
        }
        let mut cands = [0usize; 16];
        let mut n_cands = 0usize;
        let cap = max_promote.min(cands.len());
        for &k in ranked_keys {
            if n_cands >= cap {
                break;
            }
            if self.vram_cache.peek(k).is_none() && self.ram_cache.peek(k).is_some() {
                cands[n_cands] = k;
                n_cands += 1;
            }
        }
        if n_cands == 0 {
            return Ok(0);
        }
        // Extend `ram_cache` leases to include our new DMA sources alongside Layer L's leased keys
        // so any subsequent `start_spec_prefetch` cannot evict a RAM buffer mid-DMA!
        let mut keep_ram = Vec::with_capacity(self.ram_cache.current_keys().len() + n_cands);
        keep_ram.extend_from_slice(self.ram_cache.current_keys());
        for &k in &cands[..n_cands] {
            if !keep_ram.contains(&k) {
                keep_ram.push(k);
            }
        }
        self.ram_cache.release();
        self.ram_cache.protect(&keep_ram)?;

        if self.has_prev_compute {
            self.copy_stream.wait(&self.prev_compute_done)?;
        }
        let mut promoted = 0usize;
        for &key in &cands[..n_cands] {
            if let Some(rslot) = self.ram_cache.peek(key) {
                if let Some(vticket) = self.vram_cache.reserve(key, true)? {
                    self.enqueue_htod(key, rslot, vticket.slot)?;
                    scales.sync_slot_async(&self.copy_stream, key, vticket.slot)?;
                    self.vram_cache.publish(vticket)?;
                    promoted += 1;
                }
            }
        }
        if promoted > 0 {
            self.dev_map_dirty = true;
            self.copy_done.record(&self.copy_stream)?;
            self.bg_copy_pending = true;
            self.stats.ram_hits += promoted as u64;
        }
        Ok(promoted)
    }

    /// Flush any deferred `dev_map` / `d_k2` updates before a zero-demand GPU pass (`DraftResident`/`WarpZeroSync`).
    pub fn flush_dev_map_if_dirty(&mut self, compute_stream: &Stream) -> Result<(), m3tis_cuda::CudaError> {
        if !self.dev_map_dirty && !self.k2_dirty && !self.bg_copy_pending {
            return Ok(());
        }
        if self.has_prev_compute {
            self.copy_stream.wait(&self.prev_compute_done)?;
        }
        if self.dev_map_dirty {
            let slice_map = self.vram_cache.map_slice();
            let total_bytes = slice_map.len() * std::mem::size_of::<i32>();
            self.host_map.as_i32_mut()[..slice_map.len()].copy_from_slice(slice_map);
            unsafe {
                self.copy_stream.htod_async(
                    self.dev_map.ptr(),
                    self.host_map.as_ptr(),
                    total_bytes,
                )?;
            }
            self.dev_map_dirty = false;
        }
        if self.k2_dirty {
            let k2_bytes = self.vram_slots() * std::mem::size_of::<i32>();
            for p in 0..3 {
                unsafe {
                    self.copy_stream.htod_async(
                        self.d_k2[p].ptr(),
                        self.h_k2[p].as_ptr(),
                        k2_bytes,
                    )?;
                }
            }
            self.k2_dirty = false;
        }
        self.copy_done.record(&self.copy_stream)?;
        compute_stream.wait(&self.copy_done)?;
        Ok(())
    }

    pub fn mark_compute_done(&mut self, compute_stream: &Stream) -> Result<(), m3tis_cuda::CudaError> {
        self.prev_compute_done.record(compute_stream)?;
        self.has_prev_compute = true;
        Ok(())
    }

    /// Non-blocking poll of any in-flight `prefetch_ring` completions (used by Oracle Self-Spec & Warp mode).
    pub fn poll_spec_only(&mut self) -> io::Result<()> {
        if self.spec.count > 0 {
            self.poll_spec_prefetch(false)?;
        }
        Ok(())
    }

    #[inline]
    pub fn spec_inflight_count(&self) -> usize {
        self.spec.count
    }

    // =========================================================================
    // Milestone 5B: Double-Buffered Expert-Major Streaming Prefill Ring API
    // =========================================================================

    #[inline]
    pub fn pf_stage_ptr_table(&self, comp: usize) -> CuDevicePtr {
        self.pf_stage_ptrs[comp].ptr()
    }

    #[inline]
    pub fn pf_stage_k2_ptr(&self, comp: usize) -> CuDevicePtr {
        self.pf_d_k2[comp].ptr()
    }

    #[inline]
    pub fn record_pf_vram_hits(&mut self, n: usize) {
        self.stats.vram_hits += n as u64;
    }

    /// Ensure any in-flight decode speculative prefetch or background promotion finishes before streaming prefill.
    pub fn pf_begin(&mut self, compute_stream: &Stream) -> Result<(), Box<dyn std::error::Error>> {
        if self.bg_copy_pending {
            self.copy_done.synchronize()?;
            self.bg_copy_pending = false;
        }
        if self.spec.count > 0 {
            self.poll_spec_prefetch(true)?;
        }
        if self.has_prev_compute {
            self.copy_stream.wait(&self.prev_compute_done)?;
            self.has_prev_compute = false;
        }
        // Also ensure all Tier-1 VRAM experts' scales are already mirrored (no-op if prewarmed)
        self.pf_copy_armed = [false; PF_RING_BUFS];
        self.pf_compute_armed = [false; PF_RING_BUFS];
        self.pf_io_inflight_sqes = 0;
        self.pf_io_inflight_experts = 0;
        let _ = compute_stream;
        Ok(())
    }

    /// Mirror any unsynced scales for Tier-1 VRAM hits used in a streaming prefill layer.
    pub fn pf_sync_vram_scales(
        &mut self,
        compute_stream: &Stream,
        vram_pairs: &[(usize, usize)], // (global_key, vram_slot)
        scales: &mut scales::Exl3ScaleBank,
    ) -> Result<(), m3tis_cuda::CudaError> {
        let mut any = false;
        for &(gkey, vslot) in vram_pairs {
            if scales.sync_slot_async(&self.copy_stream, gkey, vslot)? {
                any = true;
            }
        }
        if any {
            self.copy_done.record(&self.copy_stream)?;
            compute_stream.wait(&self.copy_done)?;
        }
        Ok(())
    }

    /// Non-blocking `io_uring` `read_fixed` submission of `keys` (`<= PF_WAVE_SLOTS`, sorted by file offset!)
    /// directly into `pf_stage_ram[buf]`. Returns in ~5 µs while NVMe streams at ~6.8 GB/s!
    pub fn pf_submit_nvme_wave(&mut self, buf: usize, keys: &[usize]) -> Result<(), Box<dyn std::error::Error>> {
        debug_assert!(buf < PF_RING_BUFS);
        debug_assert!(keys.len() <= PF_WAVE_SLOTS);
        debug_assert_eq!(self.pf_io_inflight_sqes, 0);
        if keys.is_empty() {
            return Ok(());
        }
        if self.pf_copy_armed[buf] {
            self.pf_copy_done[buf].synchronize()?;
            self.pf_copy_armed[buf] = false;
        }
        self.pf_io_ops.clear();
        let fd = self.file.fd();
        let slot_base = buf * PF_WAVE_SLOTS;
        for (i, &key) in keys.iter().enumerate() {
            let entry = &self.catalog.entries[key];
            let s = slot_base + i;
            for p in 0..3 {
                let comp = entry.components[p];
                let dst = unsafe { self.pf_stage_ram[p].as_mut_ptr().add(s * self.hot_strides[p]) };
                self.pf_io_ops.push(ReadOp {
                    fd,
                    offset: comp.offset,
                    dst,
                    len: comp.aligned_len,
                    user_data: (i * 3 + p) as u64,
                    fixed_buf_idx: self.pf_stage_ram_buf_idx[p],
                });
            }
        }
        let n = self.io.demand.submit_async(&self.pf_io_ops, &mut self.io.demand_stats)?;
        self.pf_io_inflight_sqes = n;
        self.pf_io_inflight_experts = keys.len();
        Ok(())
    }

    /// Block until the currently in-flight streaming prefill NVMe wave completes.
    pub fn pf_reap_nvme_wave(&mut self) -> io::Result<()> {
        let mut rem = self.pf_io_inflight_sqes;
        if rem == 0 {
            return Ok(());
        }
        let mut ud = [0u64; 64];
        let mut res = [0i32; 64];
        let mut first_err: Option<io::Error> = None;
        while rem > 0 {
            let got = self.io.demand.reap_async(
                &mut ud[..rem],
                &mut res[..rem],
                rem as u32,
                &mut self.io.demand_stats,
            )?;
            for i in 0..got {
                if res[i] < 0 && first_err.is_none() {
                    first_err = Some(io::Error::from_raw_os_error(-res[i]));
                }
            }
            rem -= got;
        }
        self.stats.disk_misses += self.pf_io_inflight_experts as u64;
        self.stats.vram_misses += self.pf_io_inflight_experts as u64;
        self.stats.disk_bytes = self.io.demand_stats.bytes + self.io.prefetch_stats.bytes + self.ahead_disk_bytes;
        self.pf_io_inflight_sqes = 0;
        self.pf_io_inflight_experts = 0;
        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(())
    }

    /// Copy `keys` (`<= PF_WAVE_SLOTS`) into `pf_stage_vram[buf]` on `copy_stream` (waiting for prior
    /// compute on `buf`), mirror scales & k2s, and make `compute_stream` wait for the copy!
    /// - `from_tier2_ram = true`: copies straight from `ram_slabs` (zero NVMe, zero RAM eviction!).
    /// - `from_tier2_ram = false`: copies from `pf_stage_ram[buf]` (freshly reaped NVMe wave!).
    pub fn pf_stage_htod_wave(
        &mut self,
        compute_stream: &Stream,
        buf: usize,
        keys: &[usize],
        from_tier2_ram: bool,
        scales: &scales::Exl3ScaleBank,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug_assert!(buf < PF_RING_BUFS);
        debug_assert!(keys.len() <= PF_WAVE_SLOTS);
        if keys.is_empty() {
            return Ok(());
        }
        if self.pf_compute_armed[buf] {
            self.copy_stream.wait(&self.pf_compute_done[buf])?;
        }
        // If copying from Tier-2 RAM into `pf_h_k2[buf]`, ensure previous copy on `buf` finished reading `pf_h_k2[buf]`!
        if self.pf_copy_armed[buf] {
            self.pf_copy_done[buf].synchronize()?;
            self.pf_copy_armed[buf] = false;
        }
        let slot_base = buf * PF_WAVE_SLOTS;
        for (i, &key) in keys.iter().enumerate() {
            let stage_slot = slot_base + i;
            let entry = &self.catalog.entries[key];
            let rslot = if from_tier2_ram {
                Some(self.ram_cache.peek(key).ok_or("pf_stage_htod_wave: key missing from Tier-2 RAM")?)
            } else {
                None
            };
            for p in 0..3 {
                let len = entry.components[p].aligned_len;
                let src = if let Some(rs) = rslot {
                    let off = self.ram_slot_byte_offset(p, rs);
                    unsafe { self.ram_slabs[p].as_ptr().add(off) }
                } else {
                    unsafe { self.pf_stage_ram[p].as_ptr().add(stage_slot * self.hot_strides[p]) }
                };
                let dst = self.pf_stage_vram[p].at(stage_slot * self.hot_strides[p]);
                unsafe {
                    self.copy_stream.htod_async(dst, src, len)?;
                }
                self.stats.htod_bytes += len as u64;
                self.pf_h_k2[buf][p].as_i32_mut()[i] = (entry.components[p].bits as i32) * 2;
            }
            scales.pf_copy_slot_async(&self.copy_stream, key, stage_slot)?;
        }
        for p in 0..3 {
            let dst = self.pf_d_k2[p].at(slot_base * 4);
            unsafe {
                self.copy_stream.htod_async(dst, self.pf_h_k2[buf][p].as_ptr(), keys.len() * 4)?;
            }
        }
        if from_tier2_ram {
            self.stats.ram_hits += keys.len() as u64;
            self.stats.vram_misses += keys.len() as u64;
        }
        self.pf_copy_done[buf].record(&self.copy_stream)?;
        self.pf_copy_armed[buf] = true;
        compute_stream.wait(&self.pf_copy_done[buf])?;
        Ok(())
    }

    pub fn pf_mark_wave_compute_done(
        &mut self,
        compute_stream: &Stream,
        buf: usize,
    ) -> Result<(), m3tis_cuda::CudaError> {
        debug_assert!(buf < PF_RING_BUFS);
        self.pf_compute_done[buf].record(compute_stream)?;
        self.pf_compute_armed[buf] = true;
        Ok(())
    }

    /// Finish streaming prefill cleanly: synchronize any remaining HtoD copy events before returning to decode.
    pub fn pf_end(&mut self) -> Result<(), m3tis_cuda::CudaError> {
        debug_assert_eq!(self.pf_io_inflight_sqes, 0);
        for b in 0..PF_RING_BUFS {
            if self.pf_copy_armed[b] {
                self.pf_copy_done[b].synchronize()?;
                self.pf_copy_armed[b] = false;
            }
            self.pf_compute_armed[b] = false;
        }
        self.pf_finish_ahead();
        Ok(())
    }

    /// Stop and drain any active Layer-Ahead NVMe streamer job and fold its stats into `self.stats`.
    pub fn pf_finish_ahead(&mut self) {
        if let Some(ref mut ah) = self.ahead {
            ah.finish_and_drain();
            let (bytes, exps) = ah.take_stats();
            self.ahead_disk_bytes += bytes;
            self.stats.disk_bytes = self.io.demand_stats.bytes + self.io.prefetch_stats.bytes + self.ahead_disk_bytes;
            self.stats.disk_misses += exps;
            self.stats.vram_misses += exps;
        }
    }

    /// Synchronize any in-flight `pf_copy_done` events and release all Layer-Ahead ring slots `< layer_end`.
    pub fn pf_sync_copies_and_release_ahead(&mut self, layer_end: usize) -> Result<(), m3tis_cuda::CudaError> {
        for b in 0..PF_RING_BUFS {
            if self.pf_copy_armed[b] {
                self.pf_copy_done[b].synchronize()?;
                self.pf_copy_armed[b] = false;
            }
        }
        if let Some(ref ah) = self.ahead {
            ah.release_up_to(layer_end);
        }
        Ok(())
    }

    /// Copy a wave of `<= PF_WAVE_SLOTS` experts straight from `LayerAheadStreamer` ring slots
    /// `(global_key, plan_idx)` into `pf_stage_vram[buf]` at 23.5 GB/s PCIe Gen4!
    /// When `release_tail_on_sync` is `Some(t)`, releases ring slots `< t` right after `pf_copy_done[buf]` finishes.
    pub fn pf_stage_htod_from_ahead(
        &mut self,
        compute_stream: &Stream,
        buf: usize,
        items: &[(usize, usize)],
        scales: &scales::Exl3ScaleBank,
        release_tail_on_sync: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug_assert!(buf < PF_RING_BUFS);
        debug_assert!(items.len() <= PF_WAVE_SLOTS);
        if items.is_empty() {
            return Ok(());
        }
        if self.pf_compute_armed[buf] {
            self.copy_stream.wait(&self.pf_compute_done[buf])?;
        }
        if self.pf_copy_armed[buf] {
            self.pf_copy_done[buf].synchronize()?;
            self.pf_copy_armed[buf] = false;
        }
        let ah = self.ahead.as_ref().ok_or("LayerAheadStreamer not initialized")?;
        if let Some(rel) = release_tail_on_sync {
            ah.release_up_to(rel);
        }
        let slot_base = buf * PF_WAVE_SLOTS;
        for (i, &(key, pidx)) in items.iter().enumerate() {
            let stage_slot = slot_base + i;
            let entry = &self.catalog.entries[key];
            for p in 0..3 {
                let len = entry.components[p].aligned_len;
                let src = ah.slot_ptr(p, pidx);
                let dst = self.pf_stage_vram[p].at(stage_slot * self.hot_strides[p]);
                unsafe {
                    self.copy_stream.htod_async(dst, src, len)?;
                }
                self.stats.htod_bytes += len as u64;
                self.pf_h_k2[buf][p].as_i32_mut()[i] = (entry.components[p].bits as i32) * 2;
            }
            scales.pf_copy_slot_async(&self.copy_stream, key, stage_slot)?;
        }
        for p in 0..3 {
            let dst = self.pf_d_k2[p].at(slot_base * 4);
            unsafe {
                self.copy_stream.htod_async(dst, self.pf_h_k2[buf][p].as_ptr(), items.len() * 4)?;
            }
        }
        self.pf_copy_done[buf].record(&self.copy_stream)?;
        self.pf_copy_armed[buf] = true;
        compute_stream.wait(&self.pf_copy_done[buf])?;
        Ok(())
    }

    /// Boot-time Exclusive Tier-1 VRAM + Tier-2 RAM Pre-Warming:
    /// 1. Sequentially streams the hottest Zipf experts evenly across all MoE layers straight from NVMe
    ///    into Tier-1 VRAM + scales + `dev_map` (~2.5s), then promotes the hottest 85% into scan-resistant `A_m`.
    /// 2. When `fill_ram` is true, streams the NEXT exclusive Zipf tier (disjoint from VRAM -> 0 duplication!)
    ///    straight into Tier-2 Pinned RAM (~3-4s at ~6.5 GB/s NVMe) and locks its hottest 85% in `A_m`.
    /// Together, Tier-1 VRAM + Tier-2 RAM hold ~45-50% of the entire 320B model hot from token #0!
    pub fn prewarm_vram(
        &mut self,
        compute_stream: &Stream,
        scales: &mut scales::Exl3ScaleBank,
        fill_cold: bool,
        fill_ram: bool,
    ) -> Result<(usize, usize, f64), Box<dyn std::error::Error>> {
        let t0 = std::time::Instant::now();
        let total = self.catalog.total_experts();
        let mut hot_keys: Vec<(u32, usize)> = Vec::new();
        let mut cold_keys: Vec<(u32, usize)> = Vec::new();
        let has_split = self.vram_cold_slots > 0;
        for k in 0..total {
            let e = &self.catalog.entries[k];
            let is_c = has_split && (0..3).all(|p| e.components[p].aligned_len < self.hot_strides[p]);
            if is_c {
                cold_keys.push((e.zipf_rank, k));
            } else {
                hot_keys.push((e.zipf_rank, k));
            }
        }
        // Sort by (per-layer zipf_rank, key) so every layer receives an equal share of its hottest experts!
        hot_keys.sort_unstable();
        cold_keys.sort_unstable();

        let take_hot = hot_keys.len().min(self.vram_hot_slots);
        let take_cold = if fill_cold { cold_keys.len().min(self.vram_cold_slots) } else { 0 };
        let mut plan: Vec<usize> = Vec::with_capacity(take_hot + take_cold);
        for &(_, k) in &hot_keys[..take_hot] {
            plan.push(k);
        }
        for &(_, k) in &cold_keys[..take_cold] {
            plan.push(k);
        }

        // Step 1: Batch-load 16 experts at a time (48 SQEs/batch -> saturates NVMe 990 PRO!) into Tier-1 VRAM.
        let fd = self.file.fd();
        let mut vram_loaded = 0usize;
        for chunk in plan.chunks(16) {
            self.ram_cache.release();
            self.vram_cache.release();
            self.ram_cache.protect(chunk)?;
            self.vram_cache.protect(chunk)?;
            self.demand_jobs.clear();
            self.demand_ops.clear();
            let mut batch_keys = Vec::with_capacity(chunk.len());
            for &key in chunk {
                if self.vram_cache.peek(key).is_some() {
                    continue;
                }
                if let Some(rticket) = self.ram_cache.reserve(key, false)? {
                    let entry = &self.catalog.entries[key];
                    for p in 0..3 {
                        let comp = entry.components[p];
                        let off = self.ram_slot_byte_offset(p, rticket.slot);
                        let fidx = self.ram_slot_fixed_buf_idx(p, rticket.slot);
                        let dst = unsafe { self.ram_slabs[p].as_mut_ptr().add(off) };
                        self.demand_ops.push(ReadOp {
                            fd,
                            offset: comp.offset,
                            dst,
                            len: comp.aligned_len,
                            user_data: (self.demand_jobs.len() * 3 + p) as u64,
                            fixed_buf_idx: fidx,
                        });
                    }
                    self.demand_jobs.push(rticket);
                    batch_keys.push(key);
                }
            }
            if !self.demand_ops.is_empty() {
                self.io.read_demand(&self.demand_ops)?;
                for &t in &self.demand_jobs {
                    self.ram_cache.publish(t)?;
                }
            }
            for &key in &batch_keys {
                if let Some(rslot) = self.ram_cache.peek(key) {
                    if let Some(vticket) = self.vram_cache.reserve(key, false)? {
                        self.enqueue_htod(key, rslot, vticket.slot)?;
                        scales.sync_slot_async(&self.copy_stream, key, vticket.slot)?;
                        self.vram_cache.publish(vticket)?;
                        vram_loaded += 1;
                    }
                }
            }
            // Wait for copy_stream to finish reading this chunk's ram_slabs before the next chunk can evict/overwrite them!
            self.copy_stream.synchronize()?;
        }
        self.ram_cache.release();
        self.vram_cache.release();

        // Soft-promote top 50% of pre-warmed VRAM experts into Protected (A_m) at hist_freq=2
        // (leaving 35% of A_m + 15% A1 immediately open for the active prompt's working set, and letting
        // any expert hit >=3 times cleanly displace stale prewarmed entries!).
        let v_hot_am = (take_hot * 50) / 100;
        for &(_, k) in &hot_keys[..v_hot_am] {
            self.vram_cache.promote_prewarm_soft(k);
        }
        let v_cold_am = (take_cold * 50) / 100;
        for &(_, k) in &cold_keys[..v_cold_am] {
            self.vram_cache.promote_prewarm_soft(k);
        }

        // Sync global dev_map + mixed k2 tables once
        let slice_map = self.vram_cache.map_slice();
        self.host_map.as_i32_mut()[..slice_map.len()].copy_from_slice(slice_map);
        unsafe {
            self.copy_stream.htod_async(
                self.dev_map.ptr(),
                self.host_map.as_ptr(),
                slice_map.len() * std::mem::size_of::<i32>(),
            )?;
        }
        if self.k2_dirty {
            let k2_bytes = self.vram_slots() * std::mem::size_of::<i32>();
            for p in 0..3 {
                unsafe {
                    self.copy_stream.htod_async(self.d_k2[p].ptr(), self.h_k2[p].as_ptr(), k2_bytes)?;
                }
            }
            self.k2_dirty = false;
        }
        self.copy_done.record(&self.copy_stream)?;
        compute_stream.wait(&self.copy_done)?;
        self.copy_stream.synchronize()?;

        // Step 2: Exclusive Tier-2 Pinned RAM Pre-Warming (next Zipf tier strictly disjoint from VRAM!).
        // Clear Step-1 bounce copies out of `ram_cache` first so VRAM duplicates NEVER occupy `A_m` RAM slots,
        // sort by pack offset for sequential ~6.7 GB/s NVMe streaming, and insert via `is_prefetch = true`
        // so all 100% of disjoint Tier-2 RAM slots fill cleanly with ZERO self-eviction!
        let mut ram_loaded = 0usize;
        if fill_ram {
            self.ram_cache.clear_entries_keep_freq();
            let rem_hot = &hot_keys[take_hot..];
            let rem_cold = &cold_keys[take_cold..];
            let r_take_hot = rem_hot.len().min(self.ram_hot_slots);
            let r_take_cold = if fill_cold { rem_cold.len().min(self.ram_cold_slots) } else { 0 };
            let mut ram_plan: Vec<usize> = Vec::with_capacity(r_take_hot + r_take_cold);
            for &(_, k) in &rem_hot[..r_take_hot] {
                ram_plan.push(k);
            }
            for &(_, k) in &rem_cold[..r_take_cold] {
                ram_plan.push(k);
            }
            ram_plan.sort_unstable_by_key(|&k| self.catalog.entries[k].components[0].offset);
            for chunk in ram_plan.chunks(16) {
                self.demand_jobs.clear();
                self.demand_ops.clear();
                for &key in chunk {
                    if self.ram_cache.peek(key).is_some() {
                        continue;
                    }
                    if let Some(rticket) = self.ram_cache.reserve(key, true)? {
                        let entry = &self.catalog.entries[key];
                        for p in 0..3 {
                            let comp = entry.components[p];
                            let off = self.ram_slot_byte_offset(p, rticket.slot);
                            let fidx = self.ram_slot_fixed_buf_idx(p, rticket.slot);
                            let dst = unsafe { self.ram_slabs[p].as_mut_ptr().add(off) };
                            self.demand_ops.push(ReadOp {
                                fd,
                                offset: comp.offset,
                                dst,
                                len: comp.aligned_len,
                                user_data: (self.demand_jobs.len() * 3 + p) as u64,
                                fixed_buf_idx: fidx,
                            });
                        }
                        self.demand_jobs.push(rticket);
                    }
                }
                if !self.demand_ops.is_empty() {
                    self.io.read_demand(&self.demand_ops)?;
                    for &t in &self.demand_jobs {
                        self.ram_cache.publish(t)?;
                        ram_loaded += 1;
                    }
                }
            }

            // Soft-promote top 50% of exclusive Tier-2 RAM experts into Protected (A_m) at hist_freq=2
            // so the active session's full working set locks into RAM A_m with zero warm-run thrashing!
            let r_hot_am = (r_take_hot * 50) / 100;
            for &(_, k) in &rem_hot[..r_hot_am] {
                self.ram_cache.promote_prewarm_soft(k);
            }
            let r_cold_am = (r_take_cold * 50) / 100;
            for &(_, k) in &rem_cold[..r_cold_am] {
                self.ram_cache.promote_prewarm_soft(k);
            }
            // Sync tiered dev_map (>=0 VRAM, -2 RAM, -1 cold) after Step-2 exclusive RAM prewarm!
            self.fill_host_map_tiered();
            unsafe {
                self.copy_stream.htod_async(
                    self.dev_map.ptr(),
                    self.host_map.as_ptr(),
                    self.vram_cache.map_slice().len() * std::mem::size_of::<i32>(),
                )?;
            }
            self.copy_done.record(&self.copy_stream)?;
            compute_stream.wait(&self.copy_done)?;
            self.copy_stream.synchronize()?;
        }

        self.stats.disk_bytes = self.io.demand_stats.bytes + self.io.prefetch_stats.bytes + self.ahead_disk_bytes;
        Ok((vram_loaded, ram_loaded, t0.elapsed().as_secs_f64()))
    }

    /// Zero-Stall Background Promotion (powers `--preset warp` / `WarpZeroSync`):
    /// - Reaps completed `prefetch_ring` NVMe reads non-blockingly into Tier-2 RAM.
    /// - Promotes up to `max_ram_promote` missing Tier-2 RAM-hit experts into Tier-1 VRAM on `copy_stream`
    ///   (hidden behind DFlash2 draft + Dense Layers 0..2; `wait_bg_promote()` fences right before Layer 3!).
    /// - Fires up to `max_nvme_prefetch` missing cold NVMe experts into `prefetch_ring` asynchronously!
    pub fn promote_async_background(
        &mut self,
        ranked_missing: &[usize],
        max_ram_promote: usize,
        max_nvme_prefetch: usize,
        scales: &mut scales::Exl3ScaleBank,
    ) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        // Ensure any previous round's async copy_stream HtoD has finished reading ram_slabs before
        // releasing/reusing RAM slots (~0 us after a 16+ ms compute token!).
        if self.bg_copy_pending {
            self.copy_done.synchronize()?;
            self.bg_copy_pending = false;
        }
        if self.spec.count > 0 {
            self.poll_spec_prefetch(false)?;
        }
        if ranked_missing.is_empty() {
            return Ok((0, 0));
        }

        // Split missing keys into ready Tier-2 RAM hits vs cold NVMe misses
        let mut ram_ready = Vec::new();
        let mut nvme_cold = Vec::new();
        for &k in ranked_missing {
            if self.vram_cache.peek(k).is_some() {
                continue;
            }
            if self.ram_cache.peek(k).is_some() {
                if ram_ready.len() < max_ram_promote {
                    ram_ready.push(k);
                }
            } else if self.ram_cache.pending(k).is_none() && nvme_cold.len() < max_nvme_prefetch {
                nvme_cold.push(k);
            }
        }

        // 1. Async RAM -> VRAM promotion on copy_stream (overlaps DFlash2 + Dense Layers 0..2!)
        // CRITICAL: keep `ram_ready` protected (`leased = true`) in `ram_cache` until the NEXT call's
        // `copy_done.synchronize()` so `start_spec_prefetch` CANNOT evict/overwrite a RAM buffer mid-DMA!
        let mut promoted = 0usize;
        self.ram_cache.release();
        if !ram_ready.is_empty() {
            if self.has_prev_compute {
                self.copy_stream.wait(&self.prev_compute_done)?;
            }
            self.vram_cache.release();
            self.ram_cache.protect(&ram_ready)?;
            self.vram_cache.protect(&ram_ready)?;
            for &key in &ram_ready {
                if let Some(rslot) = self.ram_cache.peek(key) {
                    if let Some(vticket) = self.vram_cache.reserve(key, false)? {
                        self.enqueue_htod(key, rslot, vticket.slot)?;
                        scales.sync_slot_async(&self.copy_stream, key, vticket.slot)?;
                        self.vram_cache.publish(vticket)?;
                        promoted += 1;
                    }
                }
            }
            self.vram_cache.release();
            if promoted > 0 {
                let slice_map = self.vram_cache.map_slice();
                self.host_map.as_i32_mut()[..slice_map.len()].copy_from_slice(slice_map);
                unsafe {
                    self.copy_stream.htod_async(
                        self.dev_map.ptr(),
                        self.host_map.as_ptr(),
                        slice_map.len() * std::mem::size_of::<i32>(),
                    )?;
                }
                if self.k2_dirty {
                    let k2_bytes = self.vram_slots() * std::mem::size_of::<i32>();
                    for p in 0..3 {
                        unsafe {
                            self.copy_stream.htod_async(
                                self.d_k2[p].ptr(),
                                self.h_k2[p].as_ptr(),
                                k2_bytes,
                            )?;
                        }
                    }
                    self.k2_dirty = false;
                }
                self.dev_map_dirty = false;
                self.copy_done.record(&self.copy_stream)?;
                self.bg_copy_pending = true;
                self.stats.ram_hits += promoted as u64;
            } else {
                self.ram_cache.release();
            }
        }

        // 2. Non-blocking NVMe prefetch on prefetch_ring (only submit if ring is currently idle!)
        // Protected `ram_ready` slots cannot be evicted while copy_stream reads them!
        let mut prefetched = 0usize;
        if !nvme_cold.is_empty() && self.spec.count == 0 {
            prefetched = self.start_spec_prefetch(&nvme_cold, max_nvme_prefetch)?;
        }
        self.stats.disk_bytes = self.io.demand_stats.bytes + self.io.prefetch_stats.bytes + self.ahead_disk_bytes;
        Ok((promoted, prefetched))
    }

    /// Fence `compute_stream` on any in-flight background RAM->VRAM promotion (called right before Layer 3,
    /// after DFlash2 + Dense Layers 0..2 have already hidden the PCIe DMA latency!).
    pub fn wait_bg_promote(&mut self, compute_stream: &Stream) -> Result<(), m3tis_cuda::CudaError> {
        if self.bg_copy_pending {
            compute_stream.wait(&self.copy_done)?;
        }
        Ok(())
    }

    fn enqueue_htod(&mut self, key: usize, ram_slot: usize, vram_slot: usize) -> Result<(), m3tis_cuda::CudaError> {
        let entry = &self.catalog.entries[key];
        for p in 0..3 {
            let len = entry.components[p].len;
            let roff = self.ram_slot_byte_offset(p, ram_slot);
            let voff = self.vram_slot_byte_offset(p, vram_slot);
            let src = unsafe { self.ram_slabs[p].as_ptr().add(roff) };
            let dst = self.vram_slabs[p].at(voff);
            unsafe {
                self.copy_stream.htod_async(dst, src, len)?;
            }
            if self.is_mixed {
                let want_k2 = (entry.components[p].bits as i32) * 2;
                let cur_k2 = &mut self.h_k2[p].as_i32_mut()[vram_slot];
                if *cur_k2 != want_k2 {
                    *cur_k2 = want_k2;
                    self.k2_dirty = true;
                }
            }
            self.stats.htod_bytes += len as u64;
        }
        Ok(())
    }

    /// Run our sm_86 GPU XOR checksum kernel over an expert currently resident in VRAM to verify
    /// zero-copy `NVMe -> Pinned RAM -> VRAM` bit-exactness against ground truth.
    pub fn vram_xor_checksum(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Stream,
        kernels: &RouterRemapKernels,
        key: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let vslot = self.vram_cache.peek(key).ok_or("expert not resident in VRAM")?;
        let d_out = ctx.alloc_device(4)?;
        let mut h_out = ctx.alloc_pinned(4)?;
        let entry = &self.catalog.entries[key];
        let mut total_xor = 0u32;
        for p in 0..3 {
            let words = (entry.components[p].len / 4) as u64;
            let ptr = self.vram_slabs[p].at(self.vram_slot_byte_offset(p, vslot));
            kernels.xor_checksum_u32(stream, ptr, d_out.ptr(), words)?;
            unsafe {
                stream.dtoh_async(h_out.as_mut_ptr(), d_out.ptr(), 4)?;
            }
            stream.synchronize()?;
            total_xor ^= h_out.as_i32_mut()[0] as u32;
        }
        Ok(total_xor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_q_lfu_shields_hot_super_experts_against_prefill_scans() {
        let num_keys = 6000;
        let cap = 1000;
        let hot_count = 700; // fits inside Am (80% of 1000 = 800 slots)

        let mut twoq = SlotCache::new(num_keys, cap, EvictionPolicy::TwoQLfu);
        let mut clock = SlotCache::new(num_keys, cap, EvictionPolicy::PlainClock);

        // Warm up 700 hot super-experts (touched twice -> promoted into Am in 2Q/LFU)
        for _pass in 0..3 {
            for k in 0..hot_count {
                twoq.release();
                twoq.protect(&[k]).unwrap();
                if let Some(t) = twoq.reserve(k, false).unwrap() {
                    twoq.publish(t).unwrap();
                }
                clock.release();
                clock.protect(&[k]).unwrap();
                if let Some(t) = clock.reserve(k, false).unwrap() {
                    clock.publish(t).unwrap();
                }
            }
        }
        assert_eq!(twoq.protected_count(), hot_count);

        // Now stream a 4,000-expert one-shot prefill scan (keys 1000..5000)
        for k in 1000..5000 {
            twoq.release();
            twoq.protect(&[k]).unwrap();
            if let Some(t) = twoq.reserve(k, false).unwrap() {
                twoq.publish(t).unwrap();
            }
            clock.release();
            clock.protect(&[k]).unwrap();
            if let Some(t) = clock.reserve(k, false).unwrap() {
                clock.publish(t).unwrap();
            }
        }
        twoq.release();
        clock.release();

        let twoq_survivors = (0..hot_count).filter(|&k| twoq.peek(k).is_some()).count();
        let clock_survivors = (0..hot_count).filter(|&k| clock.peek(k).is_some()).count();

        // 2Q/LFU keeps 100% of hot super-experts alive; plain CLOCK wipes 100% of them!
        assert_eq!(twoq_survivors, hot_count, "2Q/LFU must shield all 700 hot super-experts");
        assert_eq!(clock_survivors, 0, "Plain CLOCK gets wiped out by one-shot scan");
    }

    #[test]
    fn inflight_prefetch_joined_by_demand_zero_duplicate_read() {
        let mut c = SlotCache::new(64, 16, EvictionPolicy::TwoQLfu);
        // Start speculative prefetch for key 7
        let t = c.reserve(7, true).unwrap().expect("reserved prefetch");
        assert!(c.peek(7).is_none());
        assert!(c.pending(7).is_some());

        // Demand arrives for key 7 while prefetch is STILL in flight
        c.protect(&[7]).unwrap();
        assert!(c.reserve(7, false).unwrap().is_none(), "must NOT issue duplicate read!");

        // Prefetch finishes and publishes
        c.publish(t).unwrap();
        assert_eq!(c.peek(7), Some(t.slot));
        c.release();
        #[cfg(debug_assertions)]
        c.assert_invariants();
    }

    #[test]
    fn wtinylfu_resists_wide_verify_cyclic_thrash() {
        // 1,600 unique accesses/round (> 1,000 cache slots): 500 recurring hot experts + 1,100 rotating cold.
        // Without persistent hist_freq, 2Q/CLOCK suffers 0% hits; W-TinyLFU locks all 500 hot experts into Am!
        let mut wt = SlotCache::new(12096, 1000, EvictionPolicy::TwoQLfu);
        let mut clk = SlotCache::new(12096, 1000, EvictionPolicy::PlainClock);
        let mut wt_hits_r4 = 0usize;
        let mut clk_hits_r4 = 0usize;
        for round in 0..5 {
            for layer in 0..40 {
                let mut keys = Vec::with_capacity(40);
                // 12 hot experts per layer (= 480 total hot experts)
                for h in 0..12 {
                    keys.push(layer * 288 + h);
                }
                // 28 cold rotating experts per layer (= 1,120 cold/round -> 1,600 total > 1,000 cap!)
                for c in 0..28 {
                    keys.push(layer * 288 + 12 + ((round * 28 + c) % 270));
                }
                wt.release();
                wt.protect(&keys).unwrap();
                clk.release();
                clk.protect(&keys).unwrap();
                for &k in &keys {
                    if wt.peek(k).is_some() {
                        if round == 4 {
                            wt_hits_r4 += 1;
                        }
                    } else if let Some(t) = wt.reserve(k, false).unwrap() {
                        wt.publish(t).unwrap();
                    }
                    if clk.peek(k).is_some() {
                        if round == 4 {
                            clk_hits_r4 += 1;
                        }
                    } else if let Some(t) = clk.reserve(k, false).unwrap() {
                        clk.publish(t).unwrap();
                    }
                }
            }
        }
        assert!(
            wt_hits_r4 >= 470,
            "W-TinyLFU must lock >=470/480 hot experts despite 1600/round > 1000 cap, got {wt_hits_r4}"
        );
        assert!(
            clk_hits_r4 < 100,
            "Plain CLOCK should thrash on 1600 > 1000 cyclic scan, got {clk_hits_r4}"
        );
    }

    #[test]
    fn split_slot_cache_isolates_hot_3b_from_cold_2b_scans() {
        let mut is_cold = vec![true; 200];
        for i in 0..40 {
            is_cold[i] = false; // 40 hot 3b keys
        }
        let mut sc = SplitSlotCache::new(200, 40, 20, is_cold, EvictionPolicy::TwoQLfu);
        // Load all 40 hot keys -> slots 0..40
        for k in 0..40 {
            sc.release();
            sc.protect(&[k]).unwrap();
            let t = sc.reserve(k, false).unwrap().unwrap();
            assert!(t.slot < 40);
            sc.publish(t).unwrap();
        }
        // Flood 160 cold keys through cold partition (slots 40..60)
        for k in 40..200 {
            sc.release();
            sc.protect(&[k]).unwrap();
            let t = sc.reserve(k, false).unwrap().unwrap();
            assert!(t.slot >= 40 && t.slot < 60);
            sc.publish(t).unwrap();
        }
        sc.release();
        // ALL 40 hot 3b keys must survive 100% untouched!
        for k in 0..40 {
            assert!(sc.peek(k).is_some(), "hot key {k} was evicted by cold scan!");
        }
    }
}
