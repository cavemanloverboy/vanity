mod check_match;
mod field;
mod group;
pub mod sha512_simd;
pub mod simd;

use check_match::MatchTarget;
use field::{batch_invert, Fe};
use group::{edwards_d2, Niels, Point};

use sha2::{Digest, Sha512};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    OnceLock,
};

pub const BATCH: usize = 512;

// ─── base point ──────────────────────────────────────────────────────────────

const BASE: Point = Point {
    x: Fe([
        1738742601995546,
        1146398526822698,
        2070867633025821,
        562264141797630,
        587772402128613,
    ]),
    y: Fe([
        1801439850948184,
        1351079888211148,
        450359962737049,
        900719925474099,
        1801439850948198,
    ]),
    z: Fe([1, 0, 0, 0, 0]),
    t: Fe([
        1841354044333475,
        16398895984059,
        755974180946558,
        900171276175154,
        1821297809914039,
    ]),
};

// ─── fixed-base comb table ───────────────────────────────────────────────────

struct CombTable(Vec<[Niels; 8]>);

fn comb_table() -> &'static CombTable {
    static TABLE: OnceLock<CombTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let d2 = edwards_d2();
        let mut tables: Vec<[Niels; 8]> = Vec::with_capacity(64);
        let mut cur = BASE; // 16^i * B
        for _ in 0..64 {
            let cur_niels = cur.as_niels(&d2);
            let mut multiple = cur; // (k+1) * 16^i * B
            let mut row = [cur.as_niels(&d2); 8];
            for k in 1..8 {
                multiple = multiple.add_niels(&cur_niels);
                row[k] = multiple.as_niels(&d2);
            }
            tables.push(row);
            // cur *= 16
            cur = cur.double().double().double().double();
        }
        CombTable(tables)
    })
}

#[inline]
fn to_radix16(bytes: &[u8; 32]) -> [i8; 64] {
    let mut d = [0i8; 64];
    for i in 0..32 {
        d[2 * i] = (bytes[i] & 15) as i8;
        d[2 * i + 1] = ((bytes[i] >> 4) & 15) as i8;
    }
    let mut carry = 0i8;
    for i in 0..63 {
        d[i] += carry;
        carry = (d[i] + 8) >> 4;
        d[i] -= carry << 4;
    }
    d[63] += carry;
    d
}

#[inline]
fn scalarmult_base(scalar: &[u8; 32], table: &CombTable) -> Point {
    let digits = to_radix16(scalar);
    let mut acc = Point::IDENTITY;
    for i in 0..64 {
        let di = digits[i];
        if di > 0 {
            acc = acc.add_niels(&table.0[i][(di - 1) as usize]);
        } else if di < 0 {
            acc = acc.sub_niels(&table.0[i][(-di - 1) as usize]);
        }
    }
    acc
}

#[inline(always)]
fn clamp(h: &[u8; 64]) -> [u8; 32] {
    let mut s = [0u8; 32];
    s.copy_from_slice(&h[..32]);
    s[0] &= 248;
    s[31] &= 63;
    s[31] |= 64;
    s
}

fn batch_compress(points: &[Point], out: &mut [[u8; 32]]) {
    let n = points.len();
    let mut zs: Vec<Fe> = points.iter().map(|p| p.z).collect();
    batch_invert(&mut zs);
    for i in 0..n {
        let zinv = zs[i];
        let x = points[i].x.mul(&zinv);
        let y = points[i].y.mul(&zinv);
        let mut bytes = y.to_bytes();
        bytes[31] |= (x.is_odd() as u8) << 7;
        out[i] = bytes;
    }
}

fn keygen_batch(
    seeds: &mut [[u8; 32]; BATCH],
    used: &mut [[u8; 32]; BATCH],
    pubkeys: &mut [[u8; 32]; BATCH],
) {
    let table = comb_table();
    let mut points = [Point::IDENTITY; BATCH];
    for j in 0..BATCH {
        used[j] = seeds[j];
        let h: [u8; 64] = Sha512::digest(seeds[j]).into();
        let scalar = clamp(&h);
        points[j] = scalarmult_base(&scalar, table);
        seeds[j].copy_from_slice(&h[32..64]);
    }
    batch_compress(&points, pubkeys);
}

#[cfg(test)]
pub(crate) fn scalarmult_compress_scalar(
    scalar: &[u8; 32],
) -> [u8; 32] {
    let p = scalarmult_base(scalar, comb_table());
    let mut buf = [[0u8; 32]; 1];
    batch_compress(&[p], &mut buf);
    buf[0]
}

// ─── SIMD (AVX-512 IFMA) batch keygen ────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn simd_table() -> &'static [u64] {
    static T: OnceLock<[u64; simd::SIMD_TABLE_LEN]> = OnceLock::new();
    &T.get_or_init(simd::build_simd_table)[..]
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
unsafe fn keygen_batch_simd(
    seeds: &mut [[u8; 32]; BATCH],
    used: &mut [[u8; 32]; BATCH],
    pubkeys: &mut [[u8; 32]; BATCH],
) {
    let table = simd_table();
    const GROUPS: usize = BATCH / 8;
    debug_assert_eq!(BATCH % 8, 0);
    let mut points = [simd::Point8::identity(); GROUPS];
    for g in 0..GROUPS {
        let base = g * 8;
        for j in 0..8 {
            used[base + j] = seeds[base + j];
        }
        let chunk: &[[u8; 32]; 8] = (&seeds[base..base + 8])
            .try_into()
            .unwrap();
        let digests = sha512_simd::sha512_8(chunk);
        let mut scalars = [[0u8; 32]; 8];
        for j in 0..8 {
            scalars[j] = clamp(&digests[j]);
            seeds[base + j].copy_from_slice(&digests[j][32..64]);
        }
        points[g] = simd::scalarmult8(&scalars, table);
    }
    // Single inversion for all GROUPS*8 keys (Montgomery batch trick).
    simd::batch_compress(&points, &mut pubkeys[..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
unsafe fn grind_thread_simd(target: &MatchTarget, count: u32) {
    let mut seeds: [[u8; 32]; BATCH] = [[0u8; 32]; BATCH];
    for s in seeds.iter_mut() {
        *s = rand::random();
    }
    let mut used = [[0u8; 32]; BATCH];
    let mut pubkeys = [[0u8; 32]; BATCH];
    let mut local: u64 = 0;

    loop {
        if grind_done(count) {
            break;
        }
        keygen_batch_simd(&mut seeds, &mut used, &mut pubkeys);
        local += BATCH as u64;
        if local >= 4096 {
            GRIND_TOTAL.fetch_add(local, Ordering::Relaxed);
            local = 0;
        }
        for j in 0..BATCH {
            if target.matches(&pubkeys[j]) {
                let prev = GRIND_FOUND.fetch_add(1, Ordering::SeqCst);
                if prev < count {
                    let s = fd_bs58::encode_32(pubkeys[j]);
                    eprintln!("\r\x1b[Kmatch: {s}");
                    print_keypair(&used[j], &pubkeys[j], &s);
                }
            }
        }
    }
    GRIND_TOTAL.fetch_add(local, Ordering::Relaxed);
}

fn grind_thread_scalar(target: &MatchTarget, count: u32) {
    let mut seeds: [[u8; 32]; BATCH] = [[0u8; 32]; BATCH];
    for s in seeds.iter_mut() {
        *s = rand::random();
    }
    let mut used = [[0u8; 32]; BATCH];
    let mut pubkeys = [[0u8; 32]; BATCH];
    let mut local: u64 = 0;

    loop {
        if grind_done(count) {
            break;
        }
        keygen_batch(&mut seeds, &mut used, &mut pubkeys);
        local += BATCH as u64;
        if local >= 4096 {
            GRIND_TOTAL.fetch_add(local, Ordering::Relaxed);
            local = 0;
        }
        for j in 0..BATCH {
            if target.matches(&pubkeys[j]) {
                let prev = GRIND_FOUND.fetch_add(1, Ordering::SeqCst);
                if prev < count {
                    let s = fd_bs58::encode_32(pubkeys[j]);
                    eprintln!("\r\x1b[Kmatch: {s}");
                    print_keypair(&used[j], &pubkeys[j], &s);
                }
            }
        }
    }
    GRIND_TOTAL.fetch_add(local, Ordering::Relaxed);
}

#[inline(always)]
fn grind_done(count: u32) -> bool {
    GRIND_FOUND.load(Ordering::Relaxed) >= count
        || GRIND_ABORT.load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn pubkey_of_seed(seed: &[u8; 32]) -> [u8; 32] {
    let h: [u8; 64] = Sha512::digest(seed).into();
    let scalar = clamp(&h);
    let p = scalarmult_base(&scalar, comb_table());
    let mut buf = [[0u8; 32]; 1];
    batch_compress(&[p], &mut buf);
    buf[0]
}

// ─── multi-threaded grind ────────────────────────────────────────────────────

use rayon::iter::{IntoParallelIterator, ParallelIterator};

static GRIND_FOUND: AtomicU32 = AtomicU32::new(0);
static GRIND_TOTAL: AtomicU64 = AtomicU64::new(0);
static GRIND_ABORT: AtomicBool = AtomicBool::new(false);

pub fn reset_grind() {
    GRIND_FOUND.store(0, Ordering::SeqCst);
    GRIND_TOTAL.store(0, Ordering::SeqCst);
    GRIND_ABORT.store(false, Ordering::SeqCst);
}

pub fn request_abort() {
    GRIND_ABORT.store(true, Ordering::SeqCst);
}

#[cfg(feature = "gpu")]
pub fn add_attempts(n: u64) {
    GRIND_TOTAL.fetch_add(n, Ordering::Relaxed);
}

/// Returns the previous found count (caller should print only if `prev < count`).
#[cfg(feature = "gpu")]
pub fn note_found() -> u32 {
    GRIND_FOUND.fetch_add(1, Ordering::SeqCst)
}

#[cfg(feature = "gpu")]
pub fn is_done(count: u32) -> bool {
    grind_done(count)
}

pub fn total_attempts() -> u64 {
    GRIND_TOTAL.load(Ordering::Relaxed)
}

pub fn backend_name() -> &'static str {
    if simd::available() {
        "avx512-ifma (8-lane)"
    } else {
        "scalar"
    }
}

/// Run batched CPU keypair workers until `count` matches or abort.
/// Call [`reset_grind`] first if coordinating with a GPU thread that shares
/// these counters via [`add_attempts`] / [`note_found`] / [`is_done`].
pub fn run_cpu_workers(
    prefixes: &[String],
    suffixes: &[String],
    case_insensitive: bool,
    num_cpus: u32,
    count: u32,
) {
    let target = MatchTarget::new(prefixes, suffixes, case_insensitive);

    (0..num_cpus).into_par_iter().for_each(|_| {
        #[cfg(target_arch = "x86_64")]
        {
            if simd::available() {
                unsafe { grind_thread_simd(&target, count) };
            } else {
                grind_thread_scalar(&target, count);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        grind_thread_scalar(&target, count);
    });
}

fn print_keypair(seed: &[u8; 32], pubkey: &[u8; 32], pubkey_str: &str) {
    let seed_hex: String = seed
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    eprintln!("pubkey:   {pubkey_str}");
    eprintln!("seed hex: {seed_hex}");
    let json: Vec<u8> = seed
        .iter()
        .chain(pubkey.iter())
        .copied()
        .collect();
    eprintln!("keypair json (solana-compatible): {json:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn basepoint_compresses_to_canonical() {
        let mut out = [[0u8; 32]; 1];
        batch_compress(&[BASE], &mut out);
        // Canonical ed25519 base point compressed encoding.
        let mut expect = [0x66u8; 32];
        expect[0] = 0x58;
        assert_eq!(out[0], expect);
    }

    #[test]
    fn matches_dalek_for_random_seeds() {
        for i in 0u32..2000 {
            // Deterministic pseudo-random seeds.
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let ours = pubkey_of_seed(&seed);
            let theirs = SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes();
            assert_eq!(ours, theirs, "seed index {i}");
        }
    }

    #[test]
    fn batch_matches_singles() {
        let mut seeds = [[0u8; 32]; BATCH];
        for j in 0..BATCH {
            let h: [u8; 64] =
                Sha512::digest((j as u64).to_le_bytes()).into();
            seeds[j] = h[..32].try_into().unwrap();
        }
        let mut used = [[0u8; 32]; BATCH];
        let mut pubs = [[0u8; 32]; BATCH];
        let mut seeds_copy = seeds;
        keygen_batch(&mut seeds_copy, &mut used, &mut pubs);
        for j in 0..BATCH {
            assert_eq!(used[j], seeds[j]);
            let theirs = SigningKey::from_bytes(&seeds[j])
                .verifying_key()
                .to_bytes();
            assert_eq!(pubs[j], theirs, "lane {j}");
        }
    }
}
