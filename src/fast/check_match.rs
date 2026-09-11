use fd_bs58::constants::{
    BINARY_SZ_32, ENC_TABLE_32, INTERMEDIATE_SZ_32, R1_DIV, RAW58_SZ_32,
};
use num_bigint::BigUint;
use num_traits::One;

pub const MAX_PATTERN_LEN: usize = 44;
pub const MAX_PATTERNS: usize = 64;

const ALPHABET: &[u8; 58] =
    b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const ALPHABET_CI: &[u8; 58] =
    b"123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";

const fn identity_lut() -> [u8; 58] {
    let mut lut = [0u8; 58];
    let mut i = 0;
    while i < 58 {
        lut[i] = i as u8;
        i += 1;
    }
    lut
}

const fn build_match_lut(alphabet: &[u8; 58]) -> [u8; 58] {
    let mut lut = [0u8; 58];
    let mut i = 0;
    while i < 58 {
        lut[i] = i as u8;
        let mut j = 0;
        while j < i {
            if alphabet[j] == alphabet[i] {
                lut[i] = j as u8;
                break;
            }
            j += 1;
        }
        i += 1;
    }
    lut
}

static MATCH_LUT_CS: [u8; 58] = identity_lut();
static MATCH_LUT_CI: [u8; 58] = build_match_lut(ALPHABET_CI);

#[inline]
pub fn match_lut(case_insensitive: bool) -> &'static [u8; 58] {
    if case_insensitive {
        &MATCH_LUT_CI
    } else {
        &MATCH_LUT_CS
    }
}

struct PatternIdx {
    prefix_ranges: Option<Vec<([u8; 32], [u8; 32])>>,
    prefix_idx: [u8; MAX_PATTERN_LEN],
    prefix_len: u8,
    suffix_idx: [u8; MAX_PATTERN_LEN],
    suffix_len: u8,
}

/// One or more prefix/suffix pairs. A pubkey matches if it matches any pair.
///
/// GPU blob layout (little-endian), shared with CUDA / OpenCL:
/// ```text
/// u32 n
/// u8  max_prefix_len
/// u8  max_suffix_len
/// n times: u8 plen, plen bytes, u8 slen, slen bytes
/// ```
/// Bytes are canonical raw_base58 indices, not ASCII.
pub struct MatchTargets {
    patterns: Vec<PatternIdx>,
    match_lut: &'static [u8; 58],
}

impl MatchTargets {
    pub fn new(
        patterns: &[(&str, &str)],
        case_insensitive: bool,
    ) -> Self {
        assert!(
            patterns.len() <= MAX_PATTERNS,
            "at most {MAX_PATTERNS} patterns"
        );
        let alphabet = if case_insensitive {
            ALPHABET_CI
        } else {
            ALPHABET
        };
        let lut = match_lut(case_insensitive);

        let mut out = Vec::with_capacity(patterns.len());
        for &(prefix, suffix) in patterns {
            debug_assert!(prefix.len() <= MAX_PATTERN_LEN);
            debug_assert!(suffix.len() <= MAX_PATTERN_LEN);
            let mut prefix_idx = [0u8; MAX_PATTERN_LEN];
            for (i, &b) in prefix.as_bytes().iter().enumerate() {
                prefix_idx[i] = char_to_canonical(b, alphabet, lut);
            }
            let mut suffix_idx = [0u8; MAX_PATTERN_LEN];
            for (i, &b) in suffix.as_bytes().iter().enumerate() {
                suffix_idx[i] = char_to_canonical(b, alphabet, lut);
            }
            out.push(PatternIdx {
                prefix_ranges: prefix_ranges(prefix, case_insensitive),
                prefix_idx,
                prefix_len: prefix.len() as u8,
                suffix_idx,
                suffix_len: suffix.len() as u8,
            });
        }

        Self {
            patterns: out,
            match_lut: lut,
        }
    }

    /// Bit `i` is set when pattern `i` matches. Empty set → no match.
    #[cfg(test)]
    pub fn match_mask(&self, bytes: &[u8; 32]) -> u64 {
        self.match_mask_active(bytes, u64::MAX)
    }

    /// Bit `i` is set when pattern `i` matches and that kind is in `active`.
    #[inline]
    pub fn match_mask_active(
        &self,
        bytes: &[u8; 32],
        active: u64,
    ) -> u64 {
        let mut mask = 0u64;
        let mut encode_active = 0u64;
        for (pi, p) in self.patterns.iter().enumerate() {
            if active & (1u64 << pi) == 0 {
                continue;
            }
            if let Some(ranges) = &p.prefix_ranges {
                if !ranges
                    .iter()
                    .any(|(lo, hi)| bytes >= lo && bytes <= hi)
                {
                    continue;
                }
                if p.suffix_len == 0 {
                    mask |= 1u64 << pi;
                    continue;
                }
            }
            encode_active |= 1u64 << pi;
        }
        if encode_active != 0 {
            mask |= check_match_32(
                bytes,
                &self.patterns,
                self.match_lut,
                encode_active,
            );
        }
        mask
    }

    /// Canonical-index blob consumed by `gpu_grind_init` / `gpu_keypair_init`.
    #[cfg(any(test, feature = "gpu"))]
    pub fn gpu_blob(&self) -> Vec<u8> {
        let max_prefix_len = self
            .patterns
            .iter()
            .map(|p| p.prefix_len)
            .max()
            .unwrap_or(0);
        let max_suffix_len = self
            .patterns
            .iter()
            .map(|p| p.suffix_len)
            .max()
            .unwrap_or(0);
        let mut v = Vec::with_capacity(
            6 + self
                .patterns
                .iter()
                .map(|p| {
                    2 + p.prefix_len as usize + p.suffix_len as usize
                })
                .sum::<usize>(),
        );
        v.extend_from_slice(
            &(self.patterns.len() as u32).to_le_bytes(),
        );
        v.push(max_prefix_len);
        v.push(max_suffix_len);
        for p in &self.patterns {
            v.push(p.prefix_len);
            v.extend_from_slice(&p.prefix_idx[..p.prefix_len as usize]);
            v.push(p.suffix_len);
            v.extend_from_slice(&p.suffix_idx[..p.suffix_len as usize]);
        }
        v
    }
}

#[inline]
fn char_to_canonical(
    c: u8,
    alphabet: &[u8; 58],
    match_lut: &[u8; 58],
) -> u8 {
    for (k, &ch) in alphabet.iter().enumerate() {
        if ch == c {
            return match_lut[k];
        }
    }
    255
}

// Exact prefixes can be checked as byte ranges without base58
// conversion. Keep the encode matcher for leading '1' and
// case-insensitive searches.
fn prefix_ranges(
    prefix: &str,
    ci: bool,
) -> Option<Vec<([u8; 32], [u8; 32])>> {
    if ci || prefix.is_empty() || prefix.starts_with('1') {
        return None;
    }
    let mut value = BigUint::from(0u8);
    for b in prefix.bytes() {
        value *= 58u8;
        value += ALPHABET.iter().position(|&c| c == b)?;
    }
    let minimum = BigUint::one() << 248usize;
    let maximum = (BigUint::one() << 256usize) - 1u8;
    let to_bytes = |n: &BigUint| {
        let v = n.to_bytes_be();
        let mut out = [0u8; 32];
        out[32 - v.len()..].copy_from_slice(&v);
        out
    };
    let mut ranges = Vec::new();
    let mut scale = BigUint::one();
    for _ in prefix.len()..=MAX_PATTERN_LEN {
        let lo = (&value * &scale).max(minimum.clone());
        let hi = ((&value + 1u8) * &scale - 1u8).min(maximum.clone());
        if lo <= hi {
            ranges.push((to_bytes(&lo), to_bytes(&hi)));
        }
        scale *= 58u8;
    }
    Some(ranges)
}

#[inline]
fn emit_limb(
    intermediate: &[u64; INTERMEDIATE_SZ_32],
    raw_base58: &mut [u8; RAW58_SZ_32],
    limb: usize,
) {
    let v = intermediate[limb] as u32;
    raw_base58[5 * limb + 4] = (v % 58) as u8;
    raw_base58[5 * limb + 3] = ((v / 58) % 58) as u8;
    raw_base58[5 * limb + 2] = ((v / 3364) % 58) as u8;
    raw_base58[5 * limb + 1] = ((v / 195_112) % 58) as u8;
    raw_base58[5 * limb] = (v / 11_316_496) as u8;
}

#[inline]
fn ensure_limb(
    intermediate: &[u64; INTERMEDIATE_SZ_32],
    raw_base58: &mut [u8; RAW58_SZ_32],
    limbs_done: &mut usize,
    need: usize,
) {
    while *limbs_done <= need {
        emit_limb(intermediate, raw_base58, *limbs_done);
        *limbs_done += 1;
    }
}

#[inline]
fn check_match_32(
    bytes: &[u8; 32],
    patterns: &[PatternIdx],
    match_lut: &[u8; 58],
    active: u64,
) -> u64 {
    let mut in_leading_0s = 0usize;
    while in_leading_0s < 32 && bytes[in_leading_0s] == 0 {
        in_leading_0s += 1;
    }

    let mut binary = [0u32; BINARY_SZ_32];
    for i in 0..BINARY_SZ_32 {
        let o = i * 4;
        binary[i] = u32::from_be_bytes([
            bytes[o],
            bytes[o + 1],
            bytes[o + 2],
            bytes[o + 3],
        ]);
    }

    let mut intermediate = [0u64; INTERMEDIATE_SZ_32];
    for i in 0..BINARY_SZ_32 {
        for j in 0..INTERMEDIATE_SZ_32 - 1 {
            intermediate[j + 1] +=
                u64::from(binary[i]) * ENC_TABLE_32[i][j];
        }
    }

    for i in (1..INTERMEDIATE_SZ_32).rev() {
        intermediate[i - 1] += intermediate[i] / R1_DIV;
        intermediate[i] %= R1_DIV;
    }

    let mut raw_base58 = [0u8; RAW58_SZ_32];
    let mut limbs_done = 0usize;

    ensure_limb(&intermediate, &mut raw_base58, &mut limbs_done, 0);

    let mut raw_leading_0s = 0usize;
    while raw_leading_0s < RAW58_SZ_32 {
        ensure_limb(
            &intermediate,
            &mut raw_base58,
            &mut limbs_done,
            raw_leading_0s / 5,
        );
        if raw_base58[raw_leading_0s] != 0 {
            break;
        }
        raw_leading_0s += 1;
    }

    let skip = raw_leading_0s - in_leading_0s;
    let encoded_length = RAW58_SZ_32 - skip;

    if patterns.is_empty() {
        return 1;
    }

    // Same per-char early-reject walk as the original single-pattern
    // matcher; extra patterns re-use already-emitted limbs.
    let mut mask = 0u64;
    for (pi, p) in patterns.iter().enumerate() {
        if active & (1u64 << pi) == 0 {
            continue;
        }
        let mut ok = true;
        for i in 0..p.prefix_len as usize {
            let rb_idx = skip + i;
            ensure_limb(
                &intermediate,
                &mut raw_base58,
                &mut limbs_done,
                rb_idx / 5,
            );
            if match_lut[raw_base58[rb_idx] as usize] != p.prefix_idx[i]
            {
                ok = false;
                break;
            }
        }
        if ok && p.suffix_len > 0 {
            let suffix_len = p.suffix_len as usize;
            let tail_start = skip + encoded_length - suffix_len;
            let last_limb = (skip + encoded_length - 1) / 5;
            ensure_limb(
                &intermediate,
                &mut raw_base58,
                &mut limbs_done,
                last_limb,
            );
            for i in 0..suffix_len {
                if match_lut[raw_base58[tail_start + i] as usize]
                    != p.suffix_idx[i]
                {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            mask |= 1u64 << pi;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::pubkey_of_seed;
    use sha2::{Digest, Sha512};

    fn encode_matches(
        bytes: &[u8; 32],
        prefix: &str,
        suffix: &str,
        ci: bool,
    ) -> bool {
        let s = fd_bs58::encode_32(*bytes);
        if ci {
            let lc: String =
                s.chars()
                    .map(|c| {
                        if c == 'L' {
                            c
                        } else {
                            c.to_ascii_lowercase()
                        }
                    })
                    .collect();
            (prefix.is_empty() || lc.starts_with(prefix))
                && (suffix.is_empty() || lc.ends_with(suffix))
        } else {
            (prefix.is_empty() || s.starts_with(prefix))
                && (suffix.is_empty() || s.ends_with(suffix))
        }
    }

    #[test]
    fn agrees_with_full_encode_random() {
        let target = MatchTargets::new(&[("zz", "LM")], false);
        for i in 0u32..5000 {
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let bytes = pubkey_of_seed(&seed);
            assert_eq!(
                target.match_mask(&bytes) != 0,
                encode_matches(&bytes, "zz", "LM", false),
                "seed {i}"
            );
        }
    }

    #[test]
    fn agrees_with_full_encode_ci() {
        let target = MatchTargets::new(&[("mithriL", "")], true);
        for i in 0u32..2000 {
            let h: [u8; 64] =
                Sha512::digest((i ^ 0xdeadbeef).to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let bytes = pubkey_of_seed(&seed);
            assert_eq!(
                target.match_mask(&bytes) != 0,
                encode_matches(&bytes, "mithriL", "", true),
                "seed {i}"
            );
        }
    }

    #[test]
    fn agrees_on_known_keys() {
        let keys = [
            "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr",
            "11111111111111111111111111111111",
            "1zfbgASTPZHoQ5DhqS5f2bnJk88rxMi137DmZowDztN",
        ];
        for key in keys {
            let bytes = fd_bs58::decode_32(key).unwrap();
            let prefix = &key[..key.len().min(3)];
            let suffix = &key[key.len().saturating_sub(2)..];
            let target = MatchTargets::new(&[(prefix, suffix)], false);
            assert!(target.match_mask(&bytes) != 0, "{key}");
        }
    }

    #[test]
    fn or_of_two_patterns() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        let miss =
            MatchTargets::new(&[("zzz", ""), ("aaa", "qq")], false);
        assert_eq!(miss.match_mask(&bytes), 0);
        let hit =
            MatchTargets::new(&[("zzz", ""), ("XkC", "KAr")], false);
        assert_eq!(hit.match_mask(&bytes), 0b10);
        let pref_only =
            MatchTargets::new(&[("nope", "nope"), ("XkC", "")], false);
        assert_eq!(pref_only.match_mask(&bytes), 0b10);
    }

    #[test]
    fn prefix_ranges_skip_leading_one_and_ci() {
        assert!(prefix_ranges("Harmonic", false).is_some());
        assert!(prefix_ranges("1abc", false).is_none());
        assert!(prefix_ranges("Harmonic", true).is_none());
        assert!(prefix_ranges("", false).is_none());
    }

    #[test]
    fn prefix_only_range_agrees_with_encode() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        let hit = MatchTargets::new(&[("XkC", "")], false);
        assert_eq!(hit.match_mask(&bytes), 0b1);
        let miss = MatchTargets::new(&[("XkD", "")], false);
        assert_eq!(miss.match_mask(&bytes), 0);
    }

    #[test]
    fn gpu_blob_roundtrip_header() {
        let t = MatchTargets::new(&[("ab", "xy"), ("Z", "")], false);
        let blob = t.gpu_blob();
        assert_eq!(&blob[..4], &2u32.to_le_bytes());
        assert_eq!(blob[4], 2); // max prefix
        assert_eq!(blob[5], 2); // max suffix
        assert_eq!(blob[6], 2); // plen of first
    }
}
