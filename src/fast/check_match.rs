use fd_bs58::constants::{
    BINARY_SZ_32, ENC_TABLE_32, INTERMEDIATE_SZ_32, R1_DIV, RAW58_SZ_32,
};

pub const MAX_PATTERN_LEN: usize = 44;

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

struct Pattern {
    indices: [u8; MAX_PATTERN_LEN],
    len: u8,
}

pub struct MatchTarget {
    prefixes: Vec<Pattern>,
    suffixes: Vec<Pattern>,
    match_lut: &'static [u8; 58],
}

impl MatchTarget {
    pub fn new(
        prefixes: &[String],
        suffixes: &[String],
        case_insensitive: bool,
    ) -> Self {
        let alphabet = if case_insensitive {
            ALPHABET_CI
        } else {
            ALPHABET
        };
        let lut = match_lut(case_insensitive);

        let encode = |pattern: &String| {
            let mut indices = [0u8; MAX_PATTERN_LEN];
            debug_assert!(pattern.len() <= MAX_PATTERN_LEN);
            for (i, &byte) in pattern.as_bytes().iter().enumerate() {
                indices[i] = char_to_canonical(byte, alphabet, lut);
            }
            Pattern {
                indices,
                len: pattern.len() as u8,
            }
        };

        Self {
            prefixes: prefixes.iter().map(encode).collect(),
            suffixes: suffixes.iter().map(encode).collect(),
            match_lut: lut,
        }
    }

    #[inline]
    pub fn matches(&self, bytes: &[u8; 32]) -> bool {
        check_match_32(
            bytes,
            &self.prefixes,
            &self.suffixes,
            self.match_lut,
        )
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
    prefixes: &[Pattern],
    suffixes: &[Pattern],
    match_lut: &[u8; 58],
) -> bool {
    let mut in_leading_0s = 0usize;
    while in_leading_0s < 32 && bytes[in_leading_0s] == 0 {
        in_leading_0s += 1;
    }

    let mut binary = [0u32; BINARY_SZ_32];
    for (i, word) in binary.iter_mut().enumerate() {
        let o = i * 4;
        *word = u32::from_be_bytes([
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

    if !prefixes.is_empty() {
        let mut any_prefix = false;
        for pattern in prefixes {
            let pattern_len = pattern.len as usize;
            if pattern_len > encoded_length {
                continue;
            }
            let mut matched = true;
            for i in 0..pattern_len {
                let rb_idx = skip + i;
                ensure_limb(
                    &intermediate,
                    &mut raw_base58,
                    &mut limbs_done,
                    rb_idx / 5,
                );
                if match_lut[raw_base58[rb_idx] as usize]
                    != pattern.indices[i]
                {
                    matched = false;
                    break;
                }
            }
            if matched {
                any_prefix = true;
                break;
            }
        }
        if !any_prefix {
            return false;
        }
    }

    if !suffixes.is_empty() {
        let last_limb = (skip + encoded_length - 1) / 5;
        ensure_limb(
            &intermediate,
            &mut raw_base58,
            &mut limbs_done,
            last_limb,
        );
        let mut any_suffix = false;
        for pattern in suffixes {
            let pattern_len = pattern.len as usize;
            if pattern_len > encoded_length {
                continue;
            }
            let tail_start = skip + encoded_length - pattern_len;
            let mut matched = true;
            for i in 0..pattern_len {
                if match_lut[raw_base58[tail_start + i] as usize]
                    != pattern.indices[i]
                {
                    matched = false;
                    break;
                }
            }
            if matched {
                any_suffix = true;
                break;
            }
        }
        if !any_suffix {
            return false;
        }
    }

    true
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
        let target = MatchTarget::new(
            &["zz".to_string()],
            &["LM".to_string()],
            false,
        );
        for i in 0u32..5000 {
            let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let bytes = pubkey_of_seed(&seed);
            assert_eq!(
                target.matches(&bytes),
                encode_matches(&bytes, "zz", "LM", false),
                "seed {i}"
            );
        }
    }

    #[test]
    fn agrees_with_full_encode_ci() {
        let target = MatchTarget::new(&["mithriL".to_string()], &[], true);
        for i in 0u32..2000 {
            let h: [u8; 64] =
                Sha512::digest((i ^ 0xdeadbeef).to_le_bytes()).into();
            let seed: [u8; 32] = h[..32].try_into().unwrap();
            let bytes = pubkey_of_seed(&seed);
            assert_eq!(
                target.matches(&bytes),
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
            let target = MatchTarget::new(
                &[prefix.to_string()],
                &[suffix.to_string()],
                false,
            );
            assert!(target.matches(&bytes), "{key}");
        }
    }

    #[test]
    fn matches_any_prefix_and_any_suffix() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        let target = MatchTarget::new(
            &["sun".to_string(), "XkC".to_string()],
            &["mint".to_string(), "KAr".to_string()],
            false,
        );
        assert!(target.matches(&bytes));

        let wrong_suffix = MatchTarget::new(
            &["sun".to_string(), "XkC".to_string()],
            &["mint".to_string(), "moon".to_string()],
            false,
        );
        assert!(!wrong_suffix.matches(&bytes));
    }

    #[test]
    fn matches_multiple_case_insensitive_patterns() {
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        let bytes = fd_bs58::decode_32(key).unwrap();
        let target = MatchTarget::new(
            &["sun".to_string(), "xkc".to_string()],
            &["mint".to_string(), "kar".to_string()],
            true,
        );
        assert!(target.matches(&bytes));
    }
}
