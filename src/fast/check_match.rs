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

pub struct MatchTarget {
    prefix_idx: [u8; MAX_PATTERN_LEN],
    prefix_len: u8,
    suffix_idx: [u8; MAX_PATTERN_LEN],
    suffix_len: u8,
    match_lut: &'static [u8; 58],
}

impl MatchTarget {
    pub fn new(
        prefix: &str,
        suffix: &str,
        case_insensitive: bool,
    ) -> Self {
        let alphabet = if case_insensitive {
            ALPHABET_CI
        } else {
            ALPHABET
        };
        let lut = match_lut(case_insensitive);

        let mut prefix_idx = [0u8; MAX_PATTERN_LEN];
        debug_assert!(prefix.len() <= MAX_PATTERN_LEN);
        for (i, &b) in prefix.as_bytes().iter().enumerate() {
            prefix_idx[i] = char_to_canonical(b, alphabet, lut);
        }

        let mut suffix_idx = [0u8; MAX_PATTERN_LEN];
        debug_assert!(suffix.len() <= MAX_PATTERN_LEN);
        for (i, &b) in suffix.as_bytes().iter().enumerate() {
            suffix_idx[i] = char_to_canonical(b, alphabet, lut);
        }

        Self {
            prefix_idx,
            prefix_len: prefix.len() as u8,
            suffix_idx,
            suffix_len: suffix.len() as u8,
            match_lut: lut,
        }
    }

    #[inline]
    pub fn matches(&self, bytes: &[u8; 32]) -> bool {
        check_match_32(
            bytes,
            &self.prefix_idx,
            self.prefix_len,
            &self.suffix_idx,
            self.suffix_len,
            &self.match_lut,
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
    prefix_idx: &[u8; MAX_PATTERN_LEN],
    prefix_len: u8,
    suffix_idx: &[u8; MAX_PATTERN_LEN],
    suffix_len: u8,
    match_lut: &[u8; 58],
) -> bool {
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

    for i in 0..prefix_len as usize {
        let target = prefix_idx[i];
        let rb_idx = skip + i;
        ensure_limb(
            &intermediate,
            &mut raw_base58,
            &mut limbs_done,
            rb_idx / 5,
        );
        if match_lut[raw_base58[rb_idx] as usize] != target {
            return false;
        }
    }

    if suffix_len > 0 {
        let suffix_len = suffix_len as usize;
        let tail_start = skip + encoded_length - suffix_len;
        let last_limb = (skip + encoded_length - 1) / 5;
        ensure_limb(
            &intermediate,
            &mut raw_base58,
            &mut limbs_done,
            last_limb,
        );
        for i in 0..suffix_len {
            let target = suffix_idx[i];
            if match_lut[raw_base58[tail_start + i] as usize] != target
            {
                return false;
            }
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
        let target = MatchTarget::new("zz", "LM", false);
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
        let target = MatchTarget::new("mithriL", "", true);
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
            let target = MatchTarget::new(prefix, suffix, false);
            assert!(target.matches(&bytes), "{key}");
        }
    }
}
