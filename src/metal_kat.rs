use ed25519_dalek::SigningKey;
use sha2::{Digest, Sha256, Sha512};
use std::ffi::c_void;
use std::time::Duration;

use crate::fast::MatchTargets;

extern "C" {
    fn vanity_metal_pubkey(seed: *const u8, out: *mut u8) -> i32;
    fn gpu_grind_init(
        id: i32,
        base: *const u8,
        owner: *const u8,
        patterns: *const u8,
        patterns_len: u64,
        case_insensitive: bool,
    ) -> *mut c_void;
    fn gpu_grind_launch(ctx: *mut c_void, seed: *const u8);
    fn gpu_grind_query(ctx: *mut c_void) -> i32;
    fn gpu_grind_read(ctx: *mut c_void, out: *mut u8);
    fn gpu_grind_destroy(ctx: *mut c_void);
}

#[test]
fn metal_pubkeys_match_dalek() {
    let mut seeds = Vec::new();
    // RFC 8032 secret (the ed25519 seed).
    seeds.push([
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c, 0xc4,
        0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae, 0x7f, 0x60,
    ]);
    for i in 0u32..8 {
        let h: [u8; 64] = Sha512::digest(i.to_le_bytes()).into();
        seeds.push(h[..32].try_into().unwrap());
    }
    for (i, seed) in seeds.iter().enumerate() {
        let mut out = [0u8; 32];
        let rc = unsafe { vanity_metal_pubkey(seed.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(rc, 0, "metal pubkey failed for seed {i}");
        let theirs = SigningKey::from_bytes(seed).verifying_key().to_bytes();
        assert_eq!(out, theirs, "metal pubkey mismatch for seed {i}");
    }
}

#[test]
fn metal_grind_seed_rehashes_to_the_pattern() {
    let base = [0u8; 32];
    let owner = [1u8; 32];
    let targets = MatchTargets::new(&[("a", "")], false);
    let blob = targets.gpu_blob();
    let ctx = unsafe {
        gpu_grind_init(
            0,
            base.as_ptr(),
            owner.as_ptr(),
            blob.as_ptr(),
            blob.len() as u64,
            false,
        )
    };
    assert!(!ctx.is_null());
    let host_seed = [9u8; 32];
    unsafe { gpu_grind_launch(ctx, host_seed.as_ptr()) };
    let start = std::time::Instant::now();
    while unsafe { gpu_grind_query(ctx) } == 0 {
        if start.elapsed() > Duration::from_secs(20) {
            unsafe { gpu_grind_destroy(ctx) };
            panic!("grind kernel did not finish");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let mut out = [0u8; 24];
    unsafe {
        gpu_grind_read(ctx, out.as_mut_ptr());
        gpu_grind_destroy(ctx);
    }
    assert_ne!(&out[..16], &[0u8; 16], "gpu grind returned no seed");
    let hash: [u8; 32] = Sha256::new()
        .chain_update(base)
        .chain_update(&out[..16])
        .chain_update(owner)
        .finalize()
        .into();
    let addr = fd_bs58::encode_32(hash);
    assert!(
        addr.starts_with('a'),
        "seed did not hash to an 'a' address: {addr}"
    );
}
