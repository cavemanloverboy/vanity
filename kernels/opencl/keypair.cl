/* keypair.cl — OpenCL port of the keypair grind kernel from
   kernels/vanity_keypair.cu.

   Each work-item derives a 32-byte seed from (host_seed || idx), then walks
   a chain of ed25519 keypairs: SHA-512(seed) -> clamp -> scalar-mult base ->
   compress -> base58, testing each pubkey against the prefix/suffix. The
   next seed is the high half of the SHA-512 output (matching the CUDA loop).

   Same OpenCL-driven changes as the grind kernel: host-provided max_iters
   instead of clock64(), and a per-work-item counts buffer summed on the
   host instead of a 64-bit atomic. */

static bool kp_matches_target(const uchar *a,
                              __global const uchar *prefix, uint prefix_len,
                              __global const uchar *suffix, uint suffix_len,
                              ulong encoded_len) {
    for (uint i = 0; i < prefix_len; i++)
        if (a[i] != prefix[i]) return false;
    for (uint i = 0; i < suffix_len; i++)
        if (a[encoded_len - suffix_len + i] != suffix[i]) return false;
    return true;
}

__kernel void vanity_keypair_search(
    __global const uchar *host_seed,     /* 32 bytes */
    __global const uchar *prefix, uint prefix_len,
    __global const uchar *suffix, uint suffix_len,
    int case_insensitive,
    __global uchar *out,                 /* 32 bytes: matched seed */
    __global volatile int *done,
    __global uint *counts,
    uint max_iters)
{
    ulong idx = get_global_id(0);

    uchar seed[32];
    uchar privatek[64];
    uchar pubkey[32];
    uchar encoded[45];
    ge_p3 A;

    /* seed = SHA-256(host_seed[32] || idx[8 LE]). Copy host_seed into
       private memory first: cuda_sha256_update takes a private pointer and
       OpenCL 1.2 has no generic address space. */
    {
        uchar hs[32];
        for (int i = 0; i < 32; ++i) hs[i] = host_seed[i];
        uchar idx_bytes[8];
        for (int b = 0; b < 8; ++b) idx_bytes[b] = (uchar)(idx >> (8*b));
        CUDA_SHA256_CTX sha;
        cuda_sha256_init(&sha);
        cuda_sha256_update(&sha, hs, 32);
        cuda_sha256_update(&sha, idx_bytes, 8);
        cuda_sha256_final(&sha, seed);
    }

    uint iter = 0;
    for (; iter < max_iters; iter++) {
        if ((iter % 100) == 0 && atomic_max(done, 0) == 1) break;

        sha512_context md;
        md.curlen = 0;
        md.length = 0;
        md.state[0] = 0x6a09e667f3bcc908UL;
        md.state[1] = 0xbb67ae8584caa73bUL;
        md.state[2] = 0x3c6ef372fe94f82bUL;
        md.state[3] = 0xa54ff53a5f1d36f1UL;
        md.state[4] = 0x510e527fade682d1UL;
        md.state[5] = 0x9b05688c2b3e6c1fUL;
        md.state[6] = 0x1f83d9abfb41bd6bUL;
        md.state[7] = 0x5be0cd19137e2179UL;
        for (int i = 0; i < 32; i++) md.buf[i] = seed[i];
        md.curlen = 32;

        sha512_final(&md, privatek);

        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(pubkey, &A);

        ulong enc_len = fd_base58_encode_32(pubkey, encoded, case_insensitive != 0);

        if (kp_matches_target(encoded, prefix, prefix_len, suffix, suffix_len, enc_len)) {
            if (atomic_cmpxchg(done, 0, 1) == 0)
                for (int i = 0; i < 32; i++) out[i] = seed[i];
            iter++;          /* count the matching attempt */
            break;
        }

        for (int i = 0; i < 32; i++) seed[i] = privatek[32 + i];
    }

    counts[idx] = iter;
}
