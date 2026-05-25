/* doppler.cl — OpenCL "doppler" keypair grind kernel.

   Like the keypair kernel (ed25519: SHA-512 -> clamp -> scalar-mult base ->
   compress), but the match is the doppler pattern rather than a base58
   prefix, so no base58 encode is needed.

   The 32-byte pubkey is split into four 8-byte segments. A segment is
   "sign-extendable" if its low 4 bytes form an i32 (little-endian) and its
   high 4 bytes are that i32's sign extension: all 0x00 when bit 31 is clear,
   all 0xFF when bit 31 is set. A key matches when at least `required_segments`
   of the four segments are sign-extendable. */

static uint doppler_count(const uchar *pk) {
    uint matched = 0;
    for (int s = 0; s < 4; ++s) {
        int o = s * 8;
        uchar fill = (pk[o + 3] & 0x80) ? 0xFF : 0x00;
        if (pk[o + 4] == fill && pk[o + 5] == fill &&
            pk[o + 6] == fill && pk[o + 7] == fill)
            matched++;
    }
    return matched;
}

__kernel void vanity_doppler_search(
    __global const uchar *host_seed,     /* 32 bytes */
    uint required_segments,
    __global uchar *out,                 /* 32 bytes: matched seed */
    __global volatile int *done,
    __global uint *counts,
    uint max_iters)
{
    ulong idx = get_global_id(0);

    uchar seed[32];
    uchar privatek[64];
    uchar pubkey[32];
    ge_p3 A;

    /* seed = SHA-256(host_seed[32] || idx[8 LE]); host_seed copied to private
       memory because cuda_sha256_update takes a private pointer. */
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

        if (doppler_count(pubkey) >= required_segments) {
            if (atomic_cmpxchg(done, 0, 1) == 0)
                for (int i = 0; i < 32; i++) out[i] = seed[i];
            iter++;          /* count the matching attempt */
            break;
        }

        for (int i = 0; i < 32; i++) seed[i] = privatek[32 + i];
    }

    counts[idx] = iter;
}
