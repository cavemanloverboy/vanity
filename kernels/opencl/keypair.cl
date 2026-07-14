/* keypair.cl — OpenCL port of the keypair grind kernel from
   kernels/vanity_keypair.cu.

   Each work-item derives a 32-byte seed from (host_seed || idx), then walks
   a chain of ed25519 keypairs: SHA-512(seed) -> clamp -> radix-32 comb
   scalarmult -> batch compress -> fused base58 early-reject match. The next
   seed is the high half of the SHA-512 output (matching the CUDA loop).

   Keys are processed in batches of KP_BATCH: one Montgomery invert covers
   the whole batch (same amortization as the CPU fast path). Fixed-base
   mult uses a radix-32 comb table (52 windows) built once at init.

   Same OpenCL-driven changes as the grind kernel: host-provided max_iters
   instead of clock64(), and a per-work-item counts buffer summed on the
   host instead of a 64-bit atomic. Prefix/suffix are canonical raw_base58
   indices; match_lut folds case-insensitive aliases (same as vanity.cl). */

#ifndef KP_BATCH
#define KP_BATCH KP_BATCH_MAX
#endif

__kernel void vanity_keypair_search(
    __global const uchar *host_seed,     /* 32 bytes */
    __constant const uchar *match_lut,   /* 58 */
    __global const uchar *prefixes, uint prefix_count,
    __global const uchar *suffixes, uint suffix_count,
    __global uchar *out,                 /* 32 bytes: matched seed */
    __global volatile int *done,
    __global uint *counts,
    __global const ge_niels *comb,       /* COMB_TABLE_LEN entries */
    uint max_iters)
{
    ulong idx = get_global_id(0);

    uchar seed[32];
    uchar privatek[64];
    uchar pubkey[32];
    uchar batch_seeds[KP_BATCH][32];
    fe Xs[KP_BATCH];
    fe Ys[KP_BATCH];
    fe Zs[KP_BATCH];
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
    while (iter < max_iters) {
        if (atomic_max(done, 0) == 1) break;

        int n = KP_BATCH;
        if ((uint)n > max_iters - iter) n = (int)(max_iters - iter);

        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 32; i++) batch_seeds[j][i] = seed[i];

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

            ge_scalarmult_base_comb(&A, privatek, comb);
            fe_copy(Xs[j], A.X);
            fe_copy(Ys[j], A.Y);
            fe_copy(Zs[j], A.Z);

            for (int i = 0; i < 32; i++) seed[i] = privatek[32 + i];
        }

        fe_batch_invert(Zs, n);

        int matched = -1;
        for (int j = 0; j < n; j++) {
            ge_p3_tobytes_inv(pubkey, Xs[j], Ys[j], Zs[j]);

            uint pubkey_words[8];
            for (int k = 0; k < 8; ++k) {
                pubkey_words[k] = ((uint)pubkey[4*k    ] << 24)
                                | ((uint)pubkey[4*k + 1] << 16)
                                | ((uint)pubkey[4*k + 2] <<  8)
                                | ((uint)pubkey[4*k + 3]      );
            }

            if (fd_base58_check_match_32_words(pubkey_words, prefixes, prefix_count,
                                               suffixes, suffix_count, match_lut)) {
                if (atomic_cmpxchg(done, 0, 1) == 0)
                    for (int i = 0; i < 32; i++) out[i] = batch_seeds[j][i];
                matched = j;
                break;
            }
        }

        if (matched >= 0) {
            iter += (uint)(matched + 1);
            break;
        }
        iter += (uint)n;
    }

    counts[idx] = iter;
}
