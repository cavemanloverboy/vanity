/* doppler32.cl — grind-doppler on the Apple GPU path.

   Same chain as vanity_keypair_search32 (SHA-512, 13-bit comb, one inversion
   per work-group) but the match is the doppler segment count, so there is no
   base58 step. A segment is sign-extendable when its high 4 bytes repeat the
   sign bit of its low 4 bytes. */

static uint doppler_count(const uchar *pk) {
    uint matched = 0;
    for (int s = 0; s < 4; s++) {
        int o = s * 8;
        uchar fill = (pk[o + 3] & 0x80) ? 0xFF : 0x00;
        if (pk[o + 4] == fill && pk[o + 5] == fill &&
            pk[o + 6] == fill && pk[o + 7] == fill)
            matched++;
    }
    return matched;
}

__kernel void vanity_doppler_search32(
    __global const uchar *host_seed,     /* 32 bytes */
    uint required_segments,
    __global uchar *out,                 /* 32 bytes: matched seed */
    __global volatile int *done,
    __global uint *counts,
    __global const ge32_precomp *comb,   /* COMB32_TABLE_LEN entries */
    uint max_iters,
    __local uint *tree)                  /* 2 * group size fe32 */
{
    ulong idx = get_global_id(0);

    uchar seed[32];
    uchar batch_start[32];
    uchar privatek[64];
    uchar pubkey[32];
    fe32 Xs[KP32_BATCH];
    fe32 Ys[KP32_BATCH];
    fe32 Zs[KP32_BATCH];
    ge32_p2 A;

    {
        uchar hs[32];
        for (int i = 0; i < 32; i++) hs[i] = host_seed[i];
        uchar idx_bytes[8];
        for (int b = 0; b < 8; b++) idx_bytes[b] = (uchar)(idx >> (8 * b));
        CUDA_SHA256_CTX sha;
        cuda_sha256_init(&sha);
        cuda_sha256_update(&sha, hs, 32);
        cuda_sha256_update(&sha, idx_bytes, 8);
        cuda_sha256_final(&sha, seed);
    }

    /* The group decides together whether to run another batch, because every
       work-item must reach fe32_group_invert's barriers equally often. */
    __local int stop;
    uint iter = 0;
    for (;;) {
        if (get_local_id(0) == 0) stop = iter >= max_iters || atomic_max(done, 0) == 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (stop) break;

        int n = KP32_BATCH;
        if ((uint)n > max_iters - iter) n = (int)(max_iters - iter);

        for (int i = 0; i < 32; i++) batch_start[i] = seed[i];
        for (int j = 0; j < n; j++) {
            sha512_32(privatek, seed);
            clamp_scalar(privatek);

            ge32_scalarmult_base_comb(&A, privatek, comb);
            fe32_copy(Xs[j], A.X);
            fe32_copy(Ys[j], A.Y);
            fe32_copy(Zs[j], A.Z);

            for (int i = 0; i < 32; i++) seed[i] = privatek[32 + i];
        }

        fe32 prefix_products[KP32_BATCH], total;
        fe32_batch_prefix(prefix_products, total, Zs, n);
        fe32_group_invert(total, tree);
        fe32_batch_finish(Zs, prefix_products, total, n);

        for (int j = 0; j < n; j++) {
            ge32_encode(pubkey, Xs[j], Ys[j], Zs[j]);
            if (doppler_count(pubkey) >= required_segments) {
                if (atomic_cmpxchg(done, 0, 1) == 0) {
                    walk_chain(batch_start, j);
                    for (int i = 0; i < 32; i++) out[i] = batch_start[i];
                }
                break;
            }
        }
        iter += (uint)n;
    }

    counts[idx] = iter;
}
