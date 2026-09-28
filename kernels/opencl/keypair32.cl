/* keypair32.cl — the keypair search kernel for Apple GPUs, and the kernels
   behind gpu-self-test for it.

   vanity_keypair_search32 is vanity_keypair_search (keypair.cl) on the Apple
   path's arithmetic. Each work-item derives a 32-byte seed from
   (host_seed || idx), then walks a chain of ed25519 keypairs: SHA-512(seed)
   -> clamp -> signed comb scalarmult (ge32.cl) -> batch compress -> base58
   match against the pattern table, with the same single-pattern and
   any-pattern matchers. The next seed is the high half of the SHA-512 output.
   Before base58, a key's top 64 bits are checked against the prefix ranges
   the host derives from the patterns, which rejects nearly every key without
   computing x or the base58 digits.

   Keys are processed in batches of KP32_BATCH, and one field inversion covers
   the batches of the whole work-group: each work-item multiplies its batch's
   Z values together and fe32_group_invert inverts all the products at once,
   so the group runs its batches in lockstep. A batch keeps only its first
   seed; a match re-walks the chain to its own seed, which is rare enough to
   cost nothing and saves 32 bytes of per-work-item memory per batch slot. */

/* The number of patterns in the table. */
static uint pattern_count(__constant const uchar *patterns) {
    return (uint)patterns[0] | ((uint)patterns[1] << 8)
         | ((uint)patterns[2] << 16) | ((uint)patterns[3] << 24);
}

/* Whether a key whose big-endian view starts with words w0, w1 can match a
   pattern: its top 64 bits lie in one of the host's sorted, disjoint prefix
   ranges (build_prefix_ranges in host.c). With no ranges every key can. */
static bool in_prefix_ranges(uint w0, uint w1, __global const ulong *ranges, uint count) {
    if (count == 0) return true;
    ulong v = ((ulong)w0 << 32) | w1;
    uint lo = 0, hi = count;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (ranges[2 * mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    return lo > 0 && v <= ranges[2 * (lo - 1) + 1];
}

/* Whether an encoded key, as the eight big-endian words of its 32 bytes,
   matches the table the way vanity_keypair_search decides: the
   single-pattern matcher for at most one pattern (with empty strings for
   none), else the any-pattern matcher, which honours the active mask. */
static bool matches_patterns(const uint words[8], __constant const uchar *patterns,
                             __constant const uchar *match_lut, uint n_pat) {
    if (n_pat > 1) return fd_base58_check_match_any_32_words(words, patterns, match_lut);
    uint prefix_len = n_pat ? (uint)patterns[VANITY_PT_PLEN] : 0;
    uint suffix_len = n_pat ? (uint)patterns[VANITY_PT_SLEN] : 0;
    return fd_base58_check_match_32_words(words, patterns + VANITY_PT_PREF, prefix_len,
                                          patterns + VANITY_PT_SUF, suffix_len, match_lut);
}

__kernel void vanity_keypair_search32(
    __global const uchar *host_seed,     /* 32 bytes */
    __constant const uchar *match_lut,   /* 58 */
    __constant const uchar *patterns,
    __global uchar *out,                 /* 32 bytes: matched seed */
    __global volatile int *done,
    __global uint *counts,
    __global const ge32_precomp *comb,   /* COMB32_TABLE_LEN entries */
    uint max_iters,
    __global const ulong *ranges,        /* range_count [start, end] pairs */
    uint range_count,
    __local uint *tree)                  /* 2 * group size fe32, for fe32_group_invert */
{
    ulong idx = get_global_id(0);
    uint n_pat = pattern_count(patterns);

    uchar seed[32];
    uchar batch_start[32];
    uchar privatek[64];
    fe32 Xs[KP32_BATCH];
    fe32 Ys[KP32_BATCH];
    fe32 Zs[KP32_BATCH];
    ge32_p2 A;

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
            fe32 y;
            ge32_affine_y(y, Ys[j], Zs[j]);
            if (!in_prefix_ranges(bswap32(y[0]), bswap32(y[1]), ranges, range_count)) continue;
            ge32_encode_sign(y, Xs[j], Zs[j]);
            uint pubkey_words[8];
            for (int k = 0; k < 8; k++) pubkey_words[k] = bswap32(y[k]);

            if (matches_patterns(pubkey_words, patterns, match_lut, n_pat)) {
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

/* Self-test: the compressed public key of each of `count` seeds, one
   work-item per seed, derived as the search derives them but with one
   inversion per key rather than the batch. */
__kernel void pubkeys_from_seeds32(__global const uchar *seeds,
                                   __global uchar *pubkeys,
                                   __global const ge32_precomp *comb,
                                   uint count)
{
    uint idx = get_global_id(0);
    if (idx >= count) return;

    uchar privatek[64];
    uchar seed[32];
    for (int i = 0; i < 32; i++) seed[i] = seeds[idx * 32 + i];
    sha512_32(privatek, seed);
    clamp_scalar(privatek);

    ge32_p2 A;
    ge32_scalarmult_base_comb(&A, privatek, comb);
    fe32 zinv;
    fe32_invert(zinv, A.Z);
    uchar pubkey[32];
    ge32_encode(pubkey, A.X, A.Y, zinv);
    for (int i = 0; i < 32; i++) pubkeys[idx * 32 + i] = pubkey[i];
}

/* Self-test: whether each of `count` arbitrary 32-byte keys passes the
   search's matcher, pre-filter and then base58. */
__kernel void match_keys32(__global const uchar *keys,
                           __global uchar *flags,
                           __constant const uchar *match_lut,
                           __constant const uchar *patterns,
                           __global const ulong *ranges, uint range_count,
                           uint count)
{
    uint idx = get_global_id(0);
    if (idx >= count) return;
    uint words[8];
    for (int k = 0; k < 8; k++) {
        __global const uchar *b = keys + idx * 32 + 4 * k;
        words[k] = ((uint)b[0] << 24) | ((uint)b[1] << 16) | ((uint)b[2] << 8) | (uint)b[3];
    }
    flags[idx] = in_prefix_ranges(words[0], words[1], ranges, range_count)
              && matches_patterns(words, patterns, match_lut, pattern_count(patterns));
}
