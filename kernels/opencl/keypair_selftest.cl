/* keypair_selftest.cl — kernels behind `vanity gpu-self-test`, compiled into
   the keypair program next to keypair.cl. pubkeys_from_seeds derives public
   keys the way vanity_keypair_search does, and match_keys runs its matcher,
   so the host can compare both against reference implementations. */

/* The compressed public key of each of `count` seeds, one work-item per
   seed. It uses one inversion per key rather than the search's batch. */
__kernel void pubkeys_from_seeds(__global const uchar *seeds,
                                 __global uchar *pubkeys,
                                 __global const ge_niels *comb,
                                 uint count)
{
    uint idx = get_global_id(0);
    if (idx >= count) return;

    uchar privatek[64];
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
    for (int i = 0; i < 32; i++) md.buf[i] = seeds[idx * 32 + i];
    md.curlen = 32;
    sha512_final(&md, privatek);

    privatek[0]  &= 248;
    privatek[31] &= 63;
    privatek[31] |= 64;

    ge_p3 A;
    ge_scalarmult_base_comb(&A, privatek, comb);
    fe zinv;
    fe_invert(zinv, A.Z);
    uchar pubkey[32];
    ge_p3_tobytes_inv(pubkey, A.X, A.Y, zinv);
    for (int i = 0; i < 32; i++) pubkeys[idx * 32 + i] = pubkey[i];
}

/* Whether each of `count` arbitrary 32-byte keys passes the search kernel's
   matcher, chosen as it chooses: the single-pattern check for at most one
   pattern, else the any-pattern check that honours the active mask. */
__kernel void match_keys(__global const uchar *keys,
                         __global uchar *flags,
                         __constant const uchar *match_lut,
                         __constant const uchar *patterns,
                         uint count)
{
    uint idx = get_global_id(0);
    if (idx >= count) return;

    uint words[8];
    for (int k = 0; k < 8; k++) {
        __global const uchar *b = keys + idx * 32 + 4 * k;
        words[k] = ((uint)b[0] << 24) | ((uint)b[1] << 16) | ((uint)b[2] << 8) | (uint)b[3];
    }

    uint n_pat = (uint)patterns[0] | ((uint)patterns[1] << 8)
               | ((uint)patterns[2] << 16) | ((uint)patterns[3] << 24);
    uint prefix_len = 0, suffix_len = 0;
    if (n_pat == 1) {
        prefix_len = (uint)patterns[VANITY_PT_PLEN];
        suffix_len = (uint)patterns[VANITY_PT_SLEN];
    }
    flags[idx] = (n_pat <= 1)
        ? fd_base58_check_match_32_words(words, patterns + VANITY_PT_PREF, prefix_len,
                                         patterns + VANITY_PT_SUF, suffix_len, match_lut)
        : fd_base58_check_match_any_32_words(words, patterns, match_lut);
}
