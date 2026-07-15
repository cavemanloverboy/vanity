/* vanity.cl — OpenCL port of the grind kernel from kernels/vanity.cu.

   Each work-item repeatedly hashes SHA-256(base || seed16 || owner), glyph-
   translates the running digest into the next 16-byte seed, and tests the
   resulting pubkey against the target prefix/suffix in base58 word form.

   Differences from the CUDA kernel, all driven by OpenCL 1.2 limits:
     - No clock64(): each launch runs a host-provided `max_iters` cap; the
       host loop relaunches and adapts the cap toward a target wall time.
     - No program-scope mutable globals: `done` and per-work-item `counts`
       are buffers; the host sums counts (avoids unsupported 64-bit atomics).
     - The loop-invariant SHA schedules / LUTs the CUDA build kept in
       __constant symbols are passed as __constant kernel args instead. */

/* Block-0 transform that skips rounds 0..7 by loading the precomputed
   working state (state_r7, fixed by `base`) and starting at round 8. */
static void vanity_sha256_block0_skip8(WORD state_out[8], WORD W[64],
                                       __constant const WORD *state_r7) {
    WORD a = state_r7[0], b = state_r7[1], c = state_r7[2], d = state_r7[3];
    WORD e = state_r7[4], f = state_r7[5], g = state_r7[6], h = state_r7[7];

    VANITY_SHA_EXPAND48(W);

    VANITY_SHA_R(0xD807AA98U, W[ 8]); VANITY_SHA_R(0x12835B01U, W[ 9]);
    VANITY_SHA_R(0x243185BEU, W[10]); VANITY_SHA_R(0x550C7DC3U, W[11]);
    VANITY_SHA_R(0x72BE5D74U, W[12]); VANITY_SHA_R(0x80DEB1FEU, W[13]);
    VANITY_SHA_R(0x9BDC06A7U, W[14]); VANITY_SHA_R(0xC19BF174U, W[15]);
    VANITY_SHA_R(0xE49B69C1U, W[16]); VANITY_SHA_R(0xEFBE4786U, W[17]);
    VANITY_SHA_R(0x0FC19DC6U, W[18]); VANITY_SHA_R(0x240CA1CCU, W[19]);
    VANITY_SHA_R(0x2DE92C6FU, W[20]); VANITY_SHA_R(0x4A7484AAU, W[21]);
    VANITY_SHA_R(0x5CB0A9DCU, W[22]); VANITY_SHA_R(0x76F988DAU, W[23]);
    VANITY_SHA_R(0x983E5152U, W[24]); VANITY_SHA_R(0xA831C66DU, W[25]);
    VANITY_SHA_R(0xB00327C8U, W[26]); VANITY_SHA_R(0xBF597FC7U, W[27]);
    VANITY_SHA_R(0xC6E00BF3U, W[28]); VANITY_SHA_R(0xD5A79147U, W[29]);
    VANITY_SHA_R(0x06CA6351U, W[30]); VANITY_SHA_R(0x14292967U, W[31]);
    VANITY_SHA_R(0x27B70A85U, W[32]); VANITY_SHA_R(0x2E1B2138U, W[33]);
    VANITY_SHA_R(0x4D2C6DFCU, W[34]); VANITY_SHA_R(0x53380D13U, W[35]);
    VANITY_SHA_R(0x650A7354U, W[36]); VANITY_SHA_R(0x766A0ABBU, W[37]);
    VANITY_SHA_R(0x81C2C92EU, W[38]); VANITY_SHA_R(0x92722C85U, W[39]);
    VANITY_SHA_R(0xA2BFE8A1U, W[40]); VANITY_SHA_R(0xA81A664BU, W[41]);
    VANITY_SHA_R(0xC24B8B70U, W[42]); VANITY_SHA_R(0xC76C51A3U, W[43]);
    VANITY_SHA_R(0xD192E819U, W[44]); VANITY_SHA_R(0xD6990624U, W[45]);
    VANITY_SHA_R(0xF40E3585U, W[46]); VANITY_SHA_R(0x106AA070U, W[47]);
    VANITY_SHA_R(0x19A4C116U, W[48]); VANITY_SHA_R(0x1E376C08U, W[49]);
    VANITY_SHA_R(0x2748774CU, W[50]); VANITY_SHA_R(0x34B0BCB5U, W[51]);
    VANITY_SHA_R(0x391C0CB3U, W[52]); VANITY_SHA_R(0x4ED8AA4AU, W[53]);
    VANITY_SHA_R(0x5B9CCA4FU, W[54]); VANITY_SHA_R(0x682E6FF3U, W[55]);
    VANITY_SHA_R(0x748F82EEU, W[56]); VANITY_SHA_R(0x78A5636FU, W[57]);
    VANITY_SHA_R(0x84C87814U, W[58]); VANITY_SHA_R(0x8CC70208U, W[59]);
    VANITY_SHA_R(0x90BEFFFAU, W[60]); VANITY_SHA_R(0xA4506CEBU, W[61]);
    VANITY_SHA_R(0xBEF9A3F7U, W[62]); VANITY_SHA_R(0xC67178F2U, W[63]);

    state_out[0] = 0x6a09e667U + a; state_out[1] = 0xbb67ae85U + b;
    state_out[2] = 0x3c6ef372U + c; state_out[3] = 0xa54ff53aU + d;
    state_out[4] = 0x510e527fU + e; state_out[5] = 0x9b05688cU + f;
    state_out[6] = 0x1f83d9abU + g; state_out[7] = 0x5be0cd19U + h;
}

/* SHA-256 of base[32] || seed16[16] || owner[32] in word form. */
static void vanity_pubkey_sha256_words(const WORD seed_words[4], WORD state_out[8],
                                       __constant const WORD *W0_fixed,
                                       __constant const WORD *state_r7,
                                       __constant const WORD *W1) {
    WORD W0[64];
    for (int i = 0; i < 16; ++i) W0[i] = W0_fixed[i];
    W0[ 8] = seed_words[0];
    W0[ 9] = seed_words[1];
    W0[10] = seed_words[2];
    W0[11] = seed_words[3];

    vanity_sha256_block0_skip8(state_out, W0, state_r7);
    cuda_sha256_transform_w(state_out, W1);
}

__kernel void vanity_search(
    __global const uchar *seed,          /* 32 bytes */
    __constant const WORD *W0_fixed,     /* 16 */
    __constant const WORD *state_r7,     /* 8  */
    __constant const WORD *W1,           /* 64 */
    __constant const uchar *glyph,       /* 256 */
    __constant const uchar *match_lut,   /* 58 */
    __global const uchar *prefixes, uint prefix_count,
    __global const uchar *suffixes, uint suffix_count,
    __global uchar *out,                 /* 16 bytes: matched seed16 */
    __global volatile int *done,
    __global uint *counts,
    uint max_iters)
{
    ulong idx = get_global_id(0);

    /* Bootstrap per-work-item digest: byte-form seed with idx mixed into
       each 64-bit lane, then packed once into 8 big-endian words. */
    WORD digest_words[8];
    {
        uchar ls[32];
        for (int i = 0; i < 32; ++i) ls[i] = seed[i];
        for (int lane = 0; lane < 4; ++lane) {
            ulong v = 0;
            for (int b = 0; b < 8; ++b) v |= ((ulong)ls[lane*8 + b]) << (8*b);
            v += idx;
            for (int b = 0; b < 8; ++b) ls[lane*8 + b] = (uchar)(v >> (8*b));
        }
        for (int i = 0; i < 8; ++i)
            digest_words[i] = ((WORD)ls[4*i] << 24) | ((WORD)ls[4*i+1] << 16)
                            | ((WORD)ls[4*i+2] << 8) | ((WORD)ls[4*i+3]);
    }

    uint iter = 0;
    for (; iter < max_iters; iter++) {
        if ((iter & 0x3FFF) == 0 && atomic_max(done, 0) == 1) break;

        /* Glyph-translate the first 16 digest bytes into the next seed16,
           repacked as 4 big-endian words (== W0[8..11]). */
        WORD seed_words[4];
        for (int k = 0; k < 4; ++k) {
            WORD s = digest_words[k];
            seed_words[k] = ((WORD)glyph[(s >> 24) & 0xFFU] << 24)
                          | ((WORD)glyph[(s >> 16) & 0xFFU] << 16)
                          | ((WORD)glyph[(s >>  8) & 0xFFU] <<  8)
                          | ((WORD)glyph[(s      ) & 0xFFU]      );
        }

        vanity_pubkey_sha256_words(seed_words, digest_words, W0_fixed, state_r7, W1);

        if (fd_base58_check_match_32_words(digest_words, prefixes, prefix_count,
                                           suffixes, suffix_count, match_lut)) {
            if (atomic_cmpxchg(done, 0, 1) == 0) {
                for (int k = 0; k < 4; ++k) {
                    WORD w = seed_words[k];
                    out[4*k    ] = (uchar)(w >> 24);
                    out[4*k + 1] = (uchar)(w >> 16);
                    out[4*k + 2] = (uchar)(w >>  8);
                    out[4*k + 3] = (uchar)(w      );
                }
            }
            iter++;          /* count the matching attempt */
            break;
        }
    }

    counts[idx] = iter;
}
