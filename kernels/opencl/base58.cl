/* base58.cl — OpenCL port of kernels/base58.{h,cu}.

   Two entry points, mirroring the CUDA build:
     - fd_base58_encode_32: full bytes->string encode (kept for completeness).
     - fd_base58_check_match_32_words: fused encode + prefix/suffix match in
       word form, with early rejection (grind + keypair kernels).
   Both descend from Firedancer's fd_base58 (cavemanloverboy port). */

__constant uchar base58_chars[]    = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
/* ci = case insensitive: map all uppercase to lowercase except for L. */
__constant uchar base58_chars_ci[] = "123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";

/* enc_table_32[j][k]: the unique values < 58^5 such that
     2^(32*(7-j)) = sum_k table[j][k]*58^(5*(7-k)). */
__constant uint enc_table_32[8][8] = {
    {513735U, 77223048U, 437087610U, 300156666U, 605448490U, 214625350U, 141436834U, 379377856U},
    {0U, 78508U, 646269101U, 118408823U, 91512303U, 209184527U, 413102373U, 153715680U},
    {0U, 0U, 11997U, 486083817U, 3737691U, 294005210U, 247894721U, 289024608U},
    {0U, 0U, 0U, 1833U, 324463681U, 385795061U, 551597588U, 21339008U},
    {0U, 0U, 0U, 0U, 280U, 127692781U, 389432875U, 357132832U},
    {0U, 0U, 0U, 0U, 0U, 42U, 537767569U, 410450016U},
    {0U, 0U, 0U, 0U, 0U, 0U, 6U, 356826688U},
    {0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U}};

/* Full base58 encode of a 32-byte big-endian integer into `out` (ASCII,
   NUL-terminated). Returns the encoded length. */
static ulong fd_base58_encode_32(const uchar *bytes, uchar *out, bool case_insensitive) {
    ulong in_leading_0s = 0UL;
    for (; in_leading_0s < 32UL; in_leading_0s++)
        if (bytes[in_leading_0s]) break;

    /* X = sum_i binary[i] * 2^(32*(7-i)), big-endian 32-bit limbs. */
    uint binary[8];
    for (ulong i = 0UL; i < 8UL; i++)
        binary[i] = ((uint)bytes[4*i    ] << 24) | ((uint)bytes[4*i + 1] << 16)
                  | ((uint)bytes[4*i + 2] <<  8) | ((uint)bytes[4*i + 3]      );

    ulong R1div = 656356768UL; /* = 58^5 */

    ulong intermediate[9] = {0};
    for (ulong i = 0UL; i < 8UL; i++)
        for (ulong j = 0UL; j < 8UL; j++)
            intermediate[j + 1UL] += (ulong)binary[i] * (ulong)enc_table_32[i][j];

    for (ulong i = 8UL; i > 0UL; i--) {
        intermediate[i - 1UL] += (intermediate[i] / R1div);
        intermediate[i] %= R1div;
    }

    uchar raw_base58[45];
    for (ulong i = 0UL; i < 9UL; i++) {
        uint v = (uint)intermediate[i];
        raw_base58[5UL * i + 4UL] = (uchar)((v / 1U)        % 58U);
        raw_base58[5UL * i + 3UL] = (uchar)((v / 58U)       % 58U);
        raw_base58[5UL * i + 2UL] = (uchar)((v / 3364U)     % 58U);
        raw_base58[5UL * i + 1UL] = (uchar)((v / 195112U)   % 58U);
        raw_base58[5UL * i + 0UL] = (uchar)( v / 11316496U);
    }

    ulong raw_leading_0s = 0UL;
    for (; raw_leading_0s < 45UL; raw_leading_0s++)
        if (raw_base58[raw_leading_0s]) break;

    __constant uchar *b58_chars = case_insensitive ? base58_chars_ci : base58_chars;

    ulong skip = raw_leading_0s - in_leading_0s;
    ulong encoded_length = 45UL - skip;

    for (ulong i = 0UL; i < encoded_length; i++)
        out[i] = b58_chars[raw_base58[skip + i]];

    out[encoded_length] = '\0';
    return encoded_length;
}

#define VANITY_MATCH_PLAN_WORDS (1UL + 44UL * 58UL + 44UL)

/* Word-form fused base58 encode + prefix/suffix match with early rejection.
   `state[8]` is the SHA-256 digest as 8 native words (MSB-first); the two
   byte-swaps cancel so binary[i] == state[i]. Prefix and suffix plans contain
   candidate bitmasks precomputed on the host; match_lut folds raw values into
   the same canonical space. */
static bool fd_base58_check_match_32_words(const uint state[8],
                                           __global const uchar *prefixes, ulong prefix_count,
                                           __global const uchar *suffixes, ulong suffix_count,
                                           __constant const uchar *match_lut) {
    (void)prefix_count;
    (void)suffix_count;
    /* Count leading zero bytes of the big-endian byte view of state[]. */
    ulong in_leading_0s = 0UL;
    for (int i = 0; i < 8; ++i) {
        ulong lz_bytes = (ulong)(clz(state[i]) >> 3);
        in_leading_0s += lz_bytes;
        if (lz_bytes != 4UL) break;
    }

    ulong R1div = 656356768UL; /* = 58^5 */

    ulong intermediate[9] = {0};
    for (ulong i = 0UL; i < 8UL; i++)
        for (ulong j = 0UL; j < 8UL; j++)
            intermediate[j + 1UL] += (ulong)state[i] * (ulong)enc_table_32[i][j];

    for (ulong i = 8UL; i > 0UL; i--) {
        intermediate[i - 1UL] += (intermediate[i] / R1div);
        intermediate[i] %= R1div;
    }

    uchar raw_base58[45];
    ulong limbs_done = 0UL;
    #define VANITY_BS58_EMIT_LIMB(L) do { \
        uint v = (uint)intermediate[(L)]; \
        raw_base58[5UL*(L) + 4UL] = (uchar)((v / 1U)        % 58U); \
        raw_base58[5UL*(L) + 3UL] = (uchar)((v / 58U)       % 58U); \
        raw_base58[5UL*(L) + 2UL] = (uchar)((v / 3364U)     % 58U); \
        raw_base58[5UL*(L) + 1UL] = (uchar)((v / 195112U)   % 58U); \
        raw_base58[5UL*(L) + 0UL] = (uchar)( v / 11316496U); \
    } while (0)
    #define VANITY_BS58_ENSURE_LIMB(L) do { \
        while (limbs_done <= (L)) { VANITY_BS58_EMIT_LIMB(limbs_done); limbs_done++; } \
    } while (0)

    VANITY_BS58_ENSURE_LIMB(0UL);

    ulong raw_leading_0s = 0UL;
    while (raw_leading_0s < 45UL) {
        if (raw_leading_0s / 5UL >= limbs_done)
            VANITY_BS58_ENSURE_LIMB(raw_leading_0s / 5UL);
        if (raw_base58[raw_leading_0s]) break;
        raw_leading_0s++;
    }

    ulong skip = raw_leading_0s - in_leading_0s;
    ulong encoded_length = 45UL - skip;

    __global const uint *prefix_plan = (__global const uint *)prefixes;
    __global const uint *suffix_plan = (__global const uint *)suffixes;
    uint candidates = prefix_plan[0];
    if (candidates != 0U) {
        for (uint position = 0U;
             position < 44U && position < (uint)encoded_length;
             position++) {
            uint rb_idx = (uint)skip + position;
            VANITY_BS58_ENSURE_LIMB((ulong)rb_idx / 5UL);
            candidates &= prefix_plan[
                1U + position * 58U
                + (uint)match_lut[raw_base58[rb_idx]]];
            if (candidates & prefix_plan[1U + 44U * 58U + position])
                goto prefix_matched;
            if (candidates == 0U) return false;
        }
        return false;
    }
prefix_matched:
    candidates = suffix_plan[0];
    if (candidates != 0U) {
        for (uint position = 0U;
             position < 44U && position < (uint)encoded_length;
             position++) {
            uint rb_idx = (uint)(skip + encoded_length - 1UL) - position;
            VANITY_BS58_ENSURE_LIMB((ulong)rb_idx / 5UL);
            candidates &= suffix_plan[
                1U + position * 58U
                + (uint)match_lut[raw_base58[rb_idx]]];
            if (candidates & suffix_plan[1U + 44U * 58U + position])
                return true;
            if (candidates == 0U) return false;
        }
        return false;
    }

    return true;

    #undef VANITY_BS58_EMIT_LIMB
    #undef VANITY_BS58_ENSURE_LIMB
}
