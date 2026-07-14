#ifndef BS58_H
#define BS58_H

#include <stdint.h>
#include <string.h>

/* Defined in base58.cu. */
extern __device__ uint8_t const base58_chars[];
extern __device__ uint8_t const base58_chars_ci[];
extern __device__ uint const enc_table_32[8][8];
extern __constant__ uint8_t d_match_lut[58];

#define VANITY_MATCH_PLAN_WORDS (1U + 44U * 58U + 44U)

__device__ ulong fd_base58_encode_32(uint8_t *bytes, uint8_t *out, bool case_insensitive);

/* Word-form fused base58 encode + prefix/suffix OR match with early
   rejection. Caller passes the SHA-256 digest as 8 native uint words
   (state[0..7], MSB-first interpretation), avoiding the byte-emit /
   byte-load / byte-swap roundtrip that the byte-form caller forced
   us into previously. The two byte-swaps cancel exactly:
   binary[i] == state[i].

   Prefix and suffix plans contain candidate bitmasks for each character and
   position, precomputed on the host. d_match_lut folds raw_base58 values into
   the same canonical space, so neither a b58_chars[] lookup nor a
   case_insensitive branch is in the path.

   Inlined into the call site so __launch_bounds__ register caps on the
   parent kernel apply transitively. */
static __device__ __forceinline__ bool fd_base58_check_match_32_words(const uint state[8],
                                                                      const uint8_t *prefixes,
                                                                      ulong prefix_count,
                                                                      const uint8_t *suffixes,
                                                                      ulong suffix_count)
{
    (void)prefix_count;
    (void)suffix_count;
    /* Count leading zero bytes of the big-endian byte view of state[].
       Each word contributes 0..4 leading zero bytes; advance through
       words that are entirely zero. __clz lowers to a single SFU op. */
    ulong in_leading_0s = 0UL;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        ulong lz_bytes = (ulong)((unsigned)__clz((int)state[i]) >> 3);
        in_leading_0s += lz_bytes;
        if (lz_bytes != 4UL) break;
    }

    ulong R1div = 656356768UL; /* = 58^5 */

    ulong intermediate[9] = {0};
    for (ulong i = 0UL; i < 8UL; i++)
        for (ulong j = 0UL; j < 8UL; j++)
            intermediate[j + 1UL] += (ulong)state[i] * (ulong)enc_table_32[i][j];

    for (ulong i = 8UL; i > 0UL; i--)
    {
        intermediate[i - 1UL] += (intermediate[i] / R1div);
        intermediate[i] %= R1div;
    }

    uint8_t raw_base58[45];
    #define VANITY_BS58_EMIT_LIMB(L) do { \
        uint v = (uint)intermediate[(L)]; \
        raw_base58[5UL*(L) + 4UL] = (uint8_t)((v / 1U) % 58U); \
        raw_base58[5UL*(L) + 3UL] = (uint8_t)((v / 58U) % 58U); \
        raw_base58[5UL*(L) + 2UL] = (uint8_t)((v / 3364U) % 58U); \
        raw_base58[5UL*(L) + 1UL] = (uint8_t)((v / 195112U) % 58U); \
        raw_base58[5UL*(L) + 0UL] = (uint8_t)(v / 11316496U); \
    } while (0)
#pragma unroll
    for (int limb = 0; limb < 9; ++limb)
        VANITY_BS58_EMIT_LIMB(limb);

    ulong raw_leading_0s = 0UL;
    while (raw_leading_0s < 45UL)
    {
        if (raw_base58[raw_leading_0s])
            break;
        raw_leading_0s++;
    }

    ulong skip = raw_leading_0s - in_leading_0s;
    ulong encoded_length = 45UL - skip;

    const uint *prefix_plan = (const uint *)prefixes;
    uint candidates = prefix_plan[0];
    if (candidates != 0U)
    {
        for (uint position = 0U;
             position < 44U && position < (uint)encoded_length;
             position++)
        {
            uint rb_idx = (uint)skip + position;
            candidates &= prefix_plan[
                1U + position * 58U
                + (uint)d_match_lut[raw_base58[rb_idx]]];
            if (candidates & prefix_plan[1U + 44U * 58U + position])
                goto prefix_matched;
            if (candidates == 0U) return false;
        }
        return false;
    }
prefix_matched:

    const uint *suffix_plan = (const uint *)suffixes;
    candidates = suffix_plan[0];
    if (candidates != 0U)
    {
        for (uint position = 0U;
             position < 44U && position < (uint)encoded_length;
             position++)
        {
            uint rb_idx = (uint)(skip + encoded_length - 1UL) - position;
            candidates &= suffix_plan[
                1U + position * 58U
                + (uint)d_match_lut[raw_base58[rb_idx]]];
            if (candidates & suffix_plan[1U + 44U * 58U + position])
                return true;
            if (candidates == 0U) return false;
        }
        return false;
    }

    return true;

    #undef VANITY_BS58_EMIT_LIMB
}

#endif
