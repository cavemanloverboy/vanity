#ifndef BS58_H
#define BS58_H

#include <stdint.h>
#include <string.h>

/* Defined in base58.cu. */
extern __device__ uint8_t const base58_chars[];
extern __device__ uint8_t const base58_chars_ci[];
extern __device__ uint const enc_table_32[8][8];
extern __constant__ uint8_t d_match_lut[58];

__device__ ulong fd_base58_encode_32(uint8_t *bytes, uint8_t *out, bool case_insensitive);

/* Fused base58 encode + prefix/suffix match with early rejection.

   `target` and `suffix` are arrays of canonical raw_base58 indices
   (0..57) precomputed on the host via gpu_grind_init. d_match_lut folds
   raw_base58 values into the same canonical space, so neither a
   b58_chars[] lookup nor a case_insensitive branch is in the path.

   Optimization vs fd_base58_encode_32 + matches_target:
     - Only the multiply + carry phase is unconditional.
     - Limbs (raw_base58 digits) are emitted lazily, just enough to cover
       the indices the prefix touches; on the typical first-character
       mismatch (~98% of attempts for short prefixes) we skip the rest of
       the digit conversion AND the full 44-char ASCII output write that
       fd_base58_encode_32 always performs.

   Inlined into the call site so __launch_bounds__ register caps on the
   parent kernel apply transitively. */
static __device__ __forceinline__ bool fd_base58_check_match_32(uint8_t *bytes,
                                                                  const uint8_t *target,
                                                                  ulong target_len,
                                                                  const uint8_t *suffix,
                                                                  ulong suffix_len)
{
    ulong in_leading_0s = 0UL;
    for (; in_leading_0s < 32UL; in_leading_0s++)
        if (bytes[in_leading_0s])
            break;

    uint binary[8];
    for (ulong i = 0UL; i < 8UL; i++)
    {
        uint u32;
        memcpy(&u32, &bytes[i * sizeof(uint)], 4);
        binary[i] = ((u32 & 0x000000FF) << 24) |
                    ((u32 & 0x0000FF00) << 8) |
                    ((u32 & 0x00FF0000) >> 8) |
                    ((u32 & 0xFF000000) >> 24);
    }

    ulong R1div = 656356768UL; /* = 58^5 */

    ulong intermediate[9] = {0};
    for (ulong i = 0UL; i < 8UL; i++)
        for (ulong j = 0UL; j < 8UL; j++)
            intermediate[j + 1UL] += (ulong)binary[i] * (ulong)enc_table_32[i][j];

    for (ulong i = 8UL; i > 0UL; i--)
    {
        intermediate[i - 1UL] += (intermediate[i] / R1div);
        intermediate[i] %= R1div;
    }

    uint8_t raw_base58[45];
    ulong limbs_done = 0UL;
    #define VANITY_BS58_EMIT_LIMB(L) do { \
        uint v = (uint)intermediate[(L)]; \
        raw_base58[5UL*(L) + 4UL] = (uint8_t)((v / 1U) % 58U); \
        raw_base58[5UL*(L) + 3UL] = (uint8_t)((v / 58U) % 58U); \
        raw_base58[5UL*(L) + 2UL] = (uint8_t)((v / 3364U) % 58U); \
        raw_base58[5UL*(L) + 1UL] = (uint8_t)((v / 195112U) % 58U); \
        raw_base58[5UL*(L) + 0UL] = (uint8_t)(v / 11316496U); \
    } while (0)
    #define VANITY_BS58_ENSURE_LIMB(L) do { \
        while (limbs_done <= (L)) { VANITY_BS58_EMIT_LIMB(limbs_done); limbs_done++; } \
    } while (0)

    VANITY_BS58_ENSURE_LIMB(0UL);

    ulong raw_leading_0s = 0UL;
    while (raw_leading_0s < 45UL)
    {
        if (raw_leading_0s / 5UL >= limbs_done)
            VANITY_BS58_ENSURE_LIMB(raw_leading_0s / 5UL);
        if (raw_base58[raw_leading_0s])
            break;
        raw_leading_0s++;
    }

    ulong skip = raw_leading_0s - in_leading_0s;
    ulong encoded_length = 45UL - skip;

    for (ulong i = 0UL; i < target_len; i++)
    {
        ulong rb_idx = skip + i;
        VANITY_BS58_ENSURE_LIMB(rb_idx / 5UL);
        if (d_match_lut[raw_base58[rb_idx]] != target[i])
            return false;
    }

    if (suffix_len > 0UL)
    {
        ulong tail_start = skip + encoded_length - suffix_len;
        ulong last_limb = (skip + encoded_length - 1UL) / 5UL;
        VANITY_BS58_ENSURE_LIMB(last_limb);
        for (ulong i = 0UL; i < suffix_len; i++)
        {
            if (d_match_lut[raw_base58[tail_start + i]] != suffix[i])
                return false;
        }
    }

    return true;

    #undef VANITY_BS58_EMIT_LIMB
    #undef VANITY_BS58_ENSURE_LIMB
}

#endif
