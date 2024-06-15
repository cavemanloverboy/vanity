#include "base58.h"

__device__ uint8_t const base58_chars[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
// ci = case insensitive (map all uppercase to lowercase except for L)
__device__ uint8_t const base58_chars_ci[] = "123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";

#define BASE58_INVALID_CHAR ((uint8_t)255)
#define BASE58_INVERSE_TABLE_OFFSET ((uint8_t)'1')
#define BASE58_INVERSE_TABLE_SENTINEL ((uint8_t)(1UL + (uint8_t)('z') - BASE58_INVERSE_TABLE_OFFSET))
#define BAD BASE58_INVALID_CHAR

/* base58_inverse maps (character value - '1') to [0, 58).  Invalid
   base58 characters map to BASE58_INVALID_CHAR.  The character after
   what 'z' would map to also maps to BASE58_INVALID_CHAR to facilitate
   branchless lookups.  Don't make it static so that it can be used from
   tests. */

#define BAD BASE58_INVALID_CHAR

// uint8_t const base58_inverse[] = {
//     (uint8_t)0, (uint8_t)1, (uint8_t)2, (uint8_t)3, (uint8_t)4, (uint8_t)5, (uint8_t)6, (uint8_t)7, (uint8_t)8, (uint8_t)BAD,
//     (uint8_t)BAD, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)9, (uint8_t)10, (uint8_t)11, (uint8_t)12,
//     (uint8_t)13, (uint8_t)14, (uint8_t)15, (uint8_t)16, (uint8_t)BAD, (uint8_t)17, (uint8_t)18, (uint8_t)19, (uint8_t)20, (uint8_t)21,
//     (uint8_t)BAD, (uint8_t)22, (uint8_t)23, (uint8_t)24, (uint8_t)25, (uint8_t)26, (uint8_t)27, (uint8_t)28, (uint8_t)29, (uint8_t)30,
//     (uint8_t)31, (uint8_t)32, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)BAD, (uint8_t)33, (uint8_t)34,
//     (uint8_t)35, (uint8_t)36, (uint8_t)37, (uint8_t)38, (uint8_t)39, (uint8_t)40, (uint8_t)41, (uint8_t)42, (uint8_t)43, (uint8_t)BAD,
//     (uint8_t)44, (uint8_t)45, (uint8_t)46, (uint8_t)47, (uint8_t)48, (uint8_t)49, (uint8_t)50, (uint8_t)51, (uint8_t)52, (uint8_t)53,
//     (uint8_t)54, (uint8_t)55, (uint8_t)56, (uint8_t)57, (uint8_t)BAD};

#undef BAD

#define N 32
#define INTERMEDIATE_SZ (9UL) /* Computed by ceil(log_(58^5) (256^32-1)) */
#define BINARY_SZ ((ulong)N / 4UL)

/* Contains the unique values less than 58^5 such that:
     2^(32*(7-j)) = sum_k table[j][k]*58^(5*(7-k))

   The second dimension of this table is actually ceil(log_(58^5)
   (2^(32*(BINARY_SZ-1))), but that's almost always INTERMEDIATE_SZ-1 */

__device__ uint const enc_table_32[BINARY_SZ][INTERMEDIATE_SZ - 1UL] = {
    {513735U, 77223048U, 437087610U, 300156666U, 605448490U, 214625350U, 141436834U, 379377856U},
    {0U, 78508U, 646269101U, 118408823U, 91512303U, 209184527U, 413102373U, 153715680U},
    {0U, 0U, 11997U, 486083817U, 3737691U, 294005210U, 247894721U, 289024608U},
    {0U, 0U, 0U, 1833U, 324463681U, 385795061U, 551597588U, 21339008U},
    {0U, 0U, 0U, 0U, 280U, 127692781U, 389432875U, 357132832U},
    {0U, 0U, 0U, 0U, 0U, 42U, 537767569U, 410450016U},
    {0U, 0U, 0U, 0U, 0U, 0U, 6U, 356826688U},
    {0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U}};

/* Contains the unique values less than 2^32 such that:
     58^(5*(8-j)) = sum_k table[j][k]*2^(32*(7-k)) */

// __device__ uint const dec_table_32[INTERMEDIATE_SZ][BINARY_SZ] = {
//     {1277U, 2650397687U, 3801011509U, 2074386530U, 3248244966U, 687255411U, 2959155456U, 0U},
//     {0U, 8360U, 1184754854U, 3047609191U, 3418394749U, 132556120U, 1199103528U, 0U},
//     {0U, 0U, 54706U, 2996985344U, 1834629191U, 3964963911U, 485140318U, 1073741824U},
//     {0U, 0U, 0U, 357981U, 1476998812U, 3337178590U, 1483338760U, 4194304000U},
//     {0U, 0U, 0U, 0U, 2342503U, 3052466824U, 2595180627U, 17825792U},
//     {0U, 0U, 0U, 0U, 0U, 15328518U, 1933902296U, 4063920128U},
//     {0U, 0U, 0U, 0U, 0U, 0U, 100304420U, 3355157504U},
//     {0U, 0U, 0U, 0U, 0U, 0U, 0U, 656356768U},
//     {0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U}};

////////

/* Declares conversion functions to/from base58 for a specific size of
   binary data.

   To use this template, define:

     N: the length of the binary data (in bytes) to convert.  N must be
         32 or 64 in the current implementation.
     INTERMEDIATE_SZ: ceil(log_(58^5) ( (256^N) - 1)). In an ideal
         world, this could be computed from N, but there's no way the
         preprocessor can do math like that.
     BINARY_SIZE: N/4.  Define it yourself to facilitate declaring the
         required tables.

   INTERMEDIATE_SZ and BINARY_SZ should expand to ulongs while N should
   be an integer literal.

   Expects that enc_table_N, dec_table_N, and FD_BASE58_ENCODED_N_SZ
   exist (substituting the numeric value of N).

   This file is safe for inclusion multiple times. */

#define BYTE_CNT ((ulong)N)
#define SUFFIX(s) FD_EXPAND_THEN_CONCAT3(s, _, N)
#define ENCODED_SZ() FD_EXPAND_THEN_CONCAT3(FD_BASE58_ENCODED_, N, _SZ)
#define RAW58_SZ (INTERMEDIATE_SZ * 5UL)

#if FD_HAS_AVX
#define INTERMEDIATE_SZ_W_PADDING FD_ULONG_ALIGN_UP(INTERMEDIATE_SZ, 4UL)
#else
#define INTERMEDIATE_SZ_W_PADDING INTERMEDIATE_SZ
#endif

__device__ void fd_base58_encode_32(uint8_t *bytes,
                                    uint8_t *out,
                                    bool case_insensitive)
{
    ulong in_leading_0s = 0UL;
    for (; in_leading_0s < BYTE_CNT; in_leading_0s++)
        if (bytes[in_leading_0s])
            break;

    /* X = sum_i bytes[i] * 2^(8*(BYTE_CNT-1-i)) */

    /* Convert N to 32-bit limbs:
       X = sum_i binary[i] * 2^(32*(BINARY_SZ-1-i)) */
    uint binary[BINARY_SZ];
    for (ulong i = 0UL; i < BINARY_SZ; i++)
    {
        uint u32;
        memcpy(&u32, &bytes[i * sizeof(uint)], 4);
        binary[i] = ((u32 & 0x000000FF) << 24) |
                    ((u32 & 0x0000FF00) << 8) |
                    ((u32 & 0x00FF0000) >> 8) |
                    ((u32 & 0xFF000000) >> 24);
    }

    ulong R1div = 656356768UL; /* = 58^5 */

    /* Convert to the intermediate format:
         X = sum_i intermediate[i] * 58^(5*(INTERMEDIATE_SZ-1-i))
       Initially, we don't require intermediate[i] < 58^5, but we do want
       to make sure the sums don't overflow. */

    ulong intermediate[INTERMEDIATE_SZ_W_PADDING] = {0};

    /* The worst case is if binary[7] is (2^32)-1. In that case
       intermediate[8] will be be just over 2^63, which is fine. */

    for (ulong i = 0UL; i < BINARY_SZ; i++)
        for (ulong j = 0UL; j < INTERMEDIATE_SZ - 1UL; j++)
            intermediate[j + 1UL] += (ulong)binary[i] * (ulong)enc_table_32[i][j];

    /* Now we make sure each term is less than 58^5. Again, we have to be
       a bit careful of overflow.

       For N==32, in the worst case, as before, intermediate[8] will be
       just over 2^63 and intermediate[7] will be just over 2^62.6.  In
       the first step, we'll add floor(intermediate[8]/58^5) to
       intermediate[7].  58^5 is pretty big though, so intermediate[7]
       barely budges, and this is still fine.

       For N==64, in the worst case, the biggest entry in intermediate at
       this point is 2^63.87, and in the worst case, we add (2^64-1)/58^5,
       which is still about 2^63.87. */

    for (ulong i = INTERMEDIATE_SZ - 1UL; i > 0UL; i--)
    {
        intermediate[i - 1UL] += (intermediate[i] / R1div);
        intermediate[i] %= R1div;
    }

    /* Convert intermediate form to base 58.  This form of conversion
       exposes tons of ILP, but it's more than the CPU can take advantage
       of.
         X = sum_i raw_base58[i] * 58^(RAW58_SZ-1-i) */

    uint8_t raw_base58[RAW58_SZ];
    for (ulong i = 0UL; i < INTERMEDIATE_SZ; i++)
    {
        /* We know intermediate[ i ] < 58^5 < 2^32 for all i, so casting to
           a uint is safe.  GCC doesn't seem to be able to realize this, so
           when it converts ulong/ulong to a magic multiplication, it
           generates the single-op 64b x 64b -> 128b mul instruction.  This
           hurts the CPU's ability to take advantage of the ILP here. */
        uint v = (uint)intermediate[i];
        raw_base58[5UL * i + 4UL] = (uint8_t)((v / 1U) % 58U);
        raw_base58[5UL * i + 3UL] = (uint8_t)((v / 58U) % 58U);
        raw_base58[5UL * i + 2UL] = (uint8_t)((v / 3364U) % 58U);
        raw_base58[5UL * i + 1UL] = (uint8_t)((v / 195112U) % 58U);
        raw_base58[5UL * i + 0UL] = (uint8_t)(v / 11316496U); /* We know this one is less than 58 */
    }

    /* Finally, actually convert to the string.  We have to ignore all the
       leading zeros in raw_base58 and instead insert in_leading_0s
       leading '1' characters.  We can show that raw_base58 actually has
       at least in_leading_0s, so we'll do this by skipping the first few
       leading zeros in raw_base58. */

    ulong raw_leading_0s = 0UL;
    for (; raw_leading_0s < RAW58_SZ; raw_leading_0s++)
        if (raw_base58[raw_leading_0s])
            break;

    /* It's not immediately obvious that raw_leading_0s >= in_leading_0s,
       but it's true.  In base b, X has floor(log_b X)+1 digits.  That
       means in_leading_0s = N-1-floor(log_256 X) and raw_leading_0s =
       RAW58_SZ-1-floor(log_58 X).  Let X<256^N be given and consider:

       raw_leading_0s - in_leading_0s =
         =  RAW58_SZ-N + floor( log_256 X ) - floor( log_58 X )
         >= RAW58_SZ-N - 1 + ( log_256 X - log_58 X ) .

       log_256 X - log_58 X is monotonically decreasing for X>0, so it
       achieves it minimum at the maximum possible value for X, i.e.
       256^N-1.
         >= RAW58_SZ-N-1 + log_256(256^N-1) - log_58(256^N-1)

       When N==32, RAW58_SZ is 45, so this gives skip >= 0.29
       When N==64, RAW58_SZ is 90, so this gives skip >= 1.59.

       Regardless, raw_leading_0s - in_leading_0s >= 0. */

    const uint8_t *b58_chars;
    if (case_insensitive)
    {
        b58_chars = base58_chars_ci;
    }
    else
    {
        b58_chars = base58_chars;
    }

    ulong skip = raw_leading_0s - in_leading_0s;
    for (ulong i = 0UL; i < RAW58_SZ - skip; i++)
        out[i] = b58_chars[raw_base58[skip + i]];

    // out[RAW58_SZ - skip] = '\0';
}

#undef RAW58_SZ
#undef ENCODED_SZ
#undef SUFFIX

#undef BINARY_SZ
#undef BYTE_CNT
#undef INTERMEDIATE_SZ
#undef N