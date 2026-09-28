/* preamble.cl — shared type aliases and struct definitions for the OpenCL
   port of the CUDA vanity kernels.

   OpenCL C has no <stdint.h>, so we map the fixed-width names the ported
   ed25519 / base58 / sha code uses onto OpenCL's guaranteed-width scalar
   types: in full-profile OpenCL `int` is 32-bit and `long` is 64-bit. */

typedef uchar  BYTE;
typedef uint   WORD;

typedef char   int8_t;
typedef uchar  uint8_t;
typedef int    int32_t;
typedef uint   uint32_t;
typedef long   int64_t;
typedef ulong  uint64_t;

/* ed25519 field element: ten signed 25.5-bit limbs (see fe.cl). */
typedef int32_t fe[10];

typedef struct { fe X; fe Y; fe Z; }        ge_p2;
typedef struct { fe X; fe Y; fe Z; fe T; }  ge_p3;
typedef struct { fe X; fe Y; fe Z; fe T; }  ge_p1p1;
typedef struct { fe yplusx; fe yminusx; fe xy2d; } ge_precomp;
/* Extended Niels for radix-32 comb (matches CPU SIMD fixed-base table). */
typedef struct { fe yplusx; fe yminusx; fe z; fe t2d; } ge_niels;

/* Radix-32 comb: 52 windows × 16 positive multiples. */
#define COMB_W        5
#define COMB_WINDOWS  52
#define COMB_POS      16
#define COMB_TABLE_LEN (COMB_WINDOWS * COMB_POS)

/* Used only by the (unused-on-this-path) fe_frombytes; kept for fidelity
   with the upstream ref10 source. Private address space is sufficient. */
static uint64_t load_3(const uchar *in) {
    uint64_t result;
    result  =  (uint64_t) in[0];
    result |= ((uint64_t) in[1]) << 8;
    result |= ((uint64_t) in[2]) << 16;
    return result;
}

static uint64_t load_4(const uchar *in) {
    uint64_t result;
    result  =  (uint64_t) in[0];
    result |= ((uint64_t) in[1]) << 8;
    result |= ((uint64_t) in[2]) << 16;
    result |= ((uint64_t) in[3]) << 24;
    return result;
}

/* Apple GPU keypair path (fe32.cl, ge32.cl, keypair32.cl): a field element
   as eight little-endian 32-bit limbs, and the point forms built from it. */
typedef uint fe32[8];
typedef struct { fe32 X; fe32 Y; fe32 Z; }          ge32_p2;
typedef struct { fe32 X; fe32 Y; fe32 Z; fe32 T; }  ge32_p3;
typedef struct { fe32 X; fe32 Y; fe32 Z; fe32 T; }  ge32_p1p1;
/* 128 bytes: three field elements and 32 bytes of padding, so each comb entry
   is one cache line on Apple GPUs and the two point forms share a stride. */
typedef struct { fe32 yplusx; fe32 yminusx; fe32 xy2d; uint pad[8]; } ge32_precomp;
/* Affine point with its x*y product: comb window 0, which loads straight
   into the accumulator with no multiplication. */
typedef struct { fe32 x; fe32 y; fe32 xy; uint pad[8]; } ge32_affine;

/* The Apple path's signed fixed-base comb: COMB32_WINDOWS windows of COMB32_W
   bits, each holding the COMB32_POS affine multiples 1..2^(COMB32_W-1) of
   2^(COMB32_W*window)*B. The host passes -DCOMB32_W. COMB32_WINDOWS*COMB32_W
   >= 256 keeps the top digit plus its carry within COMB32_POS for a clamped
   scalar, which is below 2^255. */
#ifndef COMB32_W
#define COMB32_W 13
#endif
#define COMB32_WINDOWS   ((256 + COMB32_W - 1) / COMB32_W)
#define COMB32_POS       (1 << (COMB32_W - 1))
#define COMB32_TABLE_LEN (COMB32_WINDOWS * COMB32_POS)

/* Keys per Montgomery batch in the Apple search kernel. */
#ifndef KP32_BATCH
#define KP32_BATCH 16
#endif
