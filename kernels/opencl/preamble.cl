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
