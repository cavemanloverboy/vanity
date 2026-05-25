/* sha256.cl — OpenCL port of kernels/sha256.{h,cu}.

   Same primitives as the CUDA version: the round macros, a context-based
   API (used by the keypair kernel to derive a per-thread seed) and the
   precomputed-schedule transform `cuda_sha256_transform_w` (used by the
   grind kernel, whose message schedule for block 1 lives in __constant). */

typedef struct {
    BYTE data[64];
    WORD datalen;
    ulong bitlen;
    WORD state[8];
} CUDA_SHA256_CTX;

#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)  (ROTRIGHT(x, 2)  ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x)  (ROTRIGHT(x, 6)  ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7)  ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

/* One SHA-256 main round. K is the round constant, M is W[round].
   Operates on the surrounding scope's a..h. */
#define VANITY_SHA_R(K, M)                              \
    do {                                                \
        WORD t1 = h + EP1(e) + CH(e, f, g) + (K) + (M); \
        WORD t2 = EP0(a) + MAJ(a, b, c);                \
        h = g; g = f; f = e;                            \
        e = d + t1;                                     \
        d = c; c = b; b = a;                            \
        a = t1 + t2;                                    \
    } while (0)

#define VANITY_SHA_ROUNDS(W)                                               \
    VANITY_SHA_R(0x428A2F98U, (W)[ 0]); VANITY_SHA_R(0x71374491U, (W)[ 1]); \
    VANITY_SHA_R(0xB5C0FBCFU, (W)[ 2]); VANITY_SHA_R(0xE9B5DBA5U, (W)[ 3]); \
    VANITY_SHA_R(0x3956C25BU, (W)[ 4]); VANITY_SHA_R(0x59F111F1U, (W)[ 5]); \
    VANITY_SHA_R(0x923F82A4U, (W)[ 6]); VANITY_SHA_R(0xAB1C5ED5U, (W)[ 7]); \
    VANITY_SHA_R(0xD807AA98U, (W)[ 8]); VANITY_SHA_R(0x12835B01U, (W)[ 9]); \
    VANITY_SHA_R(0x243185BEU, (W)[10]); VANITY_SHA_R(0x550C7DC3U, (W)[11]); \
    VANITY_SHA_R(0x72BE5D74U, (W)[12]); VANITY_SHA_R(0x80DEB1FEU, (W)[13]); \
    VANITY_SHA_R(0x9BDC06A7U, (W)[14]); VANITY_SHA_R(0xC19BF174U, (W)[15]); \
    VANITY_SHA_R(0xE49B69C1U, (W)[16]); VANITY_SHA_R(0xEFBE4786U, (W)[17]); \
    VANITY_SHA_R(0x0FC19DC6U, (W)[18]); VANITY_SHA_R(0x240CA1CCU, (W)[19]); \
    VANITY_SHA_R(0x2DE92C6FU, (W)[20]); VANITY_SHA_R(0x4A7484AAU, (W)[21]); \
    VANITY_SHA_R(0x5CB0A9DCU, (W)[22]); VANITY_SHA_R(0x76F988DAU, (W)[23]); \
    VANITY_SHA_R(0x983E5152U, (W)[24]); VANITY_SHA_R(0xA831C66DU, (W)[25]); \
    VANITY_SHA_R(0xB00327C8U, (W)[26]); VANITY_SHA_R(0xBF597FC7U, (W)[27]); \
    VANITY_SHA_R(0xC6E00BF3U, (W)[28]); VANITY_SHA_R(0xD5A79147U, (W)[29]); \
    VANITY_SHA_R(0x06CA6351U, (W)[30]); VANITY_SHA_R(0x14292967U, (W)[31]); \
    VANITY_SHA_R(0x27B70A85U, (W)[32]); VANITY_SHA_R(0x2E1B2138U, (W)[33]); \
    VANITY_SHA_R(0x4D2C6DFCU, (W)[34]); VANITY_SHA_R(0x53380D13U, (W)[35]); \
    VANITY_SHA_R(0x650A7354U, (W)[36]); VANITY_SHA_R(0x766A0ABBU, (W)[37]); \
    VANITY_SHA_R(0x81C2C92EU, (W)[38]); VANITY_SHA_R(0x92722C85U, (W)[39]); \
    VANITY_SHA_R(0xA2BFE8A1U, (W)[40]); VANITY_SHA_R(0xA81A664BU, (W)[41]); \
    VANITY_SHA_R(0xC24B8B70U, (W)[42]); VANITY_SHA_R(0xC76C51A3U, (W)[43]); \
    VANITY_SHA_R(0xD192E819U, (W)[44]); VANITY_SHA_R(0xD6990624U, (W)[45]); \
    VANITY_SHA_R(0xF40E3585U, (W)[46]); VANITY_SHA_R(0x106AA070U, (W)[47]); \
    VANITY_SHA_R(0x19A4C116U, (W)[48]); VANITY_SHA_R(0x1E376C08U, (W)[49]); \
    VANITY_SHA_R(0x2748774CU, (W)[50]); VANITY_SHA_R(0x34B0BCB5U, (W)[51]); \
    VANITY_SHA_R(0x391C0CB3U, (W)[52]); VANITY_SHA_R(0x4ED8AA4AU, (W)[53]); \
    VANITY_SHA_R(0x5B9CCA4FU, (W)[54]); VANITY_SHA_R(0x682E6FF3U, (W)[55]); \
    VANITY_SHA_R(0x748F82EEU, (W)[56]); VANITY_SHA_R(0x78A5636FU, (W)[57]); \
    VANITY_SHA_R(0x84C87814U, (W)[58]); VANITY_SHA_R(0x8CC70208U, (W)[59]); \
    VANITY_SHA_R(0x90BEFFFAU, (W)[60]); VANITY_SHA_R(0xA4506CEBU, (W)[61]); \
    VANITY_SHA_R(0xBEF9A3F7U, (W)[62]); VANITY_SHA_R(0xC67178F2U, (W)[63])

/* Big-endian byte view of `data` -> W[0..15]. */
#define VANITY_SHA_LOAD16(W, data)                                          \
    do {                                                                    \
        for (int _i = 0; _i < 16; ++_i)                                     \
            (W)[_i] = ((WORD)(data)[4*_i  ] << 24) | ((WORD)(data)[4*_i+1] << 16) \
                    | ((WORD)(data)[4*_i+2] <<  8) | ((WORD)(data)[4*_i+3]      ); \
    } while (0)

/* Standard SHA-256 message expansion W[16..63]. */
#define VANITY_SHA_EXPAND48(W)                                              \
    do {                                                                    \
        for (int _i = 16; _i < 64; ++_i)                                    \
            (W)[_i] = SIG1((W)[_i-2]) + (W)[_i-7] + SIG0((W)[_i-15]) + (W)[_i-16]; \
    } while (0)

static void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE *data) {
    WORD m[64];
    VANITY_SHA_LOAD16(m, data);
    VANITY_SHA_EXPAND48(m);

    WORD a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    WORD e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];

    VANITY_SHA_ROUNDS(m);

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

/* Runs the 64 main rounds against a precomputed schedule W[0..63] held in
   __constant memory (the grind kernel's block-1 schedule). */
static void cuda_sha256_transform_w(WORD *state, __constant const WORD *W) {
    WORD a = state[0], b = state[1], c = state[2], d = state[3];
    WORD e = state[4], f = state[5], g = state[6], h = state[7];

    VANITY_SHA_ROUNDS(W);

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void cuda_sha256_init(CUDA_SHA256_CTX *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

static void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            cuda_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

static void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE *hash) {
    WORD i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        cuda_sha256_transform(ctx, ctx->data);
        for (int j = 0; j < 56; ++j) ctx->data[j] = 0;
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[56] = ctx->bitlen >> 56;
    cuda_sha256_transform(ctx, ctx->data);

    for (int w = 0; w < 8; ++w) {
        hash[4*w + 0] = (ctx->state[w] >> 24) & 0xff;
        hash[4*w + 1] = (ctx->state[w] >> 16) & 0xff;
        hash[4*w + 2] = (ctx->state[w] >>  8) & 0xff;
        hash[4*w + 3] = (ctx->state[w] >>  0) & 0xff;
    }
}
