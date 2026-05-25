/* sha512.cl — OpenCL port of kernels/ed25519/sha512.cu.

   Only the pieces the keypair kernel touches: the context struct, the
   compression function and finalization. The kernel sets up the context
   inline (state + 32-byte message) and calls sha512_final. */

typedef struct {
    ulong length, state[8];
    size_t curlen;
    uchar buf[128];
} sha512_context;

__constant ulong K512[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

#define ROR64c(x, y) (rotate((ulong)(x), (ulong)(64 - (y))))

#define STORE64H(x, y) \
   { (y)[0] = (uchar)(((x)>>56)&255); (y)[1] = (uchar)(((x)>>48)&255); \
     (y)[2] = (uchar)(((x)>>40)&255); (y)[3] = (uchar)(((x)>>32)&255); \
     (y)[4] = (uchar)(((x)>>24)&255); (y)[5] = (uchar)(((x)>>16)&255); \
     (y)[6] = (uchar)(((x)>>8)&255);  (y)[7] = (uchar)((x)&255); }

#define LOAD64H(x, y) \
   { x = (((ulong)((y)[0] & 255))<<56)|(((ulong)((y)[1] & 255))<<48) | \
         (((ulong)((y)[2] & 255))<<40)|(((ulong)((y)[3] & 255))<<32) | \
         (((ulong)((y)[4] & 255))<<24)|(((ulong)((y)[5] & 255))<<16) | \
         (((ulong)((y)[6] & 255))<<8)|(((ulong)((y)[7] & 255))); }

#define Ch(x,y,z)  (z ^ (x & (y ^ z)))
#define Maj(x,y,z) (((x | y) & z) | (x & y))
#define S(x, n)    ROR64c(x, n)
#define R(x, n)    ((x) >> ((ulong)n))
#define Sigma0(x)  (S(x, 28) ^ S(x, 34) ^ S(x, 39))
#define Sigma1(x)  (S(x, 14) ^ S(x, 18) ^ S(x, 41))
#define Gamma0(x)  (S(x, 1) ^ S(x, 8) ^ R(x, 7))
#define Gamma1(x)  (S(x, 19) ^ S(x, 61) ^ R(x, 6))

static int sha512_compress(sha512_context *md, uchar *buf) {
    ulong S512[8], W[80], t0, t1;
    int i;

    for (i = 0; i < 8; i++) S512[i] = md->state[i];
    for (i = 0; i < 16; i++) { LOAD64H(W[i], buf + (8*i)); }
    for (i = 16; i < 80; i++)
        W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];

    #define RND512(a,b,c,d,e,f,g,h,i) \
        t0 = h + Sigma1(e) + Ch(e, f, g) + K512[i] + W[i]; \
        t1 = Sigma0(a) + Maj(a, b, c); \
        d += t0; \
        h  = t0 + t1;

    for (i = 0; i < 80; i += 8) {
       RND512(S512[0],S512[1],S512[2],S512[3],S512[4],S512[5],S512[6],S512[7],i+0);
       RND512(S512[7],S512[0],S512[1],S512[2],S512[3],S512[4],S512[5],S512[6],i+1);
       RND512(S512[6],S512[7],S512[0],S512[1],S512[2],S512[3],S512[4],S512[5],i+2);
       RND512(S512[5],S512[6],S512[7],S512[0],S512[1],S512[2],S512[3],S512[4],i+3);
       RND512(S512[4],S512[5],S512[6],S512[7],S512[0],S512[1],S512[2],S512[3],i+4);
       RND512(S512[3],S512[4],S512[5],S512[6],S512[7],S512[0],S512[1],S512[2],i+5);
       RND512(S512[2],S512[3],S512[4],S512[5],S512[6],S512[7],S512[0],S512[1],i+6);
       RND512(S512[1],S512[2],S512[3],S512[4],S512[5],S512[6],S512[7],S512[0],i+7);
    }
    #undef RND512

    for (i = 0; i < 8; i++) md->state[i] = md->state[i] + S512[i];
    return 0;
}

static int sha512_final(sha512_context *md, uchar *out) {
    int i;

    md->length += md->curlen * 8UL;
    md->buf[md->curlen++] = (uchar)0x80;

    if (md->curlen > 112) {
        while (md->curlen < 128) md->buf[md->curlen++] = (uchar)0;
        sha512_compress(md, md->buf);
        md->curlen = 0;
    }

    while (md->curlen < 120) md->buf[md->curlen++] = (uchar)0;

    STORE64H(md->length, md->buf + 120);
    sha512_compress(md, md->buf);

    for (i = 0; i < 8; i++) { STORE64H(md->state[i], out + (8*i)); }
    return 0;
}

#undef Ch
#undef Maj
#undef S
#undef R
#undef Sigma0
#undef Sigma1
#undef Gamma0
#undef Gamma1
