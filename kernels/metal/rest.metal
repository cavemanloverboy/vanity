// Group, hashes, base58, and grind kernels. Concatenated after field.metal.
// Host prepends COMB_W, KP_BATCH, USE_TG, DUAL, BS58_MAGIC.

#ifndef COMB_W
#define COMB_W 8
#endif
#ifndef KP_BATCH
#define KP_BATCH 1
#endif
#ifndef USE_TG
#define USE_TG 1
#endif
#ifndef DUAL
#define DUAL 0
#endif
#ifndef BS58_MAGIC
#define BS58_MAGIC 1
#endif

#if COMB_W < 5 || COMB_W > 12
#error COMB_W must be 5..12
#endif
#if KP_BATCH < 1 || KP_BATCH > 8
#error KP_BATCH must be 1..8
#endif

#ifndef LAYOUT
#define LAYOUT 0
#endif
#ifndef PREFETCH
#define PREFETCH 0
#endif
#ifndef REG_CAP
#define REG_CAP 0
#endif
#if REG_CAP > 0
#define REG_ATTR [[max_total_threads_per_threadgroup(REG_CAP)]]
#else
#define REG_ATTR
#endif

#define COMB_WINDOWS ((256 + COMB_W - 1) / COMB_W)
#define COMB_POS (1 << (COMB_W - 1))
#define COMB_TABLE_LEN (COMB_WINDOWS * COMB_POS)
#define KEYS_PER (KP_BATCH * (DUAL ? 2 : 1))

#if USE_TG && (COMB_POS * 30 * 4 > 32768)
#error comb window does not fit in 32KB threadgroup memory
#endif
#if USE_TG
#define TG_INTS (COMB_POS * 30)
#else
#define TG_INTS 4
#endif

#define VANITY_MAX_PATTERNS 64
#define VANITY_MAX_PATTERN_LEN 44
#define VANITY_PT_N 0
#define VANITY_PT_PLEN 4
#define VANITY_PT_SLEN 68
#define VANITY_PT_PREF 132
#define VANITY_PT_SUF 2948
#define VANITY_PT_ACTIVE 5768

constant uchar BASE_X[32] = {
    26, 213, 37, 143, 96, 45, 86, 201, 178, 167, 37, 149, 96, 199, 44, 105,
    92, 220, 214, 253, 49, 226, 164, 192, 254, 83, 110, 205, 211, 54, 105, 33};
constant uchar BASE_Y[32] = {
    88, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,
    102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102};

constant ulong K512[80] = {
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
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL};

constant uint ENC58[8][8] = {
    {513735U, 77223048U, 437087610U, 300156666U, 605448490U, 214625350U, 141436834U, 379377856U},
    {0U, 78508U, 646269101U, 118408823U, 91512303U, 209184527U, 413102373U, 153715680U},
    {0U, 0U, 11997U, 486083817U, 3737691U, 294005210U, 247894721U, 289024608U},
    {0U, 0U, 0U, 1833U, 324463681U, 385795061U, 551597588U, 21339008U},
    {0U, 0U, 0U, 0U, 280U, 127692781U, 389432875U, 357132832U},
    {0U, 0U, 0U, 0U, 0U, 42U, 537767569U, 410450016U},
    {0U, 0U, 0U, 0U, 0U, 0U, 6U, 356826688U},
    {0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U}};

static ulong rotr64(ulong x, uint n) { return (x >> n) | (x << (64u - n)); }

static ulong umulhi64(ulong a, ulong b) {
    ulong a_lo = a & 0xffffffffUL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xffffffffUL;
    ulong b_hi = b >> 32;
    ulong p0 = a_lo * b_lo;
    ulong p1 = a_lo * b_hi;
    ulong p2 = a_hi * b_lo;
    ulong p3 = a_hi * b_hi;
    ulong mid = (p0 >> 32) + (p1 & 0xffffffffUL) + (p2 & 0xffffffffUL);
    return p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}

static ulong div_58_5(ulong n) {
#if BS58_MAGIC
    return umulhi64(n, 0x68b2c7ad1a016ab5UL) >> 28;
#else
    return n / 656356768UL;
#endif
}

#ifndef SHA_FOLD
#define SHA_FOLD 1
#endif
#ifndef SHA_ATTR
#define SHA_ATTR __attribute__((noinline))
#endif
#ifndef STOP_EVERY
#define STOP_EVERY 64
#endif

#define G0(x) (rotr64((x), 1) ^ rotr64((x), 8) ^ ((x) >> 7))
#define G1(x) (rotr64((x), 19) ^ rotr64((x), 61) ^ ((x) >> 6))

// Kept outlined so the 80-word schedule does not spill inside the comb.
SHA_ATTR
static void sha512_32(thread const uchar *seed, thread uchar *out) {
    ulong S[8] = {
        0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
        0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL};
    ulong W[80];
    for (int i = 0; i < 4; i++) {
        ulong x = 0;
        for (int b = 0; b < 8; b++) x = (x << 8) | seed[8 * i + b];
        W[i] = x;
    }
    W[4] = 0x8000000000000000UL;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 256UL;
#if SHA_FOLD
    // W[5..14] are zero and W[15] is 256, so the first schedule words drop terms.
    W[16] = G0(W[1]) + W[0];
    W[17] = 9007199254743044UL + G0(W[2]) + W[1];
    W[18] = G1(W[16]) + G0(W[3]) + W[2];
    W[19] = G1(W[17]) + 4719772409484279808UL + W[3];
    W[20] = G1(W[18]) + 9223372036854775808UL;
    W[21] = G1(W[19]);
    W[22] = G1(W[20]) + 256UL;
    W[23] = G1(W[21]) + W[16];
    W[24] = G1(W[22]) + W[17];
    W[25] = G1(W[23]) + W[18];
    W[26] = G1(W[24]) + W[19];
    W[27] = G1(W[25]) + W[20];
    W[28] = G1(W[26]) + W[21];
    W[29] = G1(W[27]) + W[22];
    W[30] = G1(W[28]) + W[23] + 131UL;
    W[31] = G1(W[29]) + W[24] + G0(W[16]) + 256UL;
    for (int i = 32; i < 80; i++)
        W[i] = G1(W[i - 2]) + W[i - 7] + G0(W[i - 15]) + W[i - 16];
#else
    for (int i = 16; i < 80; i++) {
        ulong w2 = W[i - 2];
        ulong w15 = W[i - 15];
        ulong g1 = rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6);
        ulong g0 = rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7);
        W[i] = g1 + W[i - 7] + g0 + W[i - 16];
    }
#endif
#pragma unroll
    for (int i = 0; i < 80; i++) {
        ulong a = S[0], b = S[1], c = S[2], d = S[3], e = S[4], f = S[5], g = S[6], h = S[7];
        ulong s1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        ulong ch = (e & f) ^ (~e & g);
        ulong t0 = h + s1 + ch + K512[i] + W[i];
        ulong s0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        ulong maj = (a & b) ^ (a & c) ^ (b & c);
        S[7] = g; S[6] = f; S[5] = e; S[4] = d + t0;
        S[3] = c; S[2] = b; S[1] = a; S[0] = t0 + s0 + maj;
    }
    ulong iv[8] = {
        0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
        0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL};
    for (int i = 0; i < 8; i++) {
        ulong x = S[i] + iv[i];
        for (int b = 0; b < 8; b++) out[8 * i + b] = (uchar)(x >> (56 - 8 * b));
    }
}

constant uint K256[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

static uint rotr32(uint x, uint n) { return (x >> n) | (x << (32u - n)); }

static void sha256_compress(thread uint *st, thread const uint *in) {
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = in[i];
    for (int i = 16; i < 64; i++) {
        uint s0 = rotr32(W[i - 15], 7) ^ rotr32(W[i - 15], 18) ^ (W[i - 15] >> 3);
        uint s1 = rotr32(W[i - 2], 17) ^ rotr32(W[i - 2], 19) ^ (W[i - 2] >> 10);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint a = st[0], b = st[1], c = st[2], d = st[3];
    uint e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i++) {
        uint S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + K256[i] + W[i];
        uint S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + S0 + maj;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

// SHA-256 of a 40-byte message (host seed || le64 tid). One block.
static void sha256_40(thread const uchar *msg, thread uchar *out) {
    uint st[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                  0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    uchar block[64];
    for (int i = 0; i < 40; i++) block[i] = msg[i];
    block[40] = 0x80;
    for (int i = 41; i < 64; i++) block[i] = 0;
    block[62] = 0x01;
    block[63] = 0x40; // 320 bits
    uint w[16];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint)block[4 * i] << 24) | ((uint)block[4 * i + 1] << 16) |
               ((uint)block[4 * i + 2] << 8) | (uint)block[4 * i + 3];
    sha256_compress(st, w);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uchar)(st[i] >> 24);
        out[4 * i + 1] = (uchar)(st[i] >> 16);
        out[4 * i + 2] = (uchar)(st[i] >> 8);
        out[4 * i + 3] = (uchar)st[i];
    }
}

static void seed_from_idx(thread uchar *seed, device const uchar *host, ulong idx) {
    uchar msg[40];
    for (int i = 0; i < 32; i++) msg[i] = host[i];
    for (int b = 0; b < 8; b++) msg[32 + b] = (uchar)(idx >> (8 * b));
    sha256_40(msg, seed);
}

static void point_double(thread int *X, thread int *Y, thread int *Z, thread int *T) {
    fe x, y, z;
    fe_copy(x, X);
    fe_copy(y, Y);
    fe_copy(z, Z);
    fe XX, YY, ZZ2, sum, sumsq, rX, rY, rZ, rT;
    fe_sq(XX, x);
    fe_sq(YY, y);
    fe_sq2(ZZ2, z);
    fe_add(sum, x, y);
    fe_sq(sumsq, sum);
    fe_add(rY, YY, XX);
    fe_sub(rZ, YY, XX);
    fe_sub(rX, sumsq, rY);
    fe_sub(rT, ZZ2, rZ);
    fe_mul(X, rX, rT);
    fe_mul(Y, rY, rZ);
    fe_mul(Z, rZ, rT);
    fe_mul(T, rX, rY);
}

static void add_projective(thread int *X, thread int *Y, thread int *Z, thread int *T,
                           thread const fe ypx, thread const fe ymx, thread const fe nz, thread const fe t2d) {
    fe y_plus_x, y_minus_x, pp, mm, tt, zz, zz2;
    fe_add(y_plus_x, Y, X);
    fe_sub(y_minus_x, Y, X);
    fe_mul(pp, y_plus_x, ypx);
    fe_mul(mm, y_minus_x, ymx);
    fe_mul(tt, T, t2d);
    fe_mul(zz, Z, nz);
    fe_add(zz2, zz, zz);
    fe rx, ry, rz, rt;
    fe_sub(rx, pp, mm);
    fe_add(ry, pp, mm);
    fe_add(rz, zz2, tt);
    fe_sub(rt, zz2, tt);
    fe_mul(X, rx, rt);
    fe_mul(Y, ry, rz);
    fe_mul(Z, rz, rt);
    fe_mul(T, rx, ry);
}

static void add_affine(thread int *X, thread int *Y, thread int *Z, thread int *T,
                       thread const fe ypx, thread const fe ymx, thread const fe t2d) {
    fe y_plus_x, y_minus_x, pp, mm, tt, zz2;
    fe_add(y_plus_x, Y, X);
    fe_sub(y_minus_x, Y, X);
    fe_mul(pp, y_plus_x, ypx);
    fe_mul(mm, y_minus_x, ymx);
    fe_mul(tt, T, t2d);
    fe_add(zz2, Z, Z);
    fe rx, ry, rz, rt;
    fe_sub(rx, pp, mm);
    fe_add(ry, pp, mm);
    fe_add(rz, zz2, tt);
    fe_sub(rt, zz2, tt);
    fe_mul(X, rx, rt);
    fe_mul(Y, ry, rz);
    fe_mul(Z, rz, rt);
    fe_mul(T, rx, ry);
}

// One cache line per table entry: 32 ints = 128 bytes (30 live, 2 pad).
static void load_aos128(thread fe ypx, thread fe ymx, thread fe t2d, int digit, device const int *window_base) {
    int neg = digit < 0;
    int ad = neg ? -digit : digit;
    int idx = ad == 0 ? 0 : ad - 1;
    device const int4 *rec = reinterpret_cast<device const int4 *>(window_base + idx * 32);
    int4 q0 = rec[0], q1 = rec[1], q2 = rec[2], q3 = rec[3];
    int4 q4 = rec[4], q5 = rec[5], q6 = rec[6], q7 = rec[7];
    int a[10] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w, q2.x, q2.y};
    int b[10] = {q2.z, q2.w, q3.x, q3.y, q3.z, q3.w, q4.x, q4.y, q4.z, q4.w};
    int c[10] = {q5.x, q5.y, q5.z, q5.w, q6.x, q6.y, q6.z, q6.w, q7.x, q7.y};
    for (int k = 0; k < 10; k++) {
        int is0 = ad == 0;
        int aa = is0 ? (k == 0) : a[k];
        int bb = is0 ? (k == 0) : b[k];
        int cc = is0 ? 0 : c[k];
        ypx[k] = neg ? bb : aa;
        ymx[k] = neg ? aa : bb;
        t2d[k] = neg ? -cc : cc;
    }
}

static void load_affine(thread fe ypx, thread fe ymx, thread fe t2d, int digit,
#if USE_TG
                        threadgroup const int *win
#else
                        device const int *win
#endif
) {
    int neg = digit < 0;
    int ad = neg ? -digit : digit;
    int idx = ad == 0 ? 0 : ad - 1;
    for (int k = 0; k < 10; k++) {
        int a = win[(0 * 10 + k) * COMB_POS + idx];
        int b = win[(1 * 10 + k) * COMB_POS + idx];
        int c = win[(2 * 10 + k) * COMB_POS + idx];
        int is0 = ad == 0;
        a = is0 ? (k == 0) : a;
        b = is0 ? (k == 0) : b;
        c = is0 ? 0 : c;
        int y1 = neg ? b : a;
        int y2 = neg ? a : b;
        int td = neg ? -c : c;
        ypx[k] = y1;
        ymx[k] = y2;
        t2d[k] = td;
    }
}

static void stage_window(threadgroup int *win, device const int *comb, int window, uint lane, uint tgsize) {
#if USE_TG
    int nwords = COMB_POS * 30;
    device const int *src = comb + window * nwords;
    for (uint i = lane; i < (uint)nwords; i += tgsize) win[i] = src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#else
    (void)win; (void)comb; (void)window; (void)lane; (void)tgsize;
#endif
}

static void to_comb_digits(thread short *e, thread const uchar *a) {
    uint mask = (1u << COMB_W) - 1u;
    for (int i = 0; i < COMB_WINDOWS; i++) {
        int bit = COMB_W * i;
        int byte = bit >> 3;
        int off = bit & 7;
        uint word = 0;
        if (byte < 32) word |= (uint)a[byte];
        if (byte + 1 < 32) word |= (uint)a[byte + 1] << 8;
        if (byte + 2 < 32) word |= (uint)a[byte + 2] << 16;
        e[i] = (short)((word >> off) & mask);
    }
    int carry = 0;
    for (int i = 0; i < COMB_WINDOWS - 1; i++) {
        int digit = (int)e[i] + carry;
        carry = (digit + COMB_POS) >> COMB_W;
        e[i] = (short)(digit - (carry << COMB_W));
    }
    e[COMB_WINDOWS - 1] = (short)((int)e[COMB_WINDOWS - 1] + carry);
}

// All threads in the group must call this (threadgroup barriers).
static void scalarmult_many(thread int *X, thread int *Y, thread int *Z, thread int *T,
                            thread const short *digits, device const int *comb,
                            threadgroup int *win, uint lane, uint tgsize) {
    for (int j = 0; j < KEYS_PER; j++) {
        fe_0(X + j * 10);
        fe_1(Y + j * 10);
        fe_1(Z + j * 10);
        fe_0(T + j * 10);
    }
#if LAYOUT == 1 && PREFETCH
    // Issue the next entry's 128-byte load before the current 7-mul add so
    // the cache fill overlaps the arithmetic.
    {
        fe ypx0[KEYS_PER], ymx0[KEYS_PER], t2d0[KEYS_PER];
        fe ypx1[KEYS_PER], ymx1[KEYS_PER], t2d1[KEYS_PER];
        for (int j = 0; j < KEYS_PER; j++)
            load_aos128(ypx0[j], ymx0[j], t2d0[j], digits[j * COMB_WINDOWS], comb);
        int cur = 0;
        for (int w = 0; w < COMB_WINDOWS; w++) {
            if (w + 1 < COMB_WINDOWS) {
                for (int j = 0; j < KEYS_PER; j++) {
                    if (cur == 0)
                        load_aos128(ypx1[j], ymx1[j], t2d1[j], digits[j * COMB_WINDOWS + w + 1],
                                    comb + (w + 1) * COMB_POS * 32);
                    else
                        load_aos128(ypx0[j], ymx0[j], t2d0[j], digits[j * COMB_WINDOWS + w + 1],
                                    comb + (w + 1) * COMB_POS * 32);
                }
            }
#pragma unroll
            for (int j = 0; j < KEYS_PER; j++) {
                if (cur == 0)
                    add_affine(X + j * 10, Y + j * 10, Z + j * 10, T + j * 10, ypx0[j], ymx0[j], t2d0[j]);
                else
                    add_affine(X + j * 10, Y + j * 10, Z + j * 10, T + j * 10, ypx1[j], ymx1[j], t2d1[j]);
            }
            cur ^= 1;
        }
    }
#else
    for (int w = 0; w < COMB_WINDOWS; w++) {
#if USE_TG && LAYOUT == 0
        if (w > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        stage_window(win, comb, w, lane, tgsize);
#endif
#pragma unroll
        for (int j = 0; j < KEYS_PER; j++) {
            fe ypx, ymx, t2d;
#if LAYOUT == 1
            load_aos128(ypx, ymx, t2d, digits[j * COMB_WINDOWS + w], comb + w * COMB_POS * 32);
#elif USE_TG
            load_affine(ypx, ymx, t2d, digits[j * COMB_WINDOWS + w], win);
#else
            load_affine(ypx, ymx, t2d, digits[j * COMB_WINDOWS + w],
                        comb + w * COMB_POS * 30);
#endif
            add_affine(X + j * 10, Y + j * 10, Z + j * 10, T + j * 10, ypx, ymx, t2d);
        }
    }
#endif
}

static void batch_invert(thread int *zs) {
    int scratch[KEYS_PER * 10];
    fe acc;
    fe_1(acc);
    for (int i = 0; i < KEYS_PER; i++) {
        fe_copy(scratch + i * 10, acc);
        fe_mul(acc, acc, zs + i * 10);
    }
    fe_invert(acc, acc);
    for (int i = KEYS_PER - 1; i >= 0; i--) {
        fe tmp;
        fe_mul(tmp, acc, zs + i * 10);
        fe_mul(zs + i * 10, acc, scratch + i * 10);
        fe_copy(acc, tmp);
    }
}

static void compress_at(thread uchar *out, thread const int *X, thread const int *Y, thread const int *Zinv) {
    fe x, y;
    fe_mul(x, X, Zinv);
    fe_mul(y, Y, Zinv);
    fe_tobytes(out, y);
    out[31] ^= (uchar)(fe_isnegative(x) << 7);
}

// seeds: KEYS_PER * 32, replaced by the next chain seed. pubs: KEYS_PER * 32.
static void keygen_keys(thread uchar *seeds, thread uchar *pubs, device const int *comb,
                        threadgroup int *win, uint lane, uint tgsize) {
    short digits[KEYS_PER * COMB_WINDOWS];
    int X[KEYS_PER * 10], Y[KEYS_PER * 10], Z[KEYS_PER * 10], T[KEYS_PER * 10];
    for (int j = 0; j < KEYS_PER; j++) {
        uchar digest[64];
        sha512_32(seeds + j * 32, digest);
        uchar scalar[32];
        for (int i = 0; i < 32; i++) scalar[i] = digest[i];
        scalar[0] &= 248;
        scalar[31] &= 63;
        scalar[31] |= 64;
        to_comb_digits(digits + j * COMB_WINDOWS, scalar);
        for (int i = 0; i < 32; i++) seeds[j * 32 + i] = digest[32 + i];
    }
    scalarmult_many(X, Y, Z, T, digits, comb, win, lane, tgsize);
    batch_invert(Z);
    for (int j = 0; j < KEYS_PER; j++)
        compress_at(pubs + j * 32, X + j * 10, Y + j * 10, Z + j * 10);
}

static uint doppler_count(thread const uchar *pk) {
    uint matched = 0;
    for (int s = 0; s < 4; s++) {
        int o = s * 8;
        uchar fill = (pk[o + 3] & 0x80) ? 0xff : 0x00;
        if (pk[o + 4] == fill && pk[o + 5] == fill && pk[o + 6] == fill && pk[o + 7] == fill)
            matched++;
    }
    return matched;
}

static void b58_raw(thread const uint words[8], thread uchar *raw, thread ulong *skip_out, thread ulong *len_out) {
    ulong in_leading_0s = 0;
    for (int i = 0; i < 8; i++) {
        ulong lz_bytes = (ulong)(clz(words[i]) >> 3);
        in_leading_0s += lz_bytes;
        if (lz_bytes != 4UL) break;
    }
    ulong inter[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            inter[j + 1] += (ulong)words[i] * (ulong)ENC58[i][j];
    for (int i = 8; i > 0; i--) {
        ulong q = div_58_5(inter[i]);
        inter[i - 1] += q;
        inter[i] -= q * 656356768UL;
    }
    ulong limbs = 0;
    for (int L = 0; L < 9; L++) {
        uint v = (uint)inter[L];
        raw[5 * L + 4] = (uchar)(v % 58U);
        raw[5 * L + 3] = (uchar)((v / 58U) % 58U);
        raw[5 * L + 2] = (uchar)((v / 3364U) % 58U);
        raw[5 * L + 1] = (uchar)((v / 195112U) % 58U);
        raw[5 * L + 0] = (uchar)(v / 11316496U);
        limbs++;
    }
    ulong rl = 0;
    while (rl < 45UL && raw[rl] == 0) rl++;
    if (rl == 45UL) rl = 44UL;
    ulong skip = rl - in_leading_0s;
    *skip_out = skip;
    *len_out = 45UL - skip;
}

static bool b58_match_one(thread const uint words[8], device const uchar *patterns, device const uchar *lut) {
    uchar raw[45];
    ulong skip, elen;
    b58_raw(words, raw, &skip, &elen);
    uint plen = patterns[VANITY_PT_PLEN];
    uint slen = patterns[VANITY_PT_SLEN];
    for (uint i = 0; i < plen; i++) {
        if (lut[raw[skip + i]] != patterns[VANITY_PT_PREF + i]) return false;
    }
    if (slen) {
        ulong tail = skip + elen - slen;
        for (uint i = 0; i < slen; i++) {
            if (lut[raw[tail + i]] != patterns[VANITY_PT_SUF + i]) return false;
        }
    }
    return true;
}

static bool b58_match_any(thread const uint words[8], device const uchar *patterns, device const uchar *lut) {
    uchar raw[45];
    ulong skip, elen;
    b58_raw(words, raw, &skip, &elen);
    uint n = (uint)patterns[0] | ((uint)patterns[1] << 8) | ((uint)patterns[2] << 16) | ((uint)patterns[3] << 24);
    if (n == 0) return true;
    ulong active = 0;
    for (uint b = 0; b < 8; b++) active |= ((ulong)patterns[VANITY_PT_ACTIVE + b]) << (8 * b);
    for (uint pi = 0; pi < n; pi++) {
        if ((active & (1UL << pi)) == 0) continue;
        uint plen = patterns[VANITY_PT_PLEN + pi];
        uint slen = patterns[VANITY_PT_SLEN + pi];
        bool ok = true;
        for (uint i = 0; i < plen; i++) {
            if (lut[raw[skip + i]] != patterns[VANITY_PT_PREF + pi * VANITY_MAX_PATTERN_LEN + i]) {
                ok = false;
                break;
            }
        }
        if (ok && slen) {
            ulong tail = skip + elen - slen;
            for (uint i = 0; i < slen; i++) {
                if (lut[raw[tail + i]] != patterns[VANITY_PT_SUF + pi * VANITY_MAX_PATTERN_LEN + i]) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok) return true;
    }
    return false;
}

static bool b58_match(thread const uchar *pk, device const uchar *patterns, device const uchar *lut) {
    uint words[8];
    for (int k = 0; k < 8; k++)
        words[k] = ((uint)pk[4 * k] << 24) | ((uint)pk[4 * k + 1] << 16) |
                   ((uint)pk[4 * k + 2] << 8) | (uint)pk[4 * k + 3];
    uint n = (uint)patterns[0] | ((uint)patterns[1] << 8) | ((uint)patterns[2] << 16) | ((uint)patterns[3] << 24);
    if (n <= 1) return b58_match_one(words, patterns, lut);
    return b58_match_any(words, patterns, lut);
}

// 58^4 long division. Digits are the significant base58 indexes, LSD-last.
static uint b58_digits_e4(thread const uchar *pk, thread uchar *digits) {
    uint limbs[8];
    for (int i = 0; i < 8; i++)
        limbs[i] = ((uint)pk[4 * i] << 24) | ((uint)pk[4 * i + 1] << 16) |
                   ((uint)pk[4 * i + 2] << 8) | (uint)pk[4 * i + 3];
    uint z = 0;
    for (int i = 0; i < 32; i++) {
        if (pk[i] == 0) z++;
        else break;
    }
    uchar rev[64];
    uint nrev = 0;
    while (nrev < 64) {
        bool any = false;
        for (int i = 0; i < 8; i++)
            if (limbs[i]) any = true;
        if (!any) break;
        uint rem = 0;
        for (int i = 0; i < 8; i++) {
            ulong cur = ((ulong)rem << 32) | limbs[i];
            limbs[i] = (uint)(cur / 11316496UL);
            rem = (uint)(cur % 11316496UL);
        }
        for (int k = 0; k < 4 && nrev < 64; k++) {
            rev[nrev++] = (uchar)(rem % 58U);
            rem /= 58U;
        }
    }
    while (nrev > 0 && rev[nrev - 1] == 0) nrev--;
    uint n = 0;
    for (uint i = 0; i < z; i++) digits[n++] = 0;
    for (uint i = 0; i < nrev; i++) digits[n++] = rev[nrev - 1 - i];
    return n;
}

static uint group_stop(threadgroup uint *flag, device atomic_uint *done, uint lane) {
    if (lane == 0) *flag = atomic_load_explicit(done, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *flag;
}

static void claim_seed(device atomic_uint *done, device uchar *out, thread const uchar *seed) {
    uint expected = 0;
    if (atomic_compare_exchange_weak_explicit(done, &expected, 1u, memory_order_relaxed, memory_order_relaxed)) {
        for (int i = 0; i < 32; i++) out[i] = seed[i];
    }
}

kernel void vanity_doppler_search REG_ATTR (
    device const uchar *host_seed [[buffer(0)]],
    device const int *comb [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    device atomic_uint *done [[buffer(3)]],
    device uint *counts [[buffer(4)]],
    constant uint &required [[buffer(5)]],
    constant uint &max_iters [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
    threadgroup int win[TG_INTS];
    threadgroup uint stopf;
    uchar seeds[KEYS_PER * 32];
    uchar pubs[KEYS_PER * 32];
    uchar used[KEYS_PER * 32];
    for (int j = 0; j < KEYS_PER; j++) {
        ulong idx = (ulong)tid * (ulong)KEYS_PER + (ulong)j;
        seed_from_idx(seeds + j * 32, host_seed, idx);
    }
    uint keys = 0;
    while (keys + KEYS_PER <= max_iters) {
        // Uniform across the group: keys steps by KEYS_PER.
        if ((keys & (STOP_EVERY - 1u)) == 0u && group_stop(&stopf, done, lane)) break;
        for (int j = 0; j < KEYS_PER; j++)
            for (int i = 0; i < 32; i++) used[j * 32 + i] = seeds[j * 32 + i];
        keygen_keys(seeds, pubs, comb, win, lane, tgsize);
        for (int j = 0; j < KEYS_PER; j++) {
            if (doppler_count(pubs + j * 32) >= required)
                claim_seed(done, out, used + j * 32);
        }
        keys += KEYS_PER;
    }
    counts[tid] = keys;
}

kernel void vanity_keypair_search REG_ATTR (
    device const uchar *host_seed [[buffer(0)]],
    device const int *comb [[buffer(1)]],
    device const uchar *lut [[buffer(2)]],
    device const uchar *patterns [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    device atomic_uint *done [[buffer(5)]],
    device uint *counts [[buffer(6)]],
    constant uint &max_iters [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
    threadgroup int win[TG_INTS];
    threadgroup uint stopf;
    uchar seeds[KEYS_PER * 32];
    uchar pubs[KEYS_PER * 32];
    uchar used[KEYS_PER * 32];
    for (int j = 0; j < KEYS_PER; j++) {
        ulong idx = (ulong)tid * (ulong)KEYS_PER + (ulong)j;
        seed_from_idx(seeds + j * 32, host_seed, idx);
    }
    uint keys = 0;
    while (keys + KEYS_PER <= max_iters) {
        if ((keys & (STOP_EVERY - 1u)) == 0u && group_stop(&stopf, done, lane)) break;
        for (int j = 0; j < KEYS_PER; j++)
            for (int i = 0; i < 32; i++) used[j * 32 + i] = seeds[j * 32 + i];
        keygen_keys(seeds, pubs, comb, win, lane, tgsize);
        for (int j = 0; j < KEYS_PER; j++) {
            if (b58_match(pubs + j * 32, patterns, lut))
                claim_seed(done, out, used + j * 32);
        }
        keys += KEYS_PER;
    }
    counts[tid] = keys;
}

// Direct seeds, no SHA-256 mix. Thread tid of the grid computes seeds[tid].
kernel void kat_pubkeys(
    device const int *comb [[buffer(0)]],
    device const uchar *seeds_in [[buffer(1)]],
    device uchar *pubs [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
    threadgroup int win[TG_INTS];
    uchar seed[32];
    bool live = tid < n;
    if (live) {
        for (int i = 0; i < 32; i++) seed[i] = seeds_in[tid * 32 + i];
    } else {
        for (int i = 0; i < 32; i++) seed[i] = 0;
    }
    // One key per thread. KEYS_PER may be > 1; only slot 0 is the real seed.
    uchar seeds[KEYS_PER * 32];
    uchar outp[KEYS_PER * 32];
    for (int j = 0; j < KEYS_PER; j++)
        for (int i = 0; i < 32; i++) seeds[j * 32 + i] = (j == 0) ? seed[i] : 0;
    // keygen_keys hashes the seed. For KAT we want pubkey(seed), and keygen_keys
    // does exactly SHA-512(seed) -> clamp -> scalarmult. Slot 0 is the seed.
    keygen_keys(seeds, outp, comb, win, lane, tgsize);
    if (live)
        for (int i = 0; i < 32; i++) pubs[tid * 32 + i] = outp[i];
}

kernel void build_comb(device int *scratch [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    fe d2;
    {
        fe t, inv, d;
        fe_0(t);
        t[0] = 121666;
        fe_invert(inv, t);
        fe_0(t);
        t[0] = 121665;
        fe_neg(t, t);
        fe_mul(d, t, inv);
        fe_add(d2, d, d);
    }
    int X[10], Y[10], Z[10], T[10];
    uchar bx[32], by[32];
    for (int i = 0; i < 32; i++) {
        bx[i] = BASE_X[i];
        by[i] = BASE_Y[i];
    }
    fe_frombytes(X, bx);
    fe_frombytes(Y, by);
    fe_1(Z);
    fe_mul(T, X, Y);

    for (int w = 0; w < COMB_WINDOWS; w++) {
        // Fixed niels of this window's base. Every entry adds this, not the
        // running multiple (that would double instead of stepping by +base).
        fe bypx, bymx, bz, bt2d;
        fe_add(bypx, Y, X);
        fe_sub(bymx, Y, X);
        fe_copy(bz, Z);
        fe_mul(bt2d, T, d2);

        int mX[10], mY[10], mZ[10], mT[10];
        fe_copy(mX, X);
        fe_copy(mY, Y);
        fe_copy(mZ, Z);
        fe_copy(mT, T);
        fe syx, sym, sz, st;
        fe_copy(syx, bypx);
        fe_copy(sym, bymx);
        fe_copy(sz, bz);
        fe_copy(st, bt2d);

        for (int k = 0; k < COMB_POS; k++) {
            device int *dst = scratch + ((w * COMB_POS + k) * 40);
            for (int i = 0; i < 10; i++) {
                dst[i] = syx[i];
                dst[10 + i] = sym[i];
                dst[20 + i] = sz[i];
                dst[30 + i] = st[i];
            }
            if (k + 1 == COMB_POS) break;
            add_projective(mX, mY, mZ, mT, bypx, bymx, bz, bt2d);
            fe_add(syx, mY, mX);
            fe_sub(sym, mY, mX);
            fe_copy(sz, mZ);
            fe_mul(st, mT, d2);
        }
        for (int d = 0; d < COMB_W; d++) point_double(X, Y, Z, T);
    }
}

kernel void normalize_comb(device const int *scratch [[buffer(0)]], device int *out [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= COMB_TABLE_LEN) return;
    device const int *src = scratch + i * 40;
    fe ypx, ymx, z, t2d, inv;
    for (int k = 0; k < 10; k++) {
        ypx[k] = src[k];
        ymx[k] = src[10 + k];
        z[k] = src[20 + k];
        t2d[k] = src[30 + k];
    }
    fe_invert(inv, z);
    fe_mul(ypx, ypx, inv);
    fe_mul(ymx, ymx, inv);
    fe_mul(t2d, t2d, inv);
#if LAYOUT == 1
    device int *dst = out + i * 32;
    for (int k = 0; k < 10; k++) {
        dst[k] = ypx[k];
        dst[10 + k] = ymx[k];
        dst[20 + k] = t2d[k];
    }
    dst[30] = 0;
    dst[31] = 0;
#else
    int window = (int)(i / COMB_POS);
    int entry = (int)(i % COMB_POS);
    device int *words = out + window * COMB_POS * 30;
    for (int k = 0; k < 10; k++) {
        words[(0 * 10 + k) * COMB_POS + entry] = ypx[k];
        words[(1 * 10 + k) * COMB_POS + entry] = ymx[k];
        words[(2 * 10 + k) * COMB_POS + entry] = t2d[k];
    }
#endif
}

kernel void bench_fe_mul(device int *sink [[buffer(0)]], constant uint &n [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    fe a, b;
    fe_1(a);
    fe_1(b);
    b[0] = 2 + (int)(tid & 7u);
    for (uint i = 0; i < n; i++) fe_mul(a, a, b);
    sink[tid] = a[0] ^ a[9];
}

kernel void bench_sha512(device uint *sink [[buffer(0)]], constant uint &n [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    uchar seed[32];
    for (int i = 0; i < 32; i++) seed[i] = (uchar)(tid + i);
    uchar out[64];
    for (uint i = 0; i < n; i++) {
        sha512_32(seed, out);
        seed[0] = out[0];
        seed[1] = out[1];
    }
    sink[tid] = out[0];
}

kernel void bench_b58(device uint *sink [[buffer(0)]], constant uint &n [[buffer(1)]],
                      device const uchar *lut [[buffer(2)]], device const uchar *patterns [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
    uchar pk[32];
    for (int i = 0; i < 32; i++) pk[i] = (uchar)(i * 3 + tid);
    uint acc = 0;
    for (uint i = 0; i < n; i++) {
        pk[0] = (uchar)i;
        acc += b58_match(pk, patterns, lut);
    }
    sink[tid] = acc;
}

kernel void bench_b58_e4(device uint *sink [[buffer(0)]], constant uint &n [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    uchar pk[32];
    for (int i = 0; i < 32; i++) pk[i] = (uchar)(i * 3 + tid);
    uchar digits[64];
    uint acc = 0;
    for (uint i = 0; i < n; i++) {
        pk[0] = (uchar)i;
        acc += b58_digits_e4(pk, digits);
    }
    sink[tid] = acc ^ digits[0];
}

kernel void b58_div_selftest(device uint *out [[buffer(0)]]) {
    ulong n = 0x123456789abcdef0UL;
    out[0] = (div_58_5(n) == (n / 656356768UL)) ? 1u : 0u;
    n = 18446744073709551615UL;
    out[1] = (div_58_5(n) == (n / 656356768UL)) ? 1u : 0u;
}

// ─── SHA-256 grind (CreateAccountWithSeed). Not the ed25519 hot path. ───

#define GR_ROTR(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define GR_CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define GR_MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define GR_EP0(x) (GR_ROTR(x, 2) ^ GR_ROTR(x, 13) ^ GR_ROTR(x, 22))
#define GR_EP1(x) (GR_ROTR(x, 6) ^ GR_ROTR(x, 11) ^ GR_ROTR(x, 25))
#define GR_SIG0(x) (GR_ROTR(x, 7) ^ GR_ROTR(x, 18) ^ ((x) >> 3))
#define GR_SIG1(x) (GR_ROTR(x, 17) ^ GR_ROTR(x, 19) ^ ((x) >> 10))
#define GR_SHA_R(K, M)                          \
    do {                                        \
        uint t1 = h + GR_EP1(e) + GR_CH(e, f, g) + (K) + (M); \
        uint t2 = GR_EP0(a) + GR_MAJ(a, b, c);  \
        h = g; g = f; f = e; e = d + t1;        \
        d = c; c = b; b = a; a = t1 + t2;       \
    } while (0)

static void grind_expand(thread uint *W) {
    for (int i = 16; i < 64; i++)
        W[i] = GR_SIG1(W[i - 2]) + W[i - 7] + GR_SIG0(W[i - 15]) + W[i - 16];
}

static void grind_rounds_from8(thread uint *state, thread uint *W, device const uint *sr7) {
    uint a = sr7[0], b = sr7[1], c = sr7[2], d = sr7[3];
    uint e = sr7[4], f = sr7[5], g = sr7[6], h = sr7[7];
    grind_expand(W);
    GR_SHA_R(0xD807AA98U, W[8]); GR_SHA_R(0x12835B01U, W[9]);
    GR_SHA_R(0x243185BEU, W[10]); GR_SHA_R(0x550C7DC3U, W[11]);
    GR_SHA_R(0x72BE5D74U, W[12]); GR_SHA_R(0x80DEB1FEU, W[13]);
    GR_SHA_R(0x9BDC06A7U, W[14]); GR_SHA_R(0xC19BF174U, W[15]);
    GR_SHA_R(0xE49B69C1U, W[16]); GR_SHA_R(0xEFBE4786U, W[17]);
    GR_SHA_R(0x0FC19DC6U, W[18]); GR_SHA_R(0x240CA1CCU, W[19]);
    GR_SHA_R(0x2DE92C6FU, W[20]); GR_SHA_R(0x4A7484AAU, W[21]);
    GR_SHA_R(0x5CB0A9DCU, W[22]); GR_SHA_R(0x76F988DAU, W[23]);
    GR_SHA_R(0x983E5152U, W[24]); GR_SHA_R(0xA831C66DU, W[25]);
    GR_SHA_R(0xB00327C8U, W[26]); GR_SHA_R(0xBF597FC7U, W[27]);
    GR_SHA_R(0xC6E00BF3U, W[28]); GR_SHA_R(0xD5A79147U, W[29]);
    GR_SHA_R(0x06CA6351U, W[30]); GR_SHA_R(0x14292967U, W[31]);
    GR_SHA_R(0x27B70A85U, W[32]); GR_SHA_R(0x2E1B2138U, W[33]);
    GR_SHA_R(0x4D2C6DFCU, W[34]); GR_SHA_R(0x53380D13U, W[35]);
    GR_SHA_R(0x650A7354U, W[36]); GR_SHA_R(0x766A0ABBU, W[37]);
    GR_SHA_R(0x81C2C92EU, W[38]); GR_SHA_R(0x92722C85U, W[39]);
    GR_SHA_R(0xA2BFE8A1U, W[40]); GR_SHA_R(0xA81A664BU, W[41]);
    GR_SHA_R(0xC24B8B70U, W[42]); GR_SHA_R(0xC76C51A3U, W[43]);
    GR_SHA_R(0xD192E819U, W[44]); GR_SHA_R(0xD6990624U, W[45]);
    GR_SHA_R(0xF40E3585U, W[46]); GR_SHA_R(0x106AA070U, W[47]);
    GR_SHA_R(0x19A4C116U, W[48]); GR_SHA_R(0x1E376C08U, W[49]);
    GR_SHA_R(0x2748774CU, W[50]); GR_SHA_R(0x34B0BCB5U, W[51]);
    GR_SHA_R(0x391C0CB3U, W[52]); GR_SHA_R(0x4ED8AA4AU, W[53]);
    GR_SHA_R(0x5B9CCA4FU, W[54]); GR_SHA_R(0x682E6FF3U, W[55]);
    GR_SHA_R(0x748F82EEU, W[56]); GR_SHA_R(0x78A5636FU, W[57]);
    GR_SHA_R(0x84C87814U, W[58]); GR_SHA_R(0x8CC70208U, W[59]);
    GR_SHA_R(0x90BEFFFAU, W[60]); GR_SHA_R(0xA4506CEBU, W[61]);
    GR_SHA_R(0xBEF9A3F7U, W[62]); GR_SHA_R(0xC67178F2U, W[63]);
    state[0] = 0x6a09e667U + a; state[1] = 0xbb67ae85U + b;
    state[2] = 0x3c6ef372U + c; state[3] = 0xa54ff53aU + d;
    state[4] = 0x510e527fU + e; state[5] = 0x9b05688cU + f;
    state[6] = 0x1f83d9abU + g; state[7] = 0x5be0cd19U + h;
}

static void grind_block1(thread uint *state, device const uint *W1) {
    uint W[64];
    for (int i = 0; i < 64; i++) W[i] = W1[i];
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];
    GR_SHA_R(0x428A2F98U, W[0]); GR_SHA_R(0x71374491U, W[1]);
    GR_SHA_R(0xB5C0FBCFU, W[2]); GR_SHA_R(0xE9B5DBA5U, W[3]);
    GR_SHA_R(0x3956C25BU, W[4]); GR_SHA_R(0x59F111F1U, W[5]);
    GR_SHA_R(0x923F82A4U, W[6]); GR_SHA_R(0xAB1C5ED5U, W[7]);
    GR_SHA_R(0xD807AA98U, W[8]); GR_SHA_R(0x12835B01U, W[9]);
    GR_SHA_R(0x243185BEU, W[10]); GR_SHA_R(0x550C7DC3U, W[11]);
    GR_SHA_R(0x72BE5D74U, W[12]); GR_SHA_R(0x80DEB1FEU, W[13]);
    GR_SHA_R(0x9BDC06A7U, W[14]); GR_SHA_R(0xC19BF174U, W[15]);
    GR_SHA_R(0xE49B69C1U, W[16]); GR_SHA_R(0xEFBE4786U, W[17]);
    GR_SHA_R(0x0FC19DC6U, W[18]); GR_SHA_R(0x240CA1CCU, W[19]);
    GR_SHA_R(0x2DE92C6FU, W[20]); GR_SHA_R(0x4A7484AAU, W[21]);
    GR_SHA_R(0x5CB0A9DCU, W[22]); GR_SHA_R(0x76F988DAU, W[23]);
    GR_SHA_R(0x983E5152U, W[24]); GR_SHA_R(0xA831C66DU, W[25]);
    GR_SHA_R(0xB00327C8U, W[26]); GR_SHA_R(0xBF597FC7U, W[27]);
    GR_SHA_R(0xC6E00BF3U, W[28]); GR_SHA_R(0xD5A79147U, W[29]);
    GR_SHA_R(0x06CA6351U, W[30]); GR_SHA_R(0x14292967U, W[31]);
    GR_SHA_R(0x27B70A85U, W[32]); GR_SHA_R(0x2E1B2138U, W[33]);
    GR_SHA_R(0x4D2C6DFCU, W[34]); GR_SHA_R(0x53380D13U, W[35]);
    GR_SHA_R(0x650A7354U, W[36]); GR_SHA_R(0x766A0ABBU, W[37]);
    GR_SHA_R(0x81C2C92EU, W[38]); GR_SHA_R(0x92722C85U, W[39]);
    GR_SHA_R(0xA2BFE8A1U, W[40]); GR_SHA_R(0xA81A664BU, W[41]);
    GR_SHA_R(0xC24B8B70U, W[42]); GR_SHA_R(0xC76C51A3U, W[43]);
    GR_SHA_R(0xD192E819U, W[44]); GR_SHA_R(0xD6990624U, W[45]);
    GR_SHA_R(0xF40E3585U, W[46]); GR_SHA_R(0x106AA070U, W[47]);
    GR_SHA_R(0x19A4C116U, W[48]); GR_SHA_R(0x1E376C08U, W[49]);
    GR_SHA_R(0x2748774CU, W[50]); GR_SHA_R(0x34B0BCB5U, W[51]);
    GR_SHA_R(0x391C0CB3U, W[52]); GR_SHA_R(0x4ED8AA4AU, W[53]);
    GR_SHA_R(0x5B9CCA4FU, W[54]); GR_SHA_R(0x682E6FF3U, W[55]);
    GR_SHA_R(0x748F82EEU, W[56]); GR_SHA_R(0x78A5636FU, W[57]);
    GR_SHA_R(0x84C87814U, W[58]); GR_SHA_R(0x8CC70208U, W[59]);
    GR_SHA_R(0x90BEFFFAU, W[60]); GR_SHA_R(0xA4506CEBU, W[61]);
    GR_SHA_R(0xBEF9A3F7U, W[62]); GR_SHA_R(0xC67178F2U, W[63]);
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

kernel void vanity_search(
    device const uchar *seed [[buffer(0)]],
    device const uint *W0_fixed [[buffer(1)]],
    device const uint *state_r7 [[buffer(2)]],
    device const uint *W1 [[buffer(3)]],
    device const uchar *glyph [[buffer(4)]],
    device const uchar *lut [[buffer(5)]],
    device const uchar *patterns [[buffer(6)]],
    device uchar *out [[buffer(7)]],
    device atomic_uint *done [[buffer(8)]],
    device uint *counts [[buffer(9)]],
    constant uint &max_iters [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
    ulong idx = tid;
    uint digest[8];
    {
        uchar ls[32];
        for (int i = 0; i < 32; i++) ls[i] = seed[i];
        for (int lane = 0; lane < 4; lane++) {
            ulong v = 0;
            for (int b = 0; b < 8; b++) v |= ((ulong)ls[lane * 8 + b]) << (8 * b);
            v += idx;
            for (int b = 0; b < 8; b++) ls[lane * 8 + b] = (uchar)(v >> (8 * b));
        }
        for (int i = 0; i < 8; i++)
            digest[i] = ((uint)ls[4 * i] << 24) | ((uint)ls[4 * i + 1] << 16) |
                        ((uint)ls[4 * i + 2] << 8) | (uint)ls[4 * i + 3];
    }
    uint iter = 0;
    for (; iter < max_iters; iter++) {
        if ((iter & 0x3fffu) == 0 && atomic_load_explicit(done, memory_order_relaxed)) break;
        uint seed_words[4];
        for (int k = 0; k < 4; k++) {
            uint s = digest[k];
            seed_words[k] = ((uint)glyph[(s >> 24) & 0xffu] << 24) |
                            ((uint)glyph[(s >> 16) & 0xffu] << 16) |
                            ((uint)glyph[(s >> 8) & 0xffu] << 8) |
                            ((uint)glyph[s & 0xffu]);
        }
        uint W0[64];
        for (int i = 0; i < 16; i++) W0[i] = W0_fixed[i];
        W0[8] = seed_words[0];
        W0[9] = seed_words[1];
        W0[10] = seed_words[2];
        W0[11] = seed_words[3];
        grind_rounds_from8(digest, W0, state_r7);
        grind_block1(digest, W1);

        uchar fake_pk[32];
        for (int k = 0; k < 8; k++) {
            fake_pk[4 * k] = (uchar)(digest[k] >> 24);
            fake_pk[4 * k + 1] = (uchar)(digest[k] >> 16);
            fake_pk[4 * k + 2] = (uchar)(digest[k] >> 8);
            fake_pk[4 * k + 3] = (uchar)digest[k];
        }
        // b58_match re-packs bytes into words. Feed the digest words directly
        // by packing them back — same bytes, so the match sees digest_words.
        if (b58_match(fake_pk, patterns, lut)) {
            uint expected = 0;
            if (atomic_compare_exchange_weak_explicit(done, &expected, 1u, memory_order_relaxed,
                                                      memory_order_relaxed)) {
                for (int k = 0; k < 4; k++) {
                    uint w = seed_words[k];
                    out[4 * k] = (uchar)(w >> 24);
                    out[4 * k + 1] = (uchar)(w >> 16);
                    out[4 * k + 2] = (uchar)(w >> 8);
                    out[4 * k + 3] = (uchar)w;
                }
            }
            iter++;
            break;
        }
    }
    counts[tid] = iter;
}
