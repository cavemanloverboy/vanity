// Metal backend. Same extern "C" ABI as the CUDA and OpenCL hosts.
//
// Frozen on an M5 Pro (20 GPU cores) after A/B of comb width, batch, dual
// chains, inline, narrow field muls, register caps, and table layout:
// width-12 affine comb, 128-byte AoS records (one cache line per entry),
// 4-key batch x 2 independent SHA-512 chains. fe_mul/fe_sq and sha512_32
// stay outlined so their temps do not spill the comb. The 32-byte SHA-512
// schedule drops the zero padding words. The stop flag is sampled every
// 64 keys. ~37M keys/s on an M5 Pro (20 GPU cores).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>

#include "metal_sources.h"
#include "pattern_table.h"

#define COMB_W 12
#define KP_BATCH 4
#define DUAL 1
#define LAYOUT 1
#define KEYS_PER (KP_BATCH * (DUAL ? 2 : 1))
#define COMB_POS (1 << (COMB_W - 1))
#define COMB_WINDOWS ((256 + COMB_W - 1) / COMB_W)
#define COMB_TABLE_LEN (COMB_WINDOWS * COMB_POS)
#define COMB_STRIDE 32

#define TARGET_SEC 0.75

static id<MTLDevice> g_dev;
static id<MTLLibrary> g_lib;
static id<MTLBuffer> g_comb;
static id<MTLComputePipelineState> g_kp, g_dop, g_grind, g_kat, g_build, g_norm;

@interface GpuCtx : NSObject
@property (strong) id<MTLCommandQueue> queue;
@property (strong) id<MTLCommandBuffer> inflight;
@property (strong) id<MTLComputePipelineState> pipe;
@property (strong) id<MTLBuffer> seed;
@property (strong) id<MTLBuffer> lut;
@property (strong) id<MTLBuffer> patterns;
@property (strong) id<MTLBuffer> w0;
@property (strong) id<MTLBuffer> sr7;
@property (strong) id<MTLBuffer> w1;
@property (strong) id<MTLBuffer> glyph;
@property (strong) id<MTLBuffer> outb;
@property (strong) id<MTLBuffer> doneb;
@property (strong) id<MTLBuffer> counts;
@property (strong) id<MTLBuffer> iters;
@property (strong) id<MTLBuffer> required;
@property uint32_t *countsHost;
@property NSUInteger threads;
@property NSUInteger groups;
@property NSUInteger tg;
@property uint32_t maxIters;
@property uint32_t outBytes;
@property double launchTime;
@property int kind; // 0 grind, 1 keypair, 2 doppler
@end

@implementation GpuCtx
- (void)dealloc {
    free(_countsHost);
    _countsHost = NULL;
}
@end

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *what, NSError *err) {
    if (err)
        fprintf(stderr, "metal: %s: %s\n", what, err.localizedDescription.UTF8String);
    else
        fprintf(stderr, "metal: %s\n", what);
    exit(EXIT_FAILURE);
}

static id<MTLBuffer> make_buf(id<MTLDevice> dev, NSUInteger n) {
    id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (!b) die("buffer alloc", nil);
    return b;
}

static id<MTLComputePipelineState> make_pipe(NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [g_lib newFunctionWithName:name];
    if (!fn) die("missing kernel", nil);
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) die(name.UTF8String, err);
    return p;
}

static NSString *shader_source(void) {
    NSString *field = [NSString stringWithUTF8String:METAL_FIELD];
    NSString *rest = [NSString stringWithUTF8String:METAL_REST];
    return [NSString stringWithFormat:
        @"#define COMB_W %d\n#define KP_BATCH %d\n#define DUAL %d\n#define USE_TG 0\n"
        @"#define LAYOUT %d\n#define PREFETCH 0\n#define FE_NARROW 0\n#define BS58_MAGIC 1\n"
        @"#define REG_CAP 0\n#define FE_ATTR __attribute__((noinline))\n%@\n%@",
        COMB_W, KP_BATCH, DUAL, LAYOUT, field, rest];
}

static void build_table(id<MTLCommandQueue> queue) {
    id<MTLBuffer> scratch = make_buf(g_dev, (NSUInteger)COMB_TABLE_LEN * 40 * 4);
    g_comb = make_buf(g_dev, (NSUInteger)COMB_TABLE_LEN * COMB_STRIDE * 4);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_build];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_norm];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:g_comb offset:0 atIndex:1];
    NSUInteger tg = MIN((NSUInteger)256, g_norm.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = ((NSUInteger)COMB_TABLE_LEN + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) die("comb build", cb.error);
}

static void ensure_device(int device_index) {
    @autoreleasepool {
    id<MTLDevice> dev = nil;
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
    if (device_index >= 0 && device_index < (int)all.count) dev = all[device_index];
    if (!dev) dev = MTLCreateSystemDefaultDevice();
    if (!dev) die("no Metal device", nil);
    if (g_lib && g_dev.registryID == dev.registryID) return;

    g_dev = dev;
    g_lib = nil;
    g_comb = nil;
    NSError *err = nil;
    g_lib = [g_dev newLibraryWithSource:shader_source() options:nil error:&err];
    if (!g_lib) die("shader compile", err);
    g_kp = make_pipe(@"vanity_keypair_search");
    g_dop = make_pipe(@"vanity_doppler_search");
    g_grind = make_pipe(@"vanity_search");
    g_kat = make_pipe(@"kat_pubkeys");
    g_build = make_pipe(@"build_comb");
    g_norm = make_pipe(@"normalize_comb");
    id<MTLCommandQueue> q = [g_dev newCommandQueue];
    build_table(q);
    }
}

static uint32_t adapt_iters(uint32_t cur, double elapsed, uint32_t lo, uint32_t hi, uint32_t align) {
    if (elapsed <= 1e-6) return hi;
    double factor = TARGET_SEC / elapsed;
    if (factor < 0.25) factor = 0.25;
    if (factor > 4.0) factor = 4.0;
    double next = (double)cur * factor;
    if (next < lo) next = lo;
    if (next > hi) next = hi;
    uint32_t n = (uint32_t)next;
    if (align > 1) {
        uint32_t r = n % align;
        if (r) n += align - r;
        if (n > hi) n = hi - (hi % align);
        if (n < lo) n = lo;
    }
    return n;
}

static void size_launch(GpuCtx *c) {
    NSUInteger cap = c.pipe.maxTotalThreadsPerThreadgroup;
    // 128-wide groups and 64K threads won the launch-shape sweep on M5 Pro.
    // The SHA-256 grind stays at 256, which matches the CUDA/OpenCL launch.
    NSUInteger want = (c.kind == 0) ? 256 : 128;
    c.tg = MIN(want, cap);
    if (c.tg < 1) c.tg = 1;
    c.groups = 65536 / c.tg;
    if (c.groups < 1) c.groups = 1;
    c.threads = c.groups * c.tg;
}

static uint8_t *match_lut(bool case_insensitive) {
    static const char alphabet_normal[59] =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    static const char alphabet_ci[59] =
        "123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";
    const char *alphabet = case_insensitive ? alphabet_ci : alphabet_normal;
    uint8_t *lut = (uint8_t *)malloc(58);
    for (int i = 0; i < 58; i++) {
        lut[i] = (uint8_t)i;
        for (int j = 0; j < i; j++)
            if (alphabet[j] == alphabet[i]) { lut[i] = (uint8_t)j; break; }
    }
    return lut;
}

#define ROTR(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define H_CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define H_MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define H_EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define H_EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define H_SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define H_SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))
#define H_SHA_R(K, M)                                          \
    do {                                                       \
        uint32_t t1 = h + H_EP1(e) + H_CH(e, f, g) + (K) + (M); \
        uint32_t t2 = H_EP0(a) + H_MAJ(a, b, c);               \
        h = g; g = f; f = e; e = d + t1;                       \
        d = c; c = b; b = a; a = t1 + t2;                      \
    } while (0)

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

// First 8 SHA-256 rounds of block 0. `base` is 32 bytes. Isolated so the
// working-state name `c` does not collide with a GpuCtx pointer.
static void sha256_rounds0_7(const uint8_t *base, uint32_t out[8]) {
    uint32_t W[8];
    for (int i = 0; i < 8; i++) W[i] = be32(base + 4 * i);
    uint32_t a = 0x6a09e667U, b = 0xbb67ae85U, c = 0x3c6ef372U, d = 0xa54ff53aU;
    uint32_t e = 0x510e527fU, f = 0x9b05688cU, g = 0x1f83d9abU, h = 0x5be0cd19U;
    H_SHA_R(0x428A2F98U, W[0]); H_SHA_R(0x71374491U, W[1]);
    H_SHA_R(0xB5C0FBCFU, W[2]); H_SHA_R(0xE9B5DBA5U, W[3]);
    H_SHA_R(0x3956C25BU, W[4]); H_SHA_R(0x59F111F1U, W[5]);
    H_SHA_R(0x923F82A4U, W[6]); H_SHA_R(0xAB1C5ED5U, W[7]);
    out[0] = a; out[1] = b; out[2] = c; out[3] = d;
    out[4] = e; out[5] = f; out[6] = g; out[7] = h;
}

static GpuCtx *new_ctx(int device_index, id<MTLComputePipelineState> pipe, uint32_t iters, uint32_t out_bytes, int kind) {
    ensure_device(device_index);
    GpuCtx *c = [GpuCtx new];
    c.queue = [g_dev newCommandQueue];
    c.pipe = pipe;
    c.kind = kind;
    c.maxIters = iters;
    c.outBytes = out_bytes;
    c.seed = make_buf(g_dev, 32);
    c.outb = make_buf(g_dev, out_bytes);
    c.doneb = make_buf(g_dev, 4);
    c.iters = make_buf(g_dev, 4);
    size_launch(c);
    c.counts = make_buf(g_dev, c.threads * 4);
    c.countsHost = (uint32_t *)calloc(c.threads, 4);
    return c;
}

static void launch_common(GpuCtx *c, const uint8_t *seed) {
    @autoreleasepool {
    if (c.inflight) {
        [c.inflight waitUntilCompleted];
        c.inflight = nil;
    }
    memcpy(c.seed.contents, seed, 32);
    memset(c.outb.contents, 0, c.outBytes);
    memset(c.doneb.contents, 0, 4);
    uint32_t iters_now = c.maxIters;
    memcpy(c.iters.contents, &iters_now, 4);

    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:c.pipe];
    [enc setBuffer:c.seed offset:0 atIndex:0];
    if (c.kind == 0) {
        [enc setBuffer:c.w0 offset:0 atIndex:1];
        [enc setBuffer:c.sr7 offset:0 atIndex:2];
        [enc setBuffer:c.w1 offset:0 atIndex:3];
        [enc setBuffer:c.glyph offset:0 atIndex:4];
        [enc setBuffer:c.lut offset:0 atIndex:5];
        [enc setBuffer:c.patterns offset:0 atIndex:6];
        [enc setBuffer:c.outb offset:0 atIndex:7];
        [enc setBuffer:c.doneb offset:0 atIndex:8];
        [enc setBuffer:c.counts offset:0 atIndex:9];
        [enc setBuffer:c.iters offset:0 atIndex:10];
    } else if (c.kind == 1) {
        [enc setBuffer:g_comb offset:0 atIndex:1];
        [enc setBuffer:c.lut offset:0 atIndex:2];
        [enc setBuffer:c.patterns offset:0 atIndex:3];
        [enc setBuffer:c.outb offset:0 atIndex:4];
        [enc setBuffer:c.doneb offset:0 atIndex:5];
        [enc setBuffer:c.counts offset:0 atIndex:6];
        [enc setBuffer:c.iters offset:0 atIndex:7];
    } else {
        [enc setBuffer:g_comb offset:0 atIndex:1];
        [enc setBuffer:c.outb offset:0 atIndex:2];
        [enc setBuffer:c.doneb offset:0 atIndex:3];
        [enc setBuffer:c.counts offset:0 atIndex:4];
        [enc setBuffer:c.required offset:0 atIndex:5];
        [enc setBuffer:c.iters offset:0 atIndex:6];
    }
    [enc dispatchThreadgroups:MTLSizeMake(c.groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(c.tg, 1, 1)];
    [enc endEncoding];
    c.launchTime = now_sec();
    [cb commit];
    c.inflight = cb;
    }
}

static int query_common(GpuCtx *c) {
    if (!c.inflight) return 1;
    MTLCommandBufferStatus st = c.inflight.status;
    return (st == MTLCommandBufferStatusCompleted || st == MTLCommandBufferStatusError) ? 1 : 0;
}

static void read_common(GpuCtx *c, uint8_t *out) {
    if (c.inflight) {
        [c.inflight waitUntilCompleted];
        if (c.inflight.status != MTLCommandBufferStatusCompleted)
            die("kernel", c.inflight.error);
    }
    double elapsed = now_sec() - c.launchTime;
    memcpy(out, c.outb.contents, c.outBytes);
    memcpy(c.countsHost, c.counts.contents, c.threads * 4);
    uint64_t total = 0;
    for (NSUInteger i = 0; i < c.threads; i++) total += c.countsHost[i];
    memcpy(out + c.outBytes, &total, 8);
    c.inflight = nil;
    if (c.kind == 0)
        c.maxIters = adapt_iters(c.maxIters, elapsed, 256, 1u << 20, 1);
    else
        c.maxIters = adapt_iters(c.maxIters, elapsed, KEYS_PER, 1u << 16, KEYS_PER);
}

static void destroy_common(void *p) {
    GpuCtx *c = (__bridge_transfer GpuCtx *)p;
    if (c.inflight) [c.inflight waitUntilCompleted];
    c.inflight = nil;
}

// ─── grind ────────────────────────────────────────────────────────────────

void *gpu_grind_init(int id, uint8_t *base, uint8_t *owner, uint8_t *patterns,
                     uint64_t patterns_len, bool case_insensitive) {
    @autoreleasepool {
    ensure_device(id);
    GpuCtx *c = new_ctx(id, g_grind, 4096, 16, 0);

    uint32_t W1[64];
    uint8_t b1[64] = {0};
    memcpy(b1, owner + 16, 16);
    b1[16] = 0x80;
    b1[62] = 0x02;
    b1[63] = 0x80;
    for (int i = 0; i < 16; i++) W1[i] = be32(b1 + 4 * i);
    for (int i = 16; i < 64; i++)
        W1[i] = H_SIG1(W1[i - 2]) + W1[i - 7] + H_SIG0(W1[i - 15]) + W1[i - 16];

    uint32_t W0[16] = {0};
    for (int i = 0; i < 8; i++) W0[i] = be32(base + 4 * i);
    for (int i = 0; i < 4; i++) W0[12 + i] = be32(owner + 4 * i);

    uint32_t SR7[8];
    sha256_rounds0_7(base, SR7);

    uint8_t glyph[256];
    static const char alnum[63] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (unsigned u = 0; u < 256; u++) glyph[u] = (uint8_t)alnum[u % 62];

    uint8_t ptable[VANITY_PT_SIZE];
    vanity_unpack_patterns(patterns, patterns_len, ptable);
    uint8_t *lut = match_lut(case_insensitive);

    c.w0 = make_buf(g_dev, sizeof W0);
    c.sr7 = make_buf(g_dev, sizeof SR7);
    c.w1 = make_buf(g_dev, sizeof W1);
    c.glyph = make_buf(g_dev, sizeof glyph);
    c.lut = make_buf(g_dev, 58);
    c.patterns = make_buf(g_dev, VANITY_PT_SIZE);
    memcpy(c.w0.contents, W0, sizeof W0);
    memcpy(c.sr7.contents, SR7, sizeof SR7);
    memcpy(c.w1.contents, W1, sizeof W1);
    memcpy(c.glyph.contents, glyph, sizeof glyph);
    memcpy(c.lut.contents, lut, 58);
    memcpy(c.patterns.contents, ptable, VANITY_PT_SIZE);
    free(lut);
    return (__bridge_retained void *)c;
    }
}

void gpu_grind_launch(void *p, uint8_t *seed) { launch_common((__bridge GpuCtx *)p, seed); }
int gpu_grind_query(void *p) { return query_common((__bridge GpuCtx *)p); }
void gpu_grind_read(void *p, uint8_t *out) { read_common((__bridge GpuCtx *)p, out); }
void gpu_grind_destroy(void *p) { destroy_common(p); }

void gpu_grind_set_active_mask(void *p, unsigned long long mask) {
    GpuCtx *c = (__bridge GpuCtx *)p;
    memcpy((uint8_t *)c.patterns.contents + VANITY_PT_ACTIVE, &mask, 8);
}

// ─── keypair ──────────────────────────────────────────────────────────────

static void upload_patterns(GpuCtx *c, const uint8_t *patterns, uint64_t patterns_len, bool ci) {
    uint8_t ptable[VANITY_PT_SIZE];
    vanity_unpack_patterns(patterns, patterns_len, ptable);
    uint8_t *lut = match_lut(ci);
    c.lut = make_buf(g_dev, 58);
    c.patterns = make_buf(g_dev, VANITY_PT_SIZE);
    memcpy(c.lut.contents, lut, 58);
    memcpy(c.patterns.contents, ptable, VANITY_PT_SIZE);
    free(lut);
}

void *gpu_keypair_init(int id, uint8_t *patterns, uint64_t patterns_len, bool case_insensitive) {
    @autoreleasepool {
    ensure_device(id);
    GpuCtx *c = new_ctx(id, g_kp, 64, 32, 1);
    if (c.maxIters % KEYS_PER) c.maxIters += KEYS_PER - (c.maxIters % KEYS_PER);
    upload_patterns(c, patterns, patterns_len, case_insensitive);
    return (__bridge_retained void *)c;
    }
}

void gpu_keypair_launch(void *p, uint8_t *seed) { launch_common((__bridge GpuCtx *)p, seed); }
int gpu_keypair_query(void *p) { return query_common((__bridge GpuCtx *)p); }
void gpu_keypair_read(void *p, uint8_t *out) { read_common((__bridge GpuCtx *)p, out); }
void gpu_keypair_destroy(void *p) { destroy_common(p); }
void gpu_keypair_set_active_mask(void *p, unsigned long long mask) {
    gpu_grind_set_active_mask(p, mask);
}

// ─── doppler ──────────────────────────────────────────────────────────────

void *gpu_doppler_init(int id, uint32_t required_segments) {
    @autoreleasepool {
    ensure_device(id);
    GpuCtx *c = new_ctx(id, g_dop, 64, 32, 2);
    if (c.maxIters % KEYS_PER) c.maxIters += KEYS_PER - (c.maxIters % KEYS_PER);
    c.required = make_buf(g_dev, 4);
    memcpy(c.required.contents, &required_segments, 4);
    return (__bridge_retained void *)c;
    }
}

void gpu_doppler_launch(void *p, uint8_t *seed) { launch_common((__bridge GpuCtx *)p, seed); }
int gpu_doppler_query(void *p) { return query_common((__bridge GpuCtx *)p); }
void gpu_doppler_read(void *p, uint8_t *out) { read_common((__bridge GpuCtx *)p, out); }
void gpu_doppler_destroy(void *p) { destroy_common(p); }

// Test hook: ed25519 pubkey for one raw seed. 0 on success.
int vanity_metal_pubkey(const uint8_t *seed, uint8_t *out) {
    @autoreleasepool {
    ensure_device(0);
    id<MTLCommandQueue> q = [g_dev newCommandQueue];
    id<MTLBuffer> seeds = make_buf(g_dev, 32);
    id<MTLBuffer> pubs = make_buf(g_dev, 32);
    id<MTLBuffer> nbuf = make_buf(g_dev, 4);
    memcpy(seeds.contents, seed, 32);
    uint32_t one = 1;
    memcpy(nbuf.contents, &one, 4);
    NSUInteger tg = MIN((NSUInteger)128, g_kat.maxTotalThreadsPerThreadgroup);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_kat];
    [enc setBuffer:g_comb offset:0 atIndex:0];
    [enc setBuffer:seeds offset:0 atIndex:1];
    [enc setBuffer:pubs offset:0 atIndex:2];
    [enc setBuffer:nbuf offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return 1;
    memcpy(out, pubs.contents, 32);
    return 0;
    }
}
