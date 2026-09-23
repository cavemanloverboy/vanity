// A/B: inline, narrow field mul, 128-byte AoS records, prefetch, register cap.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static NSString *g_field, *g_rest;

typedef struct Cfg {
    const char *name;
    int w, batch, dual, layout, prefetch, narrow, reg;
    const char *attr; // FE_ATTR text, or NULL
} Cfg;

static NSString *make_src(Cfg c) {
    NSString *attr = c.attr ? [NSString stringWithUTF8String:c.attr] : @"#define FE_ATTR";
    return [NSString stringWithFormat:
        @"#define COMB_W %d\n#define KP_BATCH %d\n#define USE_TG 0\n#define DUAL %d\n"
        @"#define BS58_MAGIC 1\n#define LAYOUT %d\n#define PREFETCH %d\n#define FE_NARROW %d\n"
        @"#define REG_CAP %d\n%@\n%@\n%@",
        c.w, c.batch, c.dual, c.layout, c.prefetch, c.narrow, c.reg, attr, g_field, g_rest];
}

static BOOL dispatch_bufs(id<MTLCommandQueue> q, id<MTLComputePipelineState> p, NSArray *bufs,
                          NSUInteger groups, NSUInteger tg, double *secs) {
    NSUInteger t = MIN(tg, p.maxTotalThreadsPerThreadgroup);
    if (t < 1) return NO;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    for (NSUInteger i = 0; i < bufs.count; i++) [enc setBuffer:bufs[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
    [enc endEncoding];
    NSDate *t0 = [NSDate date];
    [cb commit];
    [cb waitUntilCompleted];
    double wall = -[t0 timeIntervalSinceNow];
    double gpu = cb.GPUEndTime - cb.GPUStartTime;
    *secs = (gpu > 1e-4) ? gpu : wall;
    return cb.status == MTLCommandBufferStatusCompleted && *secs > 0;
}

int main(void) {
    @autoreleasepool {
        NSString *dir = @"/Users/cavey/cavey/vanity/kernels/metal";
        g_field = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"field.metal"] encoding:NSUTF8StringEncoding error:nil];
        g_rest = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"rest.metal"] encoding:NSUTF8StringEncoding error:nil];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const uint8_t seed[32] = {
            0x9d,0x61,0xb1,0x9d,0xef,0xfd,0x5a,0x60,0xba,0x84,0x4a,0xf4,0x92,0xec,0x2c,0xc4,
            0x44,0x49,0xc5,0x69,0x7b,0x32,0x69,0x19,0x70,0x3b,0xac,0x03,0x1c,0xae,0x7f,0x60};
        const uint8_t expect[32] = {
            0xd7,0x5a,0x98,0x01,0x82,0xb1,0x0a,0xb7,0xd5,0x4b,0xfe,0xd3,0xc9,0x64,0x07,0x3a,
            0x0e,0xe1,0x72,0xf3,0xda,0xa6,0x23,0x25,0xaf,0x02,0x1a,0x68,0xf7,0x07,0x51,0x1a};

        Cfg cfgs[] = {
            {"old sha, check often", 12, 4, 1, 1, 0, 0, 0,
             "#define FE_ATTR __attribute__((noinline))\n#define SHA_FOLD 0\n#define SHA_ATTR\n#define STOP_EVERY 8\n"},
            {"sha outlined", 12, 4, 1, 1, 0, 0, 0,
             "#define FE_ATTR __attribute__((noinline))\n#define SHA_FOLD 0\n#define SHA_ATTR __attribute__((noinline))\n#define STOP_EVERY 8\n"},
            {"sha outlined+fold", 12, 4, 1, 1, 0, 0, 0,
             "#define FE_ATTR __attribute__((noinline))\n#define SHA_FOLD 1\n#define SHA_ATTR __attribute__((noinline))\n#define STOP_EVERY 8\n"},
            {"sha outlined+fold+rare check", 12, 4, 1, 1, 0, 0, 0,
             "#define FE_ATTR __attribute__((noinline))\n#define SHA_FOLD 1\n#define SHA_ATTR __attribute__((noinline))\n#define STOP_EVERY 64\n"},
        };
        int n = (int)(sizeof cfgs / sizeof cfgs[0]);
        for (int ci = 0; ci < n; ci++) {
            Cfg c = cfgs[ci];
            NSError *err = nil;
            NSDate *t0 = [NSDate date];
            id<MTLLibrary> lib = [dev newLibraryWithSource:make_src(c) options:nil error:&err];
            if (!lib) {
                printf("%-24s COMPILE FAIL\n", c.name);
                fflush(stdout);
                continue;
            }
            int pos = 1 << (c.w - 1);
            int windows = (256 + c.w - 1) / c.w;
            int table = windows * pos;
            int stride = c.layout ? 32 : 30;
            id<MTLBuffer> scratch = [dev newBufferWithLength:(NSUInteger)table * 40 * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> comb = [dev newBufferWithLength:(NSUInteger)table * stride * 4 options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> pb = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"build_comb"] error:&err];
            id<MTLComputePipelineState> pn = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"normalize_comb"] error:&err];
            id<MTLComputePipelineState> pk = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kat_pubkeys"] error:&err];
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"vanity_doppler_search"] error:&err];
            if (!pb || !pn || !pk || !ps) {
                printf("%-24s PIPE FAIL %s\n", c.name, err.localizedDescription.UTF8String);
                fflush(stdout);
                continue;
            }
            double secs = 0;
            dispatch_bufs(q, pb, @[scratch], 1, 1, &secs);
            NSUInteger nt = MIN((NSUInteger)256, pn.maxTotalThreadsPerThreadgroup);
            dispatch_bufs(q, pn, @[scratch, comb], (table + nt - 1) / nt, nt, &secs);

            id<MTLBuffer> seeds = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
            id<MTLBuffer> pubs = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
            id<MTLBuffer> nbuf = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
            memcpy(seeds.contents, seed, 32);
            uint32_t one = 1; memcpy(nbuf.contents, &one, 4);
            NSUInteger ktg = MIN((NSUInteger)128, pk.maxTotalThreadsPerThreadgroup);
            if (!dispatch_bufs(q, pk, @[comb, seeds, pubs, nbuf], 1, ktg, &secs) ||
                memcmp(pubs.contents, expect, 32) != 0) {
                printf("%-24s KAT FAIL\n", c.name);
                fflush(stdout);
                continue;
            }

            int kp = c.batch * (c.dual ? 2 : 1);
            uint32_t iters = 128;
            if (iters % (uint32_t)kp) iters += (uint32_t)kp - iters % (uint32_t)kp;
            NSUInteger tgs[] = {128};
            NSUInteger grids[] = {65536};
            for (int ti = 0; ti < 1; ti++) {
                NSUInteger tg = MIN(tgs[ti], ps.maxTotalThreadsPerThreadgroup);
                for (int gi = 0; gi < 1; gi++) {
                    NSUInteger threads = grids[gi];
                    NSUInteger groups = threads / tg;
                    if (groups < 1) continue;
                    threads = groups * tg;
                    id<MTLBuffer> host = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
                    arc4random_buf(host.contents, 32);
                    id<MTLBuffer> outb = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
                    id<MTLBuffer> done = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
                    id<MTLBuffer> counts = [dev newBufferWithLength:threads * 4 options:MTLResourceStorageModeShared];
                    id<MTLBuffer> req = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
                    id<MTLBuffer> it = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
                    uint32_t five = 5; memcpy(req.contents, &five, 4);
                    memcpy(it.contents, &iters, 4);
                    NSArray *bufs = @[host, comb, outb, done, counts, req, it];
                    dispatch_bufs(q, ps, bufs, groups, tg, &secs);
                    memset(done.contents, 0, 4);
                    if (!dispatch_bufs(q, ps, bufs, groups, tg, &secs)) {
                        printf("tg %lu grid %lu RUN FAIL\n", (unsigned long)tg, (unsigned long)threads);
                        continue;
                    }
                    uint64_t total = 0;
                    uint32_t *cts = (uint32_t *)counts.contents;
                    for (NSUInteger i = 0; i < threads; i++) total += cts[i];
                    printf("tg %4lu threads %7lu  %6.2f Mkeys/s\n",
                           (unsigned long)tg, (unsigned long)threads, (double)total / secs / 1e6);
                    fflush(stdout);
                }
            }
        }
    }
    return 0;
}
