// Microbench for the Metal ed25519 grinder. Prints keys/s and op rates.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static NSString *g_field;
static NSString *g_rest;

static NSString *src_with(int comb_w, int batch, int use_tg, int dual, int magic) {
    return [NSString stringWithFormat:
        @"#define COMB_W %d\n#define KP_BATCH %d\n#define USE_TG %d\n#define DUAL %d\n#define BS58_MAGIC %d\n%@\n%@",
        comb_w, batch, use_tg, dual, magic, g_field, g_rest];
}

static int keys_per(int batch, int dual) { return batch * (dual ? 2 : 1); }

static id<MTLLibrary> compile(id<MTLDevice> dev, NSString *src, NSString **errOut) {
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib && errOut) *errOut = err.localizedDescription;
    return lib;
}

static BOOL dispatch_bufs(id<MTLCommandQueue> q, id<MTLComputePipelineState> p,
                          NSArray *bufs, NSUInteger groups, NSUInteger tg, double *secs) {
    NSUInteger t = MIN(tg, p.maxTotalThreadsPerThreadgroup);
    if (t < 1) t = 1;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    for (NSUInteger i = 0; i < bufs.count; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
    [enc endEncoding];
    NSDate *t0 = [NSDate date];
    [cb commit];
    [cb waitUntilCompleted];
    *secs = -[t0 timeIntervalSinceNow];
    return cb.status == MTLCommandBufferStatusCompleted;
}

static void build_table(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib,
                        int comb_w, id<MTLBuffer> *combOut) {
    int pos = 1 << (comb_w - 1);
    int windows = (256 + comb_w - 1) / comb_w;
    int table = windows * pos;
    id<MTLBuffer> scratch = [dev newBufferWithLength:(NSUInteger)table * 40 * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> comb = [dev newBufferWithLength:(NSUInteger)table * 30 * 4 options:MTLResourceStorageModeShared];
    NSError *err = nil;
    id<MTLComputePipelineState> pb = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"build_comb"] error:&err];
    id<MTLComputePipelineState> pn = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"normalize_comb"] error:&err];
    if (!pb || !pn) { NSLog(@"table pipe %@", err); exit(1); }
    double s;
    dispatch_bufs(q, pb, @[scratch], 1, 1, &s);
    NSUInteger t = MIN((NSUInteger)256, pn.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = ((NSUInteger)table + t - 1) / t;
    dispatch_bufs(q, pn, @[scratch, comb], groups, t, &s);
    *combOut = comb;
}

// Full keygen rate via doppler with required=5 (never matches).
static double keygen_rate(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib,
                          id<MTLBuffer> comb, int batch, int dual, NSUInteger tg, NSUInteger groups,
                          uint32_t iters) {
    int kp = keys_per(batch, dual);
    if (iters % (uint32_t)kp) iters += (uint32_t)kp - (iters % (uint32_t)kp);
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"vanity_doppler_search"] error:&err];
    if (!p) { NSLog(@"pipe %@", err); return -1; }
    NSUInteger threads = groups * MIN(tg, p.maxTotalThreadsPerThreadgroup);
    id<MTLBuffer> seed = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
    arc4random_buf(seed.contents, 32);
    id<MTLBuffer> out = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
    id<MTLBuffer> done = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> counts = [dev newBufferWithLength:threads * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> req = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> it = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    uint32_t five = 5;
    memcpy(req.contents, &five, 4);
    memcpy(it.contents, &iters, 4);
    NSArray *bufs = @[seed, comb, out, done, counts, req, it];
    double secs = 0;
    dispatch_bufs(q, p, bufs, groups, tg, &secs); // warmup
    memset(done.contents, 0, 4);
    if (!dispatch_bufs(q, p, bufs, groups, tg, &secs) || secs < 1e-6) return -1;
    uint32_t *c = (uint32_t *)counts.contents;
    uint64_t total = 0;
    for (NSUInteger i = 0; i < threads; i++) total += c[i];
    return (double)total / secs;
}

static double pipe_rate(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib,
                        NSString *name, NSUInteger threads, uint32_t nops, NSArray *extra) {
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
    if (!p) { NSLog(@"%@ %@", name, err); return -1; }
    NSUInteger tg = MIN((NSUInteger)128, p.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = (threads + tg - 1) / tg;
    NSUInteger actual = groups * tg;
    id<MTLBuffer> sink = [dev newBufferWithLength:actual * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nbuf = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    memcpy(nbuf.contents, &nops, 4);
    NSMutableArray *bufs = [NSMutableArray arrayWithObjects:sink, nbuf, nil];
    if (extra) [bufs addObjectsFromArray:extra];
    double secs = 0;
    dispatch_bufs(q, p, bufs, groups, tg, &secs);
    if (!dispatch_bufs(q, p, bufs, groups, tg, &secs) || secs < 1e-6) return -1;
    return (double)actual * (double)nops / secs;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        NSString *dir = @"/Users/cavey/cavey/vanity/kernels/metal";
        g_field = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"field.metal"] encoding:NSUTF8StringEncoding error:nil];
        g_rest = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"rest.metal"] encoding:NSUTF8StringEncoding error:nil];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSLog(@"device %@", dev.name);

        const NSUInteger kGroups = 512;
        const NSUInteger kTg = 128;
        const uint32_t kIters = 64;

        int widths[] = {6, 7, 8, 9, 10, 12};
        printf("== width sweep (batch 1, tg 128, 512 groups, iters 64) ==\n");
        int best_w = 8;
        double best = 0;
        for (int i = 0; i < 6; i++) {
            int w = widths[i];
            int use_tg = (w <= 9) ? 1 : 0;
            NSString *err = nil;
            NSDate *t0 = [NSDate date];
            id<MTLLibrary> lib = compile(dev, src_with(w, 1, use_tg, 0, 1), &err);
            if (!lib) { printf("W=%d compile fail: %s\n", w, err.UTF8String); continue; }
            id<MTLBuffer> comb = nil;
            build_table(dev, q, lib, w, &comb);
            double rate = keygen_rate(dev, q, lib, comb, 1, 0, kTg, kGroups, kIters);
            printf("W=%d tg_mem=%d  %.3f Mkeys/s  (compile+table %.2fs)\n",
                   w, use_tg, rate / 1e6, -[t0 timeIntervalSinceNow]);
            fflush(stdout);
            if (rate > best) { best = rate; best_w = w; }
        }
        printf("best width %d\n", best_w);

        int use_tg = best_w <= 9 ? 1 : 0;
        printf("== batch sweep W=%d ==\n", best_w);
        int best_b = 1;
        best = 0;
        for (int b = 1; b <= 4; b *= 2) {
            NSString *err = nil;
            id<MTLLibrary> lib = compile(dev, src_with(best_w, b, use_tg, 0, 1), &err);
            if (!lib) { printf("batch %d fail %s\n", b, err.UTF8String); continue; }
            id<MTLBuffer> comb = nil;
            build_table(dev, q, lib, best_w, &comb);
            double rate = keygen_rate(dev, q, lib, comb, b, 0, kTg, kGroups, kIters);
            printf("batch=%d  %.3f Mkeys/s\n", b, rate / 1e6);
            fflush(stdout);
            if (rate > best) { best = rate; best_b = b; }
        }
        printf("best batch %d\n", best_b);

        printf("== threadgroup sweep ==\n");
        NSUInteger best_tg = 128;
        best = 0;
        {
            NSString *err = nil;
            id<MTLLibrary> lib = compile(dev, src_with(best_w, best_b, use_tg, 0, 1), &err);
            id<MTLBuffer> comb = nil;
            build_table(dev, q, lib, best_w, &comb);
            NSUInteger tgs[] = {32, 64, 128, 256, 512};
            for (int i = 0; i < 5; i++) {
                NSUInteger tg = tgs[i];
                NSUInteger groups = (kGroups * kTg) / tg;
                double rate = keygen_rate(dev, q, lib, comb, best_b, 0, tg, groups, kIters);
                printf("tg=%lu groups=%lu  %.3f Mkeys/s\n", (unsigned long)tg, (unsigned long)groups, rate / 1e6);
                fflush(stdout);
                if (rate > best) { best = rate; best_tg = tg; }
            }
        }
        printf("best tg %lu\n", (unsigned long)best_tg);

        printf("== dual vs single (batch 1 and best batch) ==\n");
        int dual_batches[] = {1, best_b};
        int nd = best_b == 1 ? 1 : 2;
        for (int i = 0; i < nd; i++) {
            for (int dual = 0; dual <= 1; dual++) {
                NSString *err = nil;
                id<MTLLibrary> lib = compile(dev, src_with(best_w, dual_batches[i], use_tg, dual, 1), &err);
                if (!lib) { printf("dual %d batch %d fail\n", dual, dual_batches[i]); continue; }
                id<MTLBuffer> comb = nil;
                build_table(dev, q, lib, best_w, &comb);
                NSUInteger groups = (kGroups * kTg) / best_tg;
                double rate = keygen_rate(dev, q, lib, comb, dual_batches[i], dual, best_tg, groups, kIters);
                printf("dual=%d batch=%d  %.3f Mkeys/s\n", dual, dual_batches[i], rate / 1e6);
                fflush(stdout);
            }
        }

        if (best_w <= 9) {
            printf("== device table vs threadgroup W=%d ==\n", best_w);
            for (int tgmem = 0; tgmem <= 1; tgmem++) {
                id<MTLLibrary> lib = compile(dev, src_with(best_w, best_b, tgmem, 0, 1), nil);
                id<MTLBuffer> comb = nil;
                build_table(dev, q, lib, best_w, &comb);
                NSUInteger groups = (kGroups * kTg) / best_tg;
                double rate = keygen_rate(dev, q, lib, comb, best_b, 0, best_tg, groups, kIters);
                printf("use_tg=%d  %.3f Mkeys/s\n", tgmem, rate / 1e6);
                fflush(stdout);
            }
        }

        printf("== microbenchmarks ==\n");
        id<MTLLibrary> lib = compile(dev, src_with(best_w, 1, 0, 0, 1), nil);
        double mul = pipe_rate(dev, q, lib, @"bench_fe_mul", 4096, 4000, nil);
        double sha = pipe_rate(dev, q, lib, @"bench_sha512", 4096, 2000, nil);
        printf("fe_mul %.3f G/s\nsha512_32 %.3f M/s\n", mul / 1e9, sha / 1e6);

        // base58: magic vs native div, and 58^4
        uint8_t lut[58], patterns[5776];
        memset(patterns, 0, sizeof patterns);
        uint32_t npat = 1; memcpy(patterns, &npat, 4);
        patterns[4] = 8; // prefix length
        for (int i = 0; i < 8; i++) patterns[132 + i] = 20;
        for (int i = 0; i < 58; i++) lut[i] = (uint8_t)i;
        unsigned long long active = ~0ULL;
        memcpy(patterns + 5768, &active, 8);
        id<MTLBuffer> lbuf = [dev newBufferWithBytes:lut length:58 options:MTLResourceStorageModeShared];
        id<MTLBuffer> pbuf = [dev newBufferWithBytes:patterns length:5776 options:MTLResourceStorageModeShared];
        id<MTLLibrary> mag = compile(dev, src_with(8, 1, 0, 0, 1), nil);
        id<MTLLibrary> div = compile(dev, src_with(8, 1, 0, 0, 0), nil);
        double rmag = pipe_rate(dev, q, mag, @"bench_b58", 2048, 2000, @[lbuf, pbuf]);
        double rdiv = pipe_rate(dev, q, div, @"bench_b58", 2048, 2000, @[lbuf, pbuf]);
        double re4 = pipe_rate(dev, q, mag, @"bench_b58_e4", 2048, 400, nil);
        printf("b58 magic %.3f M/s\nb58 div %.3f M/s\nb58 e4 %.3f M/s\n", rmag / 1e6, rdiv / 1e6, re4 / 1e6);

        id<MTLBuffer> st = [dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
        NSError *err = nil;
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[mag newFunctionWithName:@"b58_div_selftest"] error:&err];
        double secs = 0;
        dispatch_bufs(q, ps, @[st], 1, 1, &secs);
        uint32_t *so = (uint32_t *)st.contents;
        printf("div selftest %u %u\n", so[0], so[1]);
        printf("FROZEN candidate W=%d batch=%d tg=%lu use_tg=%d\n", best_w, best_b, (unsigned long)best_tg, use_tg);
    }
    return 0;
}
