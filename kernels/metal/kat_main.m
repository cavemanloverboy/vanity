// Throwaway correctness harness: build the comb and check one RFC 8032 vector.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static NSString *load_src(void) {
    NSString *dir = @"/Users/cavey/cavey/vanity/kernels/metal";
    NSError *err = nil;
    NSString *field = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"field.metal"]
                                                encoding:NSUTF8StringEncoding error:&err];
    NSString *rest = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"rest.metal"]
                                               encoding:NSUTF8StringEncoding error:&err];
    if (!field || !rest) {
        NSLog(@"read %@", err);
        exit(1);
    }
    return [NSString stringWithFormat:@"#define COMB_W 8\n#define KP_BATCH 1\n#define USE_TG 1\n#define DUAL 0\n#define BS58_MAGIC 1\n%@\n%@", field, rest];
}

static id<MTLBuffer> buf(id<MTLDevice> dev, NSUInteger n) {
    return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
}

static void run(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> p,
                NSArray<id<MTLBuffer>> *bufs, NSUInteger groups, NSUInteger tg) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    for (NSUInteger i = 0; i < bufs.count; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:i];
    NSUInteger t = MIN(tg, p.maxTotalThreadsPerThreadgroup);
    if (t == 0) t = 1;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSLog(@"gpu fail %@", cb.error);
        exit(1);
    }
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        NSDate *t0 = [NSDate date];
        id<MTLLibrary> lib = [dev newLibraryWithSource:load_src() options:nil error:&err];
        if (!lib) { NSLog(@"compile %@", err); return 1; }
        NSLog(@"compile %.2fs", -[t0 timeIntervalSinceNow]);
        id<MTLCommandQueue> q = [dev newCommandQueue];

        const uint32_t COMB_POS = 128, WINDOWS = 32, TABLE = WINDOWS * COMB_POS;
        id<MTLBuffer> scratch = buf(dev, (NSUInteger)TABLE * 40 * 4);
        id<MTLBuffer> comb = buf(dev, (NSUInteger)TABLE * 30 * 4);
        id<MTLFunction> fb = [lib newFunctionWithName:@"build_comb"];
        id<MTLFunction> fn = [lib newFunctionWithName:@"normalize_comb"];
        id<MTLComputePipelineState> pb = [dev newComputePipelineStateWithFunction:fb error:&err];
        id<MTLComputePipelineState> pn = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pb || !pn) { NSLog(@"pipe %@", err); return 1; }
        t0 = [NSDate date];
        run(dev, q, pb, @[scratch], 1, 1);
        NSUInteger ntg = MIN((NSUInteger)128, pn.maxTotalThreadsPerThreadgroup);
        NSUInteger groups = (TABLE + ntg - 1) / ntg;
        run(dev, q, pn, @[scratch, comb], groups, ntg);
        NSLog(@"table %.2fs", -[t0 timeIntervalSinceNow]);

        uint8_t seed[32] = {
            0x9d,0x61,0xb1,0x9d,0xef,0xfd,0x5a,0x60,0xba,0x84,0x4a,0xf4,0x92,0xec,0x2c,0xc4,
            0x44,0x49,0xc5,0x69,0x7b,0x32,0x69,0x19,0x70,0x3b,0xac,0x03,0x1c,0xae,0x7f,0x60};
        uint8_t expect[32] = {
            0xd7,0x5a,0x98,0x01,0x82,0xb1,0x0a,0xb7,0xd5,0x4b,0xfe,0xd3,0xc9,0x64,0x07,0x3a,
            0x0e,0xe1,0x72,0xf3,0xda,0xa6,0x23,0x25,0xaf,0x02,0x1a,0x68,0xf7,0x07,0x51,0x1a};
        id<MTLBuffer> seeds = buf(dev, 32);
        id<MTLBuffer> pubs = buf(dev, 32);
        id<MTLBuffer> nbuf = buf(dev, 4);
        memcpy(seeds.contents, seed, 32);
        uint32_t one = 1;
        memcpy(nbuf.contents, &one, 4);
        id<MTLFunction> fk = [lib newFunctionWithName:@"kat_pubkeys"];
        id<MTLComputePipelineState> pk = [dev newComputePipelineStateWithFunction:fk error:&err];
        if (!pk) { NSLog(@"kat pipe %@", err); return 1; }
        NSLog(@"kat max threads %lu", (unsigned long)pk.maxTotalThreadsPerThreadgroup);
        run(dev, q, pk, @[comb, seeds, pubs, nbuf], 1, MIN((NSUInteger)128, pk.maxTotalThreadsPerThreadgroup));
        uint8_t *got = (uint8_t *)pubs.contents;
        printf("got    ");
        for (int i = 0; i < 32; i++) printf("%02x", got[i]);
        printf("\nexpect ");
        for (int i = 0; i < 32; i++) printf("%02x", expect[i]);
        printf("\n%s\n", memcmp(got, expect, 32) == 0 ? "KAT OK" : "KAT FAIL");
        return memcmp(got, expect, 32) == 0 ? 0 : 2;
    }
}
