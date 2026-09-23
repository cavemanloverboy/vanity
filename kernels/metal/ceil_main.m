#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdint.h>

static double run(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, NSUInteger threads, uint32_t nops) {
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
    if (!p) { NSLog(@"%@", err); return -1; }
    NSUInteger tg = MIN((NSUInteger)128, p.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = (threads + tg - 1) / tg;
    NSUInteger actual = groups * tg;
    id<MTLBuffer> sink = [dev newBufferWithLength:actual * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nbuf = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    memcpy(nbuf.contents, &nops, 4);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    double secs = 0;
    for (int pass = 0; pass < 2; pass++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:sink offset:0 atIndex:0];
        [enc setBuffer:nbuf offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        secs = cb.GPUEndTime - cb.GPUStartTime;
    }
    return (double)actual * (double)nops / secs;
}

int main(void) {
    @autoreleasepool {
        NSString *field = [NSString stringWithContentsOfFile:@"/Users/cavey/cavey/vanity/kernels/metal/field.metal" encoding:NSUTF8StringEncoding error:nil];
        NSString *src = [NSString stringWithFormat:
            @"#define FE_NARROW 0\n#define FE_ATTR\n%@\n"
            "kernel void bench_mul(device int *sink [[buffer(0)]], constant uint &n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n"
            "  fe a, b; fe_1(a); fe_1(b); b[0] = 2 + (int)(tid & 7u);\n"
            "  for (uint i = 0; i < n; i++) fe_mul_hot(a, a, b);\n"
            "  sink[tid] = a[0];\n"
            "}\n"
            "kernel void bench_mul4(device int *sink [[buffer(0)]], constant uint &n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n"
            "  fe a0, a1, a2, a3, b; fe_1(a0); fe_1(a1); fe_1(a2); fe_1(a3); fe_1(b); b[0] = 3;\n"
            "  for (uint i = 0; i < n; i++) { fe_mul_hot(a0, a0, b); fe_mul_hot(a1, a1, b); fe_mul_hot(a2, a2, b); fe_mul_hot(a3, a3, b); }\n"
            "  sink[tid] = a0[0] ^ a1[0] ^ a2[0] ^ a3[0];\n"
            "}\n"
            "kernel void bench_sq(device int *sink [[buffer(0)]], constant uint &n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n"
            "  fe a; fe_1(a); a[0] = 2 + (int)(tid & 7u);\n"
            "  for (uint i = 0; i < n; i++) fe_sq_hot(a, a);\n"
            "  sink[tid] = a[0];\n"
            "}\n", field];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        NSUInteger threads = 65536;
        uint32_t nops = 2048;
        double mul = run(dev, lib, @"bench_mul", threads, nops);
        double mul4 = run(dev, lib, @"bench_mul4", threads, nops);
        double sq = run(dev, lib, @"bench_sq", threads, nops);
        printf("fe_mul  1-chain  %.3f G/s\n", mul / 1e9);
        printf("fe_mul  4-chain  %.3f G/s\n", (mul4 * 4.0) / 1e9);
        printf("fe_sq   1-chain  %.3f G/s\n", sq / 1e9);
        printf("sq/mul %.3f\n", sq / mul);
    }
    return 0;
}
