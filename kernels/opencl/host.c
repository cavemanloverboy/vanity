/* host.c — OpenCL backend for the vanity grinder.

   Provides the exact same extern "C" ABI as the CUDA host (kernels/vanity.cu
   and kernels/vanity_keypair.cu), so src/main.rs links against it unchanged.
   Built instead of the CUDA sources when the `opencl` cargo feature is set.

   The OpenCL kernels are compiled at runtime from sources embedded by
   build.rs into cl_sources.h. Because OpenCL 1.2 has no in-kernel clock and
   no portable 64-bit atomics, each launch runs a bounded `max_iters` per
   work-item (adapted toward a target wall-clock duration) and the per-work-
   item attempt counts are summed on the host. */

#define CL_SILENCE_DEPRECATION
#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#include "cl_sources.h"

/* ─── tunables ──────────────────────────────────────────────────────────── */

#define GRIND_LOCAL        256u
#define GRIND_WAVES        8u      /* work-groups per compute unit */
#define GRIND_ITERS_INIT   2048u
#define GRIND_ITERS_MIN    256u
#define GRIND_ITERS_MAX    (1u << 22)

#define KP_LOCAL           128u
#define KP_WAVES           8u
#define KP_ITERS_INIT      32u
#define KP_ITERS_MIN       4u
#define KP_ITERS_MAX       (1u << 18)

#define TARGET_LAUNCH_SEC  0.4    /* desired wall time per launch */

/* ─── SHA-256 host helpers (for grind precompute, mirrors vanity.cu) ─────── */

#define ROTR(a, b)  (((a) >> (b)) | ((a) << (32 - (b))))
#define H_CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define H_MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define H_EP0(x)  (ROTR(x, 2)  ^ ROTR(x, 13) ^ ROTR(x, 22))
#define H_EP1(x)  (ROTR(x, 6)  ^ ROTR(x, 11) ^ ROTR(x, 25))
#define H_SIG0(x) (ROTR(x, 7)  ^ ROTR(x, 18) ^ ((x) >> 3))
#define H_SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))
#define H_SHA_R(K, M)                                       \
    do {                                                    \
        uint32_t t1 = h + H_EP1(e) + H_CH(e, f, g) + (K) + (M); \
        uint32_t t2 = H_EP0(a) + H_MAJ(a, b, c);            \
        h = g; g = f; f = e; e = d + t1;                    \
        d = c; c = b; b = a; a = t1 + t2;                   \
    } while (0)

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] <<  8) | ((uint32_t)p[3]);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─── OpenCL plumbing ────────────────────────────────────────────────────── */

static void cl_die(const char *what, cl_int err) {
    fprintf(stderr, "opencl: %s failed (error %d)\n", what, (int)err);
    exit(EXIT_FAILURE);
}
#define CK(expr, what) do { cl_int _e = (expr); if (_e != CL_SUCCESS) cl_die(what, _e); } while (0)

/* Flatten devices of `type` across all platforms; return the `want`-th. */
static int pick_device(int want, cl_device_type type,
                       cl_platform_id *out_plat, cl_device_id *out_dev) {
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS || nplat == 0) return 0;
    cl_platform_id plats[32];
    if (nplat > 32) nplat = 32;
    clGetPlatformIDs(nplat, plats, NULL);

    int idx = 0;
    for (cl_uint p = 0; p < nplat; p++) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], type, 0, NULL, &nd) != CL_SUCCESS || nd == 0) continue;
        cl_device_id devs[32];
        if (nd > 32) nd = 32;
        clGetDeviceIDs(plats[p], type, nd, devs, NULL);
        for (cl_uint d = 0; d < nd; d++) {
            if (idx == want) { *out_plat = plats[p]; *out_dev = devs[d]; return 1; }
            idx++;
        }
    }
    return 0;
}

static cl_device_id select_device(int id, cl_platform_id *plat) {
    cl_device_id dev;
    if (pick_device(id, CL_DEVICE_TYPE_GPU, plat, &dev)) return dev;
    if (pick_device(id, CL_DEVICE_TYPE_ALL, plat, &dev)) return dev;
    if (pick_device(0,  CL_DEVICE_TYPE_ALL, plat, &dev)) return dev;
    fprintf(stderr, "opencl: no usable device for id %d\n", id);
    exit(EXIT_FAILURE);
}

static cl_program build_program(cl_context ctx, cl_device_id dev,
                                const char **srcs, cl_uint n, const char *name) {
    size_t lens[16];
    for (cl_uint i = 0; i < n; i++) lens[i] = strlen(srcs[i]);
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, n, srcs, lens, &err);
    CK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = (char *)malloc(log_sz + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        log[log_sz] = '\0';
        fprintf(stderr, "opencl: building %s program failed:\n%s\n", name, log);
        free(log);
        exit(EXIT_FAILURE);
    }
    return prog;
}

static cl_uint compute_units(cl_device_id dev) {
    cl_uint cu = 1;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    return cu ? cu : 1;
}

static size_t clamp_local(cl_device_id dev, size_t want) {
    size_t maxwg = 1;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof maxwg, &maxwg, NULL);
    return want < maxwg ? want : maxwg;
}

static cl_mem buf_copy(cl_context ctx, size_t sz, const void *host) {
    cl_int err;
    cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sz ? sz : 1, sz ? (void *)host : (void *)&err, &err);
    CK(err, "clCreateBuffer(copy)");
    return m;
}

#define MATCH_PLAN_BYTES ((1u + 44u * 58u + 44u) * sizeof(uint32_t))

/* Grow/shrink max_iters toward TARGET_LAUNCH_SEC based on last launch time. */
static uint32_t adapt_iters(uint32_t cur, double elapsed, uint32_t lo, uint32_t hi) {
    if (elapsed <= 1e-6) return hi;
    double factor = TARGET_LAUNCH_SEC / elapsed;
    if (factor < 0.25) factor = 0.25;
    if (factor > 4.0)  factor = 4.0;
    double next = (double)cur * factor;
    if (next < lo) next = lo;
    if (next > hi) next = hi;
    return (uint32_t)next;
}

/* ─── grind context ──────────────────────────────────────────────────────── */

typedef struct {
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
    cl_mem  seed, w0, sr7, w1, glyph, mlut, prefix, suffix, out, done, counts;
    size_t  local, global;
    uint32_t prefix_count, suffix_count;
    uint32_t max_iters;
    cl_event event;
    int      in_flight;
    double   launch_time;
    uint32_t *counts_host;
} GrindCtx;

void *gpu_grind_init(int id, uint8_t *base, uint8_t *owner,
                     uint8_t *prefixes, uint64_t prefix_count,
                     uint8_t *suffixes, uint64_t suffix_count,
                     bool case_insensitive) {
    cl_platform_id plat;
    cl_device_id dev = select_device(id, &plat);
    cl_int err;

    GrindCtx *c = (GrindCtx *)calloc(1, sizeof(GrindCtx));
    c->context = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CK(err, "clCreateContext");
    c->queue = clCreateCommandQueue(c->context, dev, 0, &err);
    CK(err, "clCreateCommandQueue");

    const char *srcs[] = { CL_PREAMBLE, CL_SHA256, CL_BASE58, CL_VANITY };
    c->program = build_program(c->context, dev, srcs, 4, "grind");
    c->kernel = clCreateKernel(c->program, "vanity_search", &err);
    CK(err, "clCreateKernel(vanity_search)");

    c->local  = clamp_local(dev, GRIND_LOCAL);
    c->global = (size_t)compute_units(dev) * GRIND_WAVES * c->local;
    c->prefix_count = (uint32_t)prefix_count;
    c->suffix_count = (uint32_t)suffix_count;
    c->max_iters  = GRIND_ITERS_INIT;
    c->counts_host = (uint32_t *)malloc(c->global * sizeof(uint32_t));

    /* Canonicalizing LUT; match plans arrive precomputed by the Rust host. */
    static const char alphabet_normal[59] =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    static const char alphabet_ci[59] =
        "123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";
    const char *alphabet = case_insensitive ? alphabet_ci : alphabet_normal;

    uint8_t match_lut[58];
    for (int i = 0; i < 58; ++i) {
        match_lut[i] = (uint8_t)i;
        for (int j = 0; j < i; ++j)
            if (alphabet[j] == alphabet[i]) { match_lut[i] = (uint8_t)j; break; }
    }

    /* SHA-256 block-1 schedule from owner[16..32] (loop invariant). */
    uint32_t W1[64];
    {
        uint8_t b1[64] = {0};
        memcpy(b1, owner + 16, 16);
        b1[16] = 0x80; b1[62] = 0x02; b1[63] = 0x80;
        for (int i = 0; i < 16; ++i) W1[i] = be32(b1 + 4 * i);
        for (int i = 16; i < 64; ++i)
            W1[i] = H_SIG1(W1[i-2]) + W1[i-7] + H_SIG0(W1[i-15]) + W1[i-16];
    }

    /* Block-0 schedule fixed slots: [0..7]=base, [12..15]=owner[0..15]. */
    uint32_t W0[16] = {0};
    for (int i = 0; i < 8; ++i) W0[i]      = be32(base  + 4 * i);
    for (int i = 0; i < 4; ++i) W0[12 + i] = be32(owner + 4 * i);

    /* Working state after block-0 rounds 0..7 (depends only on base). */
    uint32_t SR7[8];
    {
        uint32_t W[8];
        for (int i = 0; i < 8; ++i) W[i] = be32(base + 4 * i);
        /* `c` shadows the ctx pointer in this block; the H_SHA_R macro
           hardcodes the state variable names a..h. */
        uint32_t a = 0x6a09e667U, b = 0xbb67ae85U, c = 0x3c6ef372U, d = 0xa54ff53aU;
        uint32_t e = 0x510e527fU, f = 0x9b05688cU, g = 0x1f83d9abU, h = 0x5be0cd19U;
        H_SHA_R(0x428A2F98U, W[0]); H_SHA_R(0x71374491U, W[1]);
        H_SHA_R(0xB5C0FBCFU, W[2]); H_SHA_R(0xE9B5DBA5U, W[3]);
        H_SHA_R(0x3956C25BU, W[4]); H_SHA_R(0x59F111F1U, W[5]);
        H_SHA_R(0x923F82A4U, W[6]); H_SHA_R(0xAB1C5ED5U, W[7]);
        SR7[0]=a; SR7[1]=b; SR7[2]=c; SR7[3]=d; SR7[4]=e; SR7[5]=f; SR7[6]=g; SR7[7]=h;
    }

    /* hash byte -> alnum glyph LUT. */
    uint8_t glyph[256];
    static const char alnum[63] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (unsigned u = 0; u < 256; ++u) glyph[u] = (uint8_t)alnum[u % 62];

    c->seed   = clCreateBuffer(c->context, CL_MEM_READ_ONLY, 32, NULL, &err); CK(err, "buf seed");
    c->w0     = buf_copy(c->context, sizeof W0, W0);
    c->sr7    = buf_copy(c->context, sizeof SR7, SR7);
    c->w1     = buf_copy(c->context, sizeof W1, W1);
    c->glyph  = buf_copy(c->context, sizeof glyph, glyph);
    c->mlut   = buf_copy(c->context, sizeof match_lut, match_lut);
    c->prefix = buf_copy(c->context, MATCH_PLAN_BYTES, prefixes);
    c->suffix = buf_copy(c->context, MATCH_PLAN_BYTES, suffixes);
    c->out    = clCreateBuffer(c->context, CL_MEM_WRITE_ONLY, 16, NULL, &err); CK(err, "buf out");
    c->done   = clCreateBuffer(c->context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err); CK(err, "buf done");
    c->counts = clCreateBuffer(c->context, CL_MEM_WRITE_ONLY, c->global * sizeof(cl_uint), NULL, &err); CK(err, "buf counts");

    /* Bind invariant args once; seed/done/max_iters refreshed per launch. */
    CK(clSetKernelArg(c->kernel, 1, sizeof(cl_mem), &c->w0),     "arg w0");
    CK(clSetKernelArg(c->kernel, 2, sizeof(cl_mem), &c->sr7),    "arg sr7");
    CK(clSetKernelArg(c->kernel, 3, sizeof(cl_mem), &c->w1),     "arg w1");
    CK(clSetKernelArg(c->kernel, 4, sizeof(cl_mem), &c->glyph),  "arg glyph");
    CK(clSetKernelArg(c->kernel, 5, sizeof(cl_mem), &c->mlut),   "arg mlut");
    CK(clSetKernelArg(c->kernel, 6, sizeof(cl_mem), &c->prefix), "arg prefix");
    CK(clSetKernelArg(c->kernel, 7, sizeof(cl_uint), &c->prefix_count), "arg prefix_count");
    CK(clSetKernelArg(c->kernel, 8, sizeof(cl_mem), &c->suffix), "arg suffix");
    CK(clSetKernelArg(c->kernel, 9, sizeof(cl_uint), &c->suffix_count), "arg suffix_count");
    CK(clSetKernelArg(c->kernel, 10, sizeof(cl_mem), &c->out),   "arg out");
    CK(clSetKernelArg(c->kernel, 11, sizeof(cl_mem), &c->done),  "arg done");
    CK(clSetKernelArg(c->kernel, 12, sizeof(cl_mem), &c->counts),"arg counts");

    return c;
}

void gpu_grind_launch(void *opaque, uint8_t *seed) {
    GrindCtx *c = (GrindCtx *)opaque;
    if (c->in_flight) { clReleaseEvent(c->event); c->in_flight = 0; }

    cl_int zero = 0;
    uint8_t out_zero[16] = {0};
    CK(clEnqueueWriteBuffer(c->queue, c->seed, CL_FALSE, 0, 32, seed, 0, NULL, NULL), "write seed");
    CK(clEnqueueWriteBuffer(c->queue, c->out, CL_FALSE, 0, 16, out_zero, 0, NULL, NULL), "clear out");
    CK(clEnqueueWriteBuffer(c->queue, c->done, CL_FALSE, 0, sizeof zero, &zero, 0, NULL, NULL), "write done");
    CK(clSetKernelArg(c->kernel, 0, sizeof(cl_mem), &c->seed), "arg seed");
    CK(clSetKernelArg(c->kernel, 13, sizeof(cl_uint), &c->max_iters), "arg max_iters");

    c->launch_time = now_sec();
    CK(clEnqueueNDRangeKernel(c->queue, c->kernel, 1, NULL, &c->global, &c->local, 0, NULL, &c->event),
       "enqueue grind");
    clFlush(c->queue);
    c->in_flight = 1;
}

int gpu_grind_query(void *opaque) {
    GrindCtx *c = (GrindCtx *)opaque;
    if (!c->in_flight) return 1;
    cl_int status;
    if (clGetEventInfo(c->event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                       sizeof status, &status, NULL) != CL_SUCCESS) return 1;
    return status == CL_COMPLETE ? 1 : 0;
}

/* out: [seed16:16][count:8] */
void gpu_grind_read(void *opaque, uint8_t *out) {
    GrindCtx *c = (GrindCtx *)opaque;
    clWaitForEvents(1, &c->event);
    double elapsed = now_sec() - c->launch_time;

    CK(clEnqueueReadBuffer(c->queue, c->out, CL_TRUE, 0, 16, out, 0, NULL, NULL), "read out");
    CK(clEnqueueReadBuffer(c->queue, c->counts, CL_TRUE, 0, c->global * sizeof(cl_uint),
                           c->counts_host, 0, NULL, NULL), "read counts");
    uint64_t total = 0;
    for (size_t i = 0; i < c->global; ++i) total += c->counts_host[i];
    memcpy(out + 16, &total, 8);

    clReleaseEvent(c->event);
    c->in_flight = 0;
    c->max_iters = adapt_iters(c->max_iters, elapsed, GRIND_ITERS_MIN, GRIND_ITERS_MAX);
}

void gpu_grind_destroy(void *opaque) {
    GrindCtx *c = (GrindCtx *)opaque;
    clFinish(c->queue);
    if (c->in_flight) clReleaseEvent(c->event);
    cl_mem bufs[] = {c->seed,c->w0,c->sr7,c->w1,c->glyph,c->mlut,c->prefix,c->suffix,c->out,c->done,c->counts};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; ++i) clReleaseMemObject(bufs[i]);
    clReleaseKernel(c->kernel);
    clReleaseProgram(c->program);
    clReleaseCommandQueue(c->queue);
    clReleaseContext(c->context);
    free(c->counts_host);
    free(c);
}

/* ─── keypair context ────────────────────────────────────────────────────── */

typedef struct {
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
    cl_mem  seed, mlut, prefix, suffix, out, done, counts, comb;
    size_t  local, global;
    uint32_t prefix_count, suffix_count;
    uint32_t max_iters;
    cl_event event;
    int      in_flight;
    double   launch_time;
    uint32_t *counts_host;
} KeypairCtx;

/* ge_niels = 4 fe × 10 × 4 bytes; must match OpenCL layout. */
#define COMB_NIELS_BYTES 160u
#define COMB_TABLE_BYTES (52u * 16u * COMB_NIELS_BYTES)

void *gpu_keypair_init(int id, uint8_t *prefixes, uint64_t prefix_count,
                       uint8_t *suffixes, uint64_t suffix_count, bool case_insensitive) {
    cl_platform_id plat;
    cl_device_id dev = select_device(id, &plat);
    cl_int err;

    KeypairCtx *c = (KeypairCtx *)calloc(1, sizeof(KeypairCtx));
    c->context = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CK(err, "clCreateContext");
    c->queue = clCreateCommandQueue(c->context, dev, 0, &err);
    CK(err, "clCreateCommandQueue");

    const char *srcs[] = { CL_PREAMBLE, CL_SHA256, CL_SHA512, CL_FE, CL_PRECOMP, CL_GE, CL_BASE58, CL_KEYPAIR };
    c->program = build_program(c->context, dev, srcs, 8, "keypair");
    c->kernel = clCreateKernel(c->program, "vanity_keypair_search", &err);
    CK(err, "clCreateKernel(vanity_keypair_search)");

    c->local  = clamp_local(dev, KP_LOCAL);
    c->global = (size_t)compute_units(dev) * KP_WAVES * c->local;
    c->prefix_count = (uint32_t)prefix_count;
    c->suffix_count = (uint32_t)suffix_count;
    c->max_iters = KP_ITERS_INIT;
    c->counts_host = (uint32_t *)malloc(c->global * sizeof(uint32_t));

    /* Canonical base58 LUT (same scheme as gpu_grind_init / CUDA
       gpu_keypair_init). Match plans arrive precomputed by the Rust host. */
    static const char alphabet_normal[59] =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    static const char alphabet_ci[59] =
        "123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";
    const char *alphabet = case_insensitive ? alphabet_ci : alphabet_normal;

    uint8_t match_lut[58];
    for (int i = 0; i < 58; ++i) {
        match_lut[i] = (uint8_t)i;
        for (int j = 0; j < i; ++j)
            if (alphabet[j] == alphabet[i]) { match_lut[i] = (uint8_t)j; break; }
    }
    c->seed   = clCreateBuffer(c->context, CL_MEM_READ_ONLY, 32, NULL, &err); CK(err, "buf seed");
    c->mlut   = buf_copy(c->context, sizeof match_lut, match_lut);
    c->prefix = buf_copy(c->context, MATCH_PLAN_BYTES, prefixes);
    c->suffix = buf_copy(c->context, MATCH_PLAN_BYTES, suffixes);
    c->out    = clCreateBuffer(c->context, CL_MEM_WRITE_ONLY, 32, NULL, &err); CK(err, "buf out");
    c->done   = clCreateBuffer(c->context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err); CK(err, "buf done");
    c->counts = clCreateBuffer(c->context, CL_MEM_WRITE_ONLY, c->global * sizeof(cl_uint), NULL, &err); CK(err, "buf counts");
    c->comb   = clCreateBuffer(c->context, CL_MEM_READ_WRITE, COMB_TABLE_BYTES, NULL, &err); CK(err, "buf comb");

    /* Build radix-32 comb table once (single work-item). */
    {
        cl_kernel build = clCreateKernel(c->program, "build_comb_table", &err);
        CK(err, "clCreateKernel(build_comb_table)");
        CK(clSetKernelArg(build, 0, sizeof(cl_mem), &c->comb), "arg comb");
        size_t one = 1;
        CK(clEnqueueNDRangeKernel(c->queue, build, 1, NULL, &one, &one, 0, NULL, NULL),
           "enqueue build_comb_table");
        CK(clFinish(c->queue), "finish build_comb_table");
        clReleaseKernel(build);
    }

    CK(clSetKernelArg(c->kernel, 1, sizeof(cl_mem), &c->mlut), "arg mlut");
    CK(clSetKernelArg(c->kernel, 2, sizeof(cl_mem), &c->prefix), "arg prefix");
    CK(clSetKernelArg(c->kernel, 3, sizeof(cl_uint), &c->prefix_count), "arg prefix_count");
    CK(clSetKernelArg(c->kernel, 4, sizeof(cl_mem), &c->suffix), "arg suffix");
    CK(clSetKernelArg(c->kernel, 5, sizeof(cl_uint), &c->suffix_count), "arg suffix_count");
    CK(clSetKernelArg(c->kernel, 6, sizeof(cl_mem), &c->out), "arg out");
    CK(clSetKernelArg(c->kernel, 7, sizeof(cl_mem), &c->done), "arg done");
    CK(clSetKernelArg(c->kernel, 8, sizeof(cl_mem), &c->counts), "arg counts");
    CK(clSetKernelArg(c->kernel, 9, sizeof(cl_mem), &c->comb), "arg comb");

    return c;
}

void gpu_keypair_launch(void *opaque, uint8_t *seed) {
    KeypairCtx *c = (KeypairCtx *)opaque;
    if (c->in_flight) { clReleaseEvent(c->event); c->in_flight = 0; }

    cl_int zero = 0;
    uint8_t out_zero[32] = {0};
    CK(clEnqueueWriteBuffer(c->queue, c->seed, CL_FALSE, 0, 32, seed, 0, NULL, NULL), "write seed");
    CK(clEnqueueWriteBuffer(c->queue, c->out, CL_FALSE, 0, 32, out_zero, 0, NULL, NULL), "clear out");
    CK(clEnqueueWriteBuffer(c->queue, c->done, CL_FALSE, 0, sizeof zero, &zero, 0, NULL, NULL), "write done");
    CK(clSetKernelArg(c->kernel, 0, sizeof(cl_mem), &c->seed), "arg seed");
    CK(clSetKernelArg(c->kernel, 10, sizeof(cl_uint), &c->max_iters), "arg max_iters");

    c->launch_time = now_sec();
    CK(clEnqueueNDRangeKernel(c->queue, c->kernel, 1, NULL, &c->global, &c->local, 0, NULL, &c->event),
       "enqueue keypair");
    clFlush(c->queue);
    c->in_flight = 1;
}

int gpu_keypair_query(void *opaque) {
    KeypairCtx *c = (KeypairCtx *)opaque;
    if (!c->in_flight) return 1;
    cl_int status;
    if (clGetEventInfo(c->event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                       sizeof status, &status, NULL) != CL_SUCCESS) return 1;
    return status == CL_COMPLETE ? 1 : 0;
}

/* out: [seed:32][count:8] */
void gpu_keypair_read(void *opaque, uint8_t *out) {
    KeypairCtx *c = (KeypairCtx *)opaque;
    clWaitForEvents(1, &c->event);
    double elapsed = now_sec() - c->launch_time;

    CK(clEnqueueReadBuffer(c->queue, c->out, CL_TRUE, 0, 32, out, 0, NULL, NULL), "read out");
    CK(clEnqueueReadBuffer(c->queue, c->counts, CL_TRUE, 0, c->global * sizeof(cl_uint),
                           c->counts_host, 0, NULL, NULL), "read counts");
    uint64_t total = 0;
    for (size_t i = 0; i < c->global; ++i) total += c->counts_host[i];
    memcpy(out + 32, &total, 8);

    clReleaseEvent(c->event);
    c->in_flight = 0;
    c->max_iters = adapt_iters(c->max_iters, elapsed, KP_ITERS_MIN, KP_ITERS_MAX);
}

void gpu_keypair_destroy(void *opaque) {
    KeypairCtx *c = (KeypairCtx *)opaque;
    clFinish(c->queue);
    if (c->in_flight) clReleaseEvent(c->event);
    cl_mem bufs[] = {c->seed,c->mlut,c->prefix,c->suffix,c->out,c->done,c->counts,c->comb};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; ++i) clReleaseMemObject(bufs[i]);
    clReleaseKernel(c->kernel);
    clReleaseProgram(c->program);
    clReleaseCommandQueue(c->queue);
    clReleaseContext(c->context);
    free(c->counts_host);
    free(c);
}

/* ─── doppler context ────────────────────────────────────────────────────── */

/* Same ed25519 machinery as the keypair backend (reuses the KP_* tunables
   and radix-32 comb table); only the match differs (sign-extendable-segment
   count, no base58). */
typedef struct {
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
    cl_mem  seed, out, done, counts, comb;
    size_t  local, global;
    cl_uint  required_segments;
    uint32_t max_iters;
    cl_event event;
    int      in_flight;
    double   launch_time;
    uint32_t *counts_host;
} DopplerCtx;

void *gpu_doppler_init(int id, uint32_t required_segments) {
    cl_platform_id plat;
    cl_device_id dev = select_device(id, &plat);
    cl_int err;

    DopplerCtx *c = (DopplerCtx *)calloc(1, sizeof(DopplerCtx));
    c->context = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CK(err, "clCreateContext");
    c->queue = clCreateCommandQueue(c->context, dev, 0, &err);
    CK(err, "clCreateCommandQueue");

    const char *srcs[] = { CL_PREAMBLE, CL_SHA256, CL_SHA512, CL_FE, CL_PRECOMP, CL_GE, CL_DOPPLER };
    c->program = build_program(c->context, dev, srcs, 7, "doppler");
    c->kernel = clCreateKernel(c->program, "vanity_doppler_search", &err);
    CK(err, "clCreateKernel(vanity_doppler_search)");

    c->local  = clamp_local(dev, KP_LOCAL);
    c->global = (size_t)compute_units(dev) * KP_WAVES * c->local;
    c->required_segments = required_segments;
    c->max_iters = KP_ITERS_INIT;
    c->counts_host = (uint32_t *)malloc(c->global * sizeof(uint32_t));

    c->seed   = clCreateBuffer(c->context, CL_MEM_READ_ONLY, 32, NULL, &err); CK(err, "buf seed");
    c->out    = clCreateBuffer(c->context, CL_MEM_WRITE_ONLY, 32, NULL, &err); CK(err, "buf out");
    c->done   = clCreateBuffer(c->context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err); CK(err, "buf done");
    c->counts = clCreateBuffer(c->context, CL_MEM_WRITE_ONLY, c->global * sizeof(cl_uint), NULL, &err); CK(err, "buf counts");
    c->comb   = clCreateBuffer(c->context, CL_MEM_READ_WRITE, COMB_TABLE_BYTES, NULL, &err); CK(err, "buf comb");

    {
        cl_kernel build = clCreateKernel(c->program, "build_comb_table", &err);
        CK(err, "clCreateKernel(build_comb_table)");
        CK(clSetKernelArg(build, 0, sizeof(cl_mem), &c->comb), "arg comb");
        size_t one = 1;
        CK(clEnqueueNDRangeKernel(c->queue, build, 1, NULL, &one, &one, 0, NULL, NULL),
           "enqueue build_comb_table");
        CK(clFinish(c->queue), "finish build_comb_table");
        clReleaseKernel(build);
    }

    CK(clSetKernelArg(c->kernel, 1, sizeof(cl_uint), &c->required_segments), "arg required_segments");
    CK(clSetKernelArg(c->kernel, 2, sizeof(cl_mem), &c->out), "arg out");
    CK(clSetKernelArg(c->kernel, 3, sizeof(cl_mem), &c->done), "arg done");
    CK(clSetKernelArg(c->kernel, 4, sizeof(cl_mem), &c->counts), "arg counts");
    CK(clSetKernelArg(c->kernel, 5, sizeof(cl_mem), &c->comb), "arg comb");

    return c;
}

void gpu_doppler_launch(void *opaque, uint8_t *seed) {
    DopplerCtx *c = (DopplerCtx *)opaque;
    if (c->in_flight) { clReleaseEvent(c->event); c->in_flight = 0; }

    cl_int zero = 0;
    uint8_t out_zero[32] = {0};
    CK(clEnqueueWriteBuffer(c->queue, c->seed, CL_FALSE, 0, 32, seed, 0, NULL, NULL), "write seed");
    CK(clEnqueueWriteBuffer(c->queue, c->out, CL_FALSE, 0, 32, out_zero, 0, NULL, NULL), "clear out");
    CK(clEnqueueWriteBuffer(c->queue, c->done, CL_FALSE, 0, sizeof zero, &zero, 0, NULL, NULL), "write done");
    CK(clSetKernelArg(c->kernel, 0, sizeof(cl_mem), &c->seed), "arg seed");
    CK(clSetKernelArg(c->kernel, 6, sizeof(cl_uint), &c->max_iters), "arg max_iters");

    c->launch_time = now_sec();
    CK(clEnqueueNDRangeKernel(c->queue, c->kernel, 1, NULL, &c->global, &c->local, 0, NULL, &c->event),
       "enqueue doppler");
    clFlush(c->queue);
    c->in_flight = 1;
}

int gpu_doppler_query(void *opaque) {
    DopplerCtx *c = (DopplerCtx *)opaque;
    if (!c->in_flight) return 1;
    cl_int status;
    if (clGetEventInfo(c->event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                       sizeof status, &status, NULL) != CL_SUCCESS) return 1;
    return status == CL_COMPLETE ? 1 : 0;
}

/* out: [seed:32][count:8] */
void gpu_doppler_read(void *opaque, uint8_t *out) {
    DopplerCtx *c = (DopplerCtx *)opaque;
    clWaitForEvents(1, &c->event);
    double elapsed = now_sec() - c->launch_time;

    CK(clEnqueueReadBuffer(c->queue, c->out, CL_TRUE, 0, 32, out, 0, NULL, NULL), "read out");
    CK(clEnqueueReadBuffer(c->queue, c->counts, CL_TRUE, 0, c->global * sizeof(cl_uint),
                           c->counts_host, 0, NULL, NULL), "read counts");
    uint64_t total = 0;
    for (size_t i = 0; i < c->global; ++i) total += c->counts_host[i];
    memcpy(out + 32, &total, 8);

    clReleaseEvent(c->event);
    c->in_flight = 0;
    c->max_iters = adapt_iters(c->max_iters, elapsed, KP_ITERS_MIN, KP_ITERS_MAX);
}

void gpu_doppler_destroy(void *opaque) {
    DopplerCtx *c = (DopplerCtx *)opaque;
    clFinish(c->queue);
    if (c->in_flight) clReleaseEvent(c->event);
    cl_mem bufs[] = {c->seed,c->out,c->done,c->counts,c->comb};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; ++i) clReleaseMemObject(bufs[i]);
    clReleaseKernel(c->kernel);
    clReleaseProgram(c->program);
    clReleaseCommandQueue(c->queue);
    clReleaseContext(c->context);
    free(c->counts_host);
    free(c);
}
