#ifndef VANITY_KEYPAIR_H
#define VANITY_KEYPAIR_H

#include <stdint.h>

extern "C" void* gpu_keypair_init(int id, uint8_t *prefixes, uint64_t prefix_count, uint8_t *suffixes, uint64_t suffix_count, bool case_insensitive);
extern "C" void  gpu_keypair_launch(void *ctx, uint8_t *seed);
extern "C" int   gpu_keypair_query(void *ctx);
extern "C" void  gpu_keypair_read(void *ctx, uint8_t *out);
extern "C" void  gpu_keypair_destroy(void *ctx);

/* "doppler" keypair grind: matches ed25519 pubkeys with at least
   required_segments sign-extendable 32-bit segments (see kernels/opencl/
   doppler.cl for the pattern). Defined in vanity_keypair.cu to share the
   ed25519 translation unit. */
extern "C" void* gpu_doppler_init(int id, uint32_t required_segments);
extern "C" void  gpu_doppler_launch(void *ctx, uint8_t *seed);
extern "C" int   gpu_doppler_query(void *ctx);
extern "C" void  gpu_doppler_read(void *ctx, uint8_t *out);
extern "C" void  gpu_doppler_destroy(void *ctx);

#endif
