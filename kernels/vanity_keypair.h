#ifndef VANITY_KEYPAIR_H
#define VANITY_KEYPAIR_H

#include <stdint.h>

extern "C" void* gpu_keypair_init(int id, uint8_t *prefix, uint64_t prefix_len, uint8_t *suffix, uint64_t suffix_len, bool case_insensitive);
extern "C" void  gpu_keypair_launch(void *ctx, uint8_t *seed);
extern "C" int   gpu_keypair_query(void *ctx);
extern "C" void  gpu_keypair_read(void *ctx, uint8_t *out);
extern "C" void  gpu_keypair_destroy(void *ctx);

#endif
