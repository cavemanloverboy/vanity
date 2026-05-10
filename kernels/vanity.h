#ifndef VANITY_H
#define VANITY_H

#include <stdint.h>
#include "utils.h"

extern "C" void* gpu_grind_init(int id, uint8_t *base, uint8_t *owner, uint8_t *target, uint64_t target_len, uint8_t *suffix, uint64_t suffix_len, bool case_insensitive);
extern "C" void  gpu_grind_launch(void *ctx, uint8_t *seed);
extern "C" int   gpu_grind_query(void *ctx);
extern "C" void  gpu_grind_read(void *ctx, uint8_t *out);
extern "C" void  gpu_grind_destroy(void *ctx);

__global__ void vanity_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles);

#endif
