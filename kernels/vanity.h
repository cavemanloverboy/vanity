#ifndef VANITY_H
#define VANITY_H

#include <stdint.h>
#include "utils.h"

extern "C" void* gpu_grind_init(int id, uint8_t *base, uint8_t *owner, uint8_t *patterns, uint64_t patterns_len, bool case_insensitive);
extern "C" void  gpu_grind_launch(void *ctx, uint8_t *seed);
extern "C" int   gpu_grind_query(void *ctx);
extern "C" void  gpu_grind_read(void *ctx, uint8_t *out);
extern "C" void  gpu_grind_destroy(void *ctx);
extern "C" void  gpu_grind_set_active_mask(void *ctx, unsigned long long mask);

#endif
