#ifndef BS58_H
#define BS58_H

#include <stdint.h>

__device__ void fd_base58_encode_32(uint8_t *bytes, uint8_t *out, bool case_insensitive);

#endif