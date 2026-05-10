#ifndef BS58_H
#define BS58_H

#include <stdint.h>

__device__ ulong fd_base58_encode_32(uint8_t *bytes, uint8_t *out, bool case_insensitive);

__device__ bool fd_base58_check_match_32(uint8_t *bytes,
                                         const uint8_t *target,
                                         ulong target_len,
                                         const uint8_t *suffix,
                                         ulong suffix_len,
                                         bool case_insensitive);

#endif