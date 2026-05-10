#ifndef BS58_H
#define BS58_H

#include <stdint.h>

__device__ ulong fd_base58_encode_32(uint8_t *bytes, uint8_t *out, bool case_insensitive);

/* target/suffix are canonical raw_base58 indices (0..57) precomputed
   on the host. d_match_lut (defined in base58.cu, populated via
   gpu_grind_init) folds raw_base58 values into the same canonical
   space, so the same comparison path handles both case-sensitive and
   case-insensitive modes. */
__device__ bool fd_base58_check_match_32(uint8_t *bytes,
                                         const uint8_t *target,
                                         ulong target_len,
                                         const uint8_t *suffix,
                                         ulong suffix_len);

#endif