#ifndef ED25519_SHA512_H
#define ED25519_SHA512_H

#include <stddef.h>
#include "fixedint.h"

typedef struct {
    uint64_t length, state[8];
    size_t curlen;
    unsigned char buf[128];
} sha512_context;

__device__ int sha512_init(sha512_context *md);
__device__ int sha512_final(sha512_context *md, unsigned char *out);
__device__ int sha512(const unsigned char *message, size_t message_len, unsigned char *out);

#endif
