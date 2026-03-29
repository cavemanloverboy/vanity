#ifndef VANITY_KEYPAIR_H
#define VANITY_KEYPAIR_H

#include <stdint.h>

extern "C" void vanity_keypair_round(
    int gpu_id,
    uint8_t *seed,
    char *prefix,
    char *suffix,
    uint64_t prefix_len,
    uint64_t suffix_len,
    uint8_t *out,
    bool case_insensitive);

#endif
