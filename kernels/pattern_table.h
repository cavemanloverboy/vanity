#ifndef VANITY_PATTERN_TABLE_H
#define VANITY_PATTERN_TABLE_H

#include <stdint.h>
#include <string.h>

#define VANITY_MAX_PATTERNS 64
#define VANITY_MAX_PATTERN_LEN 44

/* Flat SoA uploaded to CUDA __constant__ / OpenCL __constant.
   Offsets are explicit so host and kernels agree (no struct padding). */
#define VANITY_PT_N      0
#define VANITY_PT_PLEN   4
#define VANITY_PT_SLEN   (4 + VANITY_MAX_PATTERNS)
#define VANITY_PT_PREF   (VANITY_PT_SLEN + VANITY_MAX_PATTERNS)
#define VANITY_PT_SUF    (VANITY_PT_PREF + VANITY_MAX_PATTERNS * VANITY_MAX_PATTERN_LEN)
#define VANITY_PT_SIZE   (VANITY_PT_SUF + VANITY_MAX_PATTERNS * VANITY_MAX_PATTERN_LEN)

/* Packed blob from MatchTargets::gpu_blob → flat SoA. */
static void vanity_unpack_patterns(const uint8_t *blob, uint64_t blob_len,
                                   uint8_t *out)
{
    memset(out, 0, VANITY_PT_SIZE);
    if (blob_len < 6) return;
    uint32_t n = 0;
    memcpy(&n, blob, 4);
    if (n > VANITY_MAX_PATTERNS) n = VANITY_MAX_PATTERNS;
    memcpy(out + VANITY_PT_N, &n, 4);
    const uint8_t *p = blob + 6;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t plen = *p++;
        out[VANITY_PT_PLEN + i] = plen;
        memcpy(out + VANITY_PT_PREF + i * VANITY_MAX_PATTERN_LEN, p, plen);
        p += plen;
        uint8_t slen = *p++;
        out[VANITY_PT_SLEN + i] = slen;
        memcpy(out + VANITY_PT_SUF + i * VANITY_MAX_PATTERN_LEN, p, slen);
        p += slen;
    }
}

#endif
