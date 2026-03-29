#ifndef ED25519_GE_H
#define ED25519_GE_H

#include "fe.h"

typedef struct { fe X; fe Y; fe Z; } ge_p2;
typedef struct { fe X; fe Y; fe Z; fe T; } ge_p3;
typedef struct { fe X; fe Y; fe Z; fe T; } ge_p1p1;
typedef struct { fe yplusx; fe yminusx; fe xy2d; } ge_precomp;

__device__ void ge_p3_tobytes(unsigned char *s, const ge_p3 *h);
__device__ void ge_scalarmult_base(ge_p3 *h, const unsigned char *a);

#endif
