/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>

#include "sha256.h"

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32 // SHA256 outputs a 32 byte digest

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#endif

#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))

#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/

/*********************** FUNCTION DEFINITIONS ***********************/

// Unrolled one-block compressor (explicit W-schedule + 64 rounds).
__device__ void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE data[])
{
    WORD a, b, c, d, e, f, g, h, t1, t2;
    WORD m[64];

    m[0] = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | (data[3]);
    m[1] = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | (data[7]);
    m[2] = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | (data[11]);
    m[3] = (data[12] << 24) | (data[13] << 16) | (data[14] << 8) | (data[15]);
    m[4] = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | (data[19]);
    m[5] = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | (data[23]);
    m[6] = (data[24] << 24) | (data[25] << 16) | (data[26] << 8) | (data[27]);
    m[7] = (data[28] << 24) | (data[29] << 16) | (data[30] << 8) | (data[31]);
    m[8] = (data[32] << 24) | (data[33] << 16) | (data[34] << 8) | (data[35]);
    m[9] = (data[36] << 24) | (data[37] << 16) | (data[38] << 8) | (data[39]);
    m[10] = (data[40] << 24) | (data[41] << 16) | (data[42] << 8) | (data[43]);
    m[11] = (data[44] << 24) | (data[45] << 16) | (data[46] << 8) | (data[47]);
    m[12] = (data[48] << 24) | (data[49] << 16) | (data[50] << 8) | (data[51]);
    m[13] = (data[52] << 24) | (data[53] << 16) | (data[54] << 8) | (data[55]);
    m[14] = (data[56] << 24) | (data[57] << 16) | (data[58] << 8) | (data[59]);
    m[15] = (data[60] << 24) | (data[61] << 16) | (data[62] << 8) | (data[63]);

    m[16] = SIG1(m[14]) + m[9] + SIG0(m[1]) + m[0];
    m[17] = SIG1(m[15]) + m[10] + SIG0(m[2]) + m[1];
    m[18] = SIG1(m[16]) + m[11] + SIG0(m[3]) + m[2];
    m[19] = SIG1(m[17]) + m[12] + SIG0(m[4]) + m[3];
    m[20] = SIG1(m[18]) + m[13] + SIG0(m[5]) + m[4];
    m[21] = SIG1(m[19]) + m[14] + SIG0(m[6]) + m[5];
    m[22] = SIG1(m[20]) + m[15] + SIG0(m[7]) + m[6];
    m[23] = SIG1(m[21]) + m[16] + SIG0(m[8]) + m[7];
    m[24] = SIG1(m[22]) + m[17] + SIG0(m[9]) + m[8];
    m[25] = SIG1(m[23]) + m[18] + SIG0(m[10]) + m[9];
    m[26] = SIG1(m[24]) + m[19] + SIG0(m[11]) + m[10];
    m[27] = SIG1(m[25]) + m[20] + SIG0(m[12]) + m[11];
    m[28] = SIG1(m[26]) + m[21] + SIG0(m[13]) + m[12];
    m[29] = SIG1(m[27]) + m[22] + SIG0(m[14]) + m[13];
    m[30] = SIG1(m[28]) + m[23] + SIG0(m[15]) + m[14];
    m[31] = SIG1(m[29]) + m[24] + SIG0(m[16]) + m[15];
    m[32] = SIG1(m[30]) + m[25] + SIG0(m[17]) + m[16];
    m[33] = SIG1(m[31]) + m[26] + SIG0(m[18]) + m[17];
    m[34] = SIG1(m[32]) + m[27] + SIG0(m[19]) + m[18];
    m[35] = SIG1(m[33]) + m[28] + SIG0(m[20]) + m[19];
    m[36] = SIG1(m[34]) + m[29] + SIG0(m[21]) + m[20];
    m[37] = SIG1(m[35]) + m[30] + SIG0(m[22]) + m[21];
    m[38] = SIG1(m[36]) + m[31] + SIG0(m[23]) + m[22];
    m[39] = SIG1(m[37]) + m[32] + SIG0(m[24]) + m[23];
    m[40] = SIG1(m[38]) + m[33] + SIG0(m[25]) + m[24];
    m[41] = SIG1(m[39]) + m[34] + SIG0(m[26]) + m[25];
    m[42] = SIG1(m[40]) + m[35] + SIG0(m[27]) + m[26];
    m[43] = SIG1(m[41]) + m[36] + SIG0(m[28]) + m[27];
    m[44] = SIG1(m[42]) + m[37] + SIG0(m[29]) + m[28];
    m[45] = SIG1(m[43]) + m[38] + SIG0(m[30]) + m[29];
    m[46] = SIG1(m[44]) + m[39] + SIG0(m[31]) + m[30];
    m[47] = SIG1(m[45]) + m[40] + SIG0(m[32]) + m[31];
    m[48] = SIG1(m[46]) + m[41] + SIG0(m[33]) + m[32];
    m[49] = SIG1(m[47]) + m[42] + SIG0(m[34]) + m[33];
    m[50] = SIG1(m[48]) + m[43] + SIG0(m[35]) + m[34];
    m[51] = SIG1(m[49]) + m[44] + SIG0(m[36]) + m[35];
    m[52] = SIG1(m[50]) + m[45] + SIG0(m[37]) + m[36];
    m[53] = SIG1(m[51]) + m[46] + SIG0(m[38]) + m[37];
    m[54] = SIG1(m[52]) + m[47] + SIG0(m[39]) + m[38];
    m[55] = SIG1(m[53]) + m[48] + SIG0(m[40]) + m[39];
    m[56] = SIG1(m[54]) + m[49] + SIG0(m[41]) + m[40];
    m[57] = SIG1(m[55]) + m[50] + SIG0(m[42]) + m[41];
    m[58] = SIG1(m[56]) + m[51] + SIG0(m[43]) + m[42];
    m[59] = SIG1(m[57]) + m[52] + SIG0(m[44]) + m[43];
    m[60] = SIG1(m[58]) + m[53] + SIG0(m[45]) + m[44];
    m[61] = SIG1(m[59]) + m[54] + SIG0(m[46]) + m[45];
    m[62] = SIG1(m[60]) + m[55] + SIG0(m[47]) + m[46];
    m[63] = SIG1(m[61]) + m[56] + SIG0(m[48]) + m[47];

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    t1 = h + EP1(e) + CH(e, f, g) + 0x428A2F98U + m[0];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x71374491U + m[1];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xB5C0FBCFU + m[2];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xE9B5DBA5U + m[3];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x3956C25BU + m[4];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x59F111F1U + m[5];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x923F82A4U + m[6];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xAB1C5ED5U + m[7];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xD807AA98U + m[8];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x12835B01U + m[9];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x243185BEU + m[10];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x550C7DC3U + m[11];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x72BE5D74U + m[12];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x80DEB1FEU + m[13];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x9BDC06A7U + m[14];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xC19BF174U + m[15];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xE49B69C1U + m[16];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xEFBE4786U + m[17];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x0FC19DC6U + m[18];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x240CA1CCU + m[19];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x2DE92C6FU + m[20];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x4A7484AAU + m[21];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x5CB0A9DCU + m[22];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x76F988DAU + m[23];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x983E5152U + m[24];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xA831C66DU + m[25];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xB00327C8U + m[26];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xBF597FC7U + m[27];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xC6E00BF3U + m[28];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xD5A79147U + m[29];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x06CA6351U + m[30];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x14292967U + m[31];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x27B70A85U + m[32];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x2E1B2138U + m[33];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x4D2C6DFCU + m[34];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x53380D13U + m[35];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x650A7354U + m[36];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x766A0ABBU + m[37];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x81C2C92EU + m[38];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x92722C85U + m[39];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xA2BFE8A1U + m[40];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xA81A664BU + m[41];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xC24B8B70U + m[42];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xC76C51A3U + m[43];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xD192E819U + m[44];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xD6990624U + m[45];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xF40E3585U + m[46];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x106AA070U + m[47];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x19A4C116U + m[48];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x1E376C08U + m[49];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x2748774CU + m[50];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x34B0BCB5U + m[51];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x391C0CB3U + m[52];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x4ED8AA4AU + m[53];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x5B9CCA4FU + m[54];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x682E6FF3U + m[55];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x748F82EEU + m[56];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x78A5636FU + m[57];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x84C87814U + m[58];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x8CC70208U + m[59];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0x90BEFFFAU + m[60];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xA4506CEBU + m[61];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xBEF9A3F7U + m[62];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    t1 = h + EP1(e) + CH(e, f, g) + 0xC67178F2U + m[63];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

__device__ void cuda_sha256_expand_w(const BYTE data[64], WORD W[64])
{
    W[0] = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | (data[3]);
    W[1] = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | (data[7]);
    W[2] = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | (data[11]);
    W[3] = (data[12] << 24) | (data[13] << 16) | (data[14] << 8) | (data[15]);
    W[4] = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | (data[19]);
    W[5] = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | (data[23]);
    W[6] = (data[24] << 24) | (data[25] << 16) | (data[26] << 8) | (data[27]);
    W[7] = (data[28] << 24) | (data[29] << 16) | (data[30] << 8) | (data[31]);
    W[8] = (data[32] << 24) | (data[33] << 16) | (data[34] << 8) | (data[35]);
    W[9] = (data[36] << 24) | (data[37] << 16) | (data[38] << 8) | (data[39]);
    W[10] = (data[40] << 24) | (data[41] << 16) | (data[42] << 8) | (data[43]);
    W[11] = (data[44] << 24) | (data[45] << 16) | (data[46] << 8) | (data[47]);
    W[12] = (data[48] << 24) | (data[49] << 16) | (data[50] << 8) | (data[51]);
    W[13] = (data[52] << 24) | (data[53] << 16) | (data[54] << 8) | (data[55]);
    W[14] = (data[56] << 24) | (data[57] << 16) | (data[58] << 8) | (data[59]);
    W[15] = (data[60] << 24) | (data[61] << 16) | (data[62] << 8) | (data[63]);
    W[16] = SIG1(W[14]) + W[9] + SIG0(W[1]) + W[0];
    W[17] = SIG1(W[15]) + W[10] + SIG0(W[2]) + W[1];
    W[18] = SIG1(W[16]) + W[11] + SIG0(W[3]) + W[2];
    W[19] = SIG1(W[17]) + W[12] + SIG0(W[4]) + W[3];
    W[20] = SIG1(W[18]) + W[13] + SIG0(W[5]) + W[4];
    W[21] = SIG1(W[19]) + W[14] + SIG0(W[6]) + W[5];
    W[22] = SIG1(W[20]) + W[15] + SIG0(W[7]) + W[6];
    W[23] = SIG1(W[21]) + W[16] + SIG0(W[8]) + W[7];
    W[24] = SIG1(W[22]) + W[17] + SIG0(W[9]) + W[8];
    W[25] = SIG1(W[23]) + W[18] + SIG0(W[10]) + W[9];
    W[26] = SIG1(W[24]) + W[19] + SIG0(W[11]) + W[10];
    W[27] = SIG1(W[25]) + W[20] + SIG0(W[12]) + W[11];
    W[28] = SIG1(W[26]) + W[21] + SIG0(W[13]) + W[12];
    W[29] = SIG1(W[27]) + W[22] + SIG0(W[14]) + W[13];
    W[30] = SIG1(W[28]) + W[23] + SIG0(W[15]) + W[14];
    W[31] = SIG1(W[29]) + W[24] + SIG0(W[16]) + W[15];
    W[32] = SIG1(W[30]) + W[25] + SIG0(W[17]) + W[16];
    W[33] = SIG1(W[31]) + W[26] + SIG0(W[18]) + W[17];
    W[34] = SIG1(W[32]) + W[27] + SIG0(W[19]) + W[18];
    W[35] = SIG1(W[33]) + W[28] + SIG0(W[20]) + W[19];
    W[36] = SIG1(W[34]) + W[29] + SIG0(W[21]) + W[20];
    W[37] = SIG1(W[35]) + W[30] + SIG0(W[22]) + W[21];
    W[38] = SIG1(W[36]) + W[31] + SIG0(W[23]) + W[22];
    W[39] = SIG1(W[37]) + W[32] + SIG0(W[24]) + W[23];
    W[40] = SIG1(W[38]) + W[33] + SIG0(W[25]) + W[24];
    W[41] = SIG1(W[39]) + W[34] + SIG0(W[26]) + W[25];
    W[42] = SIG1(W[40]) + W[35] + SIG0(W[27]) + W[26];
    W[43] = SIG1(W[41]) + W[36] + SIG0(W[28]) + W[27];
    W[44] = SIG1(W[42]) + W[37] + SIG0(W[29]) + W[28];
    W[45] = SIG1(W[43]) + W[38] + SIG0(W[30]) + W[29];
    W[46] = SIG1(W[44]) + W[39] + SIG0(W[31]) + W[30];
    W[47] = SIG1(W[45]) + W[40] + SIG0(W[32]) + W[31];
    W[48] = SIG1(W[46]) + W[41] + SIG0(W[33]) + W[32];
    W[49] = SIG1(W[47]) + W[42] + SIG0(W[34]) + W[33];
    W[50] = SIG1(W[48]) + W[43] + SIG0(W[35]) + W[34];
    W[51] = SIG1(W[49]) + W[44] + SIG0(W[36]) + W[35];
    W[52] = SIG1(W[50]) + W[45] + SIG0(W[37]) + W[36];
    W[53] = SIG1(W[51]) + W[46] + SIG0(W[38]) + W[37];
    W[54] = SIG1(W[52]) + W[47] + SIG0(W[39]) + W[38];
    W[55] = SIG1(W[53]) + W[48] + SIG0(W[40]) + W[39];
    W[56] = SIG1(W[54]) + W[49] + SIG0(W[41]) + W[40];
    W[57] = SIG1(W[55]) + W[50] + SIG0(W[42]) + W[41];
    W[58] = SIG1(W[56]) + W[51] + SIG0(W[43]) + W[42];
    W[59] = SIG1(W[57]) + W[52] + SIG0(W[44]) + W[43];
    W[60] = SIG1(W[58]) + W[53] + SIG0(W[45]) + W[44];
    W[61] = SIG1(W[59]) + W[54] + SIG0(W[46]) + W[45];
    W[62] = SIG1(W[60]) + W[55] + SIG0(W[47]) + W[46];
    W[63] = SIG1(W[61]) + W[56] + SIG0(W[48]) + W[47];
}

/* 64 main rounds against an externally-provided W[0..63]. Identical
   semantics to cuda_sha256_transform's compression phase. */
__device__ void cuda_sha256_transform_w(WORD state[8], const WORD W[64])
{
    WORD a = state[0], b = state[1], c = state[2], d = state[3];
    WORD e = state[4], f = state[5], g = state[6], h = state[7];
    WORD t1, t2;

#define VANITY_SHA_R(j, k_const)                              \
    do {                                                      \
        t1 = h + EP1(e) + CH(e, f, g) + (k_const) + W[(j)];   \
        t2 = EP0(a) + MAJ(a, b, c);                           \
        h = g; g = f; f = e;                                  \
        e = d + t1;                                           \
        d = c; c = b; b = a;                                  \
        a = t1 + t2;                                          \
    } while (0)

    VANITY_SHA_R( 0, 0x428A2F98U); VANITY_SHA_R( 1, 0x71374491U);
    VANITY_SHA_R( 2, 0xB5C0FBCFU); VANITY_SHA_R( 3, 0xE9B5DBA5U);
    VANITY_SHA_R( 4, 0x3956C25BU); VANITY_SHA_R( 5, 0x59F111F1U);
    VANITY_SHA_R( 6, 0x923F82A4U); VANITY_SHA_R( 7, 0xAB1C5ED5U);
    VANITY_SHA_R( 8, 0xD807AA98U); VANITY_SHA_R( 9, 0x12835B01U);
    VANITY_SHA_R(10, 0x243185BEU); VANITY_SHA_R(11, 0x550C7DC3U);
    VANITY_SHA_R(12, 0x72BE5D74U); VANITY_SHA_R(13, 0x80DEB1FEU);
    VANITY_SHA_R(14, 0x9BDC06A7U); VANITY_SHA_R(15, 0xC19BF174U);
    VANITY_SHA_R(16, 0xE49B69C1U); VANITY_SHA_R(17, 0xEFBE4786U);
    VANITY_SHA_R(18, 0x0FC19DC6U); VANITY_SHA_R(19, 0x240CA1CCU);
    VANITY_SHA_R(20, 0x2DE92C6FU); VANITY_SHA_R(21, 0x4A7484AAU);
    VANITY_SHA_R(22, 0x5CB0A9DCU); VANITY_SHA_R(23, 0x76F988DAU);
    VANITY_SHA_R(24, 0x983E5152U); VANITY_SHA_R(25, 0xA831C66DU);
    VANITY_SHA_R(26, 0xB00327C8U); VANITY_SHA_R(27, 0xBF597FC7U);
    VANITY_SHA_R(28, 0xC6E00BF3U); VANITY_SHA_R(29, 0xD5A79147U);
    VANITY_SHA_R(30, 0x06CA6351U); VANITY_SHA_R(31, 0x14292967U);
    VANITY_SHA_R(32, 0x27B70A85U); VANITY_SHA_R(33, 0x2E1B2138U);
    VANITY_SHA_R(34, 0x4D2C6DFCU); VANITY_SHA_R(35, 0x53380D13U);
    VANITY_SHA_R(36, 0x650A7354U); VANITY_SHA_R(37, 0x766A0ABBU);
    VANITY_SHA_R(38, 0x81C2C92EU); VANITY_SHA_R(39, 0x92722C85U);
    VANITY_SHA_R(40, 0xA2BFE8A1U); VANITY_SHA_R(41, 0xA81A664BU);
    VANITY_SHA_R(42, 0xC24B8B70U); VANITY_SHA_R(43, 0xC76C51A3U);
    VANITY_SHA_R(44, 0xD192E819U); VANITY_SHA_R(45, 0xD6990624U);
    VANITY_SHA_R(46, 0xF40E3585U); VANITY_SHA_R(47, 0x106AA070U);
    VANITY_SHA_R(48, 0x19A4C116U); VANITY_SHA_R(49, 0x1E376C08U);
    VANITY_SHA_R(50, 0x2748774CU); VANITY_SHA_R(51, 0x34B0BCB5U);
    VANITY_SHA_R(52, 0x391C0CB3U); VANITY_SHA_R(53, 0x4ED8AA4AU);
    VANITY_SHA_R(54, 0x5B9CCA4FU); VANITY_SHA_R(55, 0x682E6FF3U);
    VANITY_SHA_R(56, 0x748F82EEU); VANITY_SHA_R(57, 0x78A5636FU);
    VANITY_SHA_R(58, 0x84C87814U); VANITY_SHA_R(59, 0x8CC70208U);
    VANITY_SHA_R(60, 0x90BEFFFAU); VANITY_SHA_R(61, 0xA4506CEBU);
    VANITY_SHA_R(62, 0xBEF9A3F7U); VANITY_SHA_R(63, 0xC67178F2U);

#undef VANITY_SHA_R

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len)
{
    WORD i;

    for (i = 0; i < len; ++i)
    {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64)
        {
            cuda_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[])
{
    WORD i;

    i = ctx->datalen;

    // Pad whatever data is left in the buffer.
    if (ctx->datalen < 56)
    {
        ctx->data[i++] = 0x80;
        while (i < 56)
            ctx->data[i++] = 0x00;
    }
    else
    {
        ctx->data[i++] = 0x80;
        while (i < 64)
            ctx->data[i++] = 0x00;
        cuda_sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    // Append to the padding the total message's length in bits and transform.
    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[56] = ctx->bitlen >> 56;
    cuda_sha256_transform(ctx, ctx->data);

    /* Big-endian emission of digest words */
    hash[0] = (ctx->state[0] >> 24) & 0x000000ff;
    hash[1] = (ctx->state[0] >> 16) & 0x000000ff;
    hash[2] = (ctx->state[0] >> 8) & 0x000000ff;
    hash[3] = (ctx->state[0] >> 0) & 0x000000ff;
    hash[4] = (ctx->state[1] >> 24) & 0x000000ff;
    hash[5] = (ctx->state[1] >> 16) & 0x000000ff;
    hash[6] = (ctx->state[1] >> 8) & 0x000000ff;
    hash[7] = (ctx->state[1] >> 0) & 0x000000ff;
    hash[8] = (ctx->state[2] >> 24) & 0x000000ff;
    hash[9] = (ctx->state[2] >> 16) & 0x000000ff;
    hash[10] = (ctx->state[2] >> 8) & 0x000000ff;
    hash[11] = (ctx->state[2] >> 0) & 0x000000ff;
    hash[12] = (ctx->state[3] >> 24) & 0x000000ff;
    hash[13] = (ctx->state[3] >> 16) & 0x000000ff;
    hash[14] = (ctx->state[3] >> 8) & 0x000000ff;
    hash[15] = (ctx->state[3] >> 0) & 0x000000ff;
    hash[16] = (ctx->state[4] >> 24) & 0x000000ff;
    hash[17] = (ctx->state[4] >> 16) & 0x000000ff;
    hash[18] = (ctx->state[4] >> 8) & 0x000000ff;
    hash[19] = (ctx->state[4] >> 0) & 0x000000ff;
    hash[20] = (ctx->state[5] >> 24) & 0x000000ff;
    hash[21] = (ctx->state[5] >> 16) & 0x000000ff;
    hash[22] = (ctx->state[5] >> 8) & 0x000000ff;
    hash[23] = (ctx->state[5] >> 0) & 0x000000ff;
    hash[24] = (ctx->state[6] >> 24) & 0x000000ff;
    hash[25] = (ctx->state[6] >> 16) & 0x000000ff;
    hash[26] = (ctx->state[6] >> 8) & 0x000000ff;
    hash[27] = (ctx->state[6] >> 0) & 0x000000ff;
    hash[28] = (ctx->state[7] >> 24) & 0x000000ff;
    hash[29] = (ctx->state[7] >> 16) & 0x000000ff;
    hash[30] = (ctx->state[7] >> 8) & 0x000000ff;
    hash[31] = (ctx->state[7] >> 0) & 0x000000ff;
}