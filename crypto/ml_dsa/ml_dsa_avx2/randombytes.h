#ifndef RANDOMBYTES_H
#define RANDOMBYTES_H

#include <stddef.h>
#include <stdint.h>

#define ML_DSA_AVX2_NAMESPACE(s) pqcrystals_ml_dsa_avx2##s
#define randombytes ML_DSA_AVX2_NAMESPACE(_randombytes)

void randombytes(uint8_t *out, size_t outlen);

#endif
