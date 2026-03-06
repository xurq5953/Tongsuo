#ifndef API_H
#define API_H

#include "config.h"
#include <stddef.h>
#include <stdint.h>

#if DILITHIUM_MODE == 2
#define CRYPTO_PUBLICKEYBYTES 1312
#define CRYPTO_SECRETKEYBYTES 2560
#define CRYPTO_BYTES 2420

#elif DILITHIUM_MODE == 3
#define CRYPTO_PUBLICKEYBYTES 1952
#define CRYPTO_SECRETKEYBYTES 4032
#define CRYPTO_BYTES 3309

#elif DILITHIUM_MODE == 5
#define CRYPTO_PUBLICKEYBYTES 2592
#define CRYPTO_SECRETKEYBYTES 4896
#define CRYPTO_BYTES 4627
#endif

int pqcrystals_ml_dsa_65_keypair(uint8_t *pk, uint8_t *sk,
                                 uint8_t *seed, int rand_seed);

int pqcrystals_ml_dsa_65_signature_internal(uint8_t *sig,
                                            size_t *siglen,
                                            const uint8_t *mu,
                                            const uint8_t rnd[32],
                                            const uint8_t *sk);

int pqcrystals_ml_dsa_65_signature(uint8_t *sig, size_t *siglen,
                                   const uint8_t *m, size_t mlen,
                                   const uint8_t *ctx, size_t ctxlen,
                                   const int deterministic,
                                   const uint8_t *sk);

int pqcrystals_ml_dsa_65(uint8_t *sm, size_t *smlen,
                         const uint8_t *m, size_t mlen,
                         const uint8_t *ctx, size_t ctxlen,
                         const uint8_t *sk);

int pqcrystals_ml_dsa_65_verify_internal(const uint8_t *sig,
                                         size_t siglen,
                                         const uint8_t *mu,
                                         const uint8_t *pk);

int pqcrystals_ml_dsa_65_verify(const uint8_t *sig, size_t siglen,
                                const uint8_t *m, size_t mlen,
                                const uint8_t *ctx, size_t ctxlen,
                                const uint8_t *pk);

int pqcrystals_ml_dsa_65_open(uint8_t *m, size_t *mlen,
                              const uint8_t *sm, size_t smlen,
                              const uint8_t *ctx, size_t ctxlen,
                              const uint8_t *pk);

#endif
