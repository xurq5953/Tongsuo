#ifndef SIGN_H
#define SIGN_H

#include <stddef.h>
#include <stdint.h>
#include "params.h"
#include "polyvec.h"
#include "poly.h"

void challenge(poly *c, const uint8_t seed[SEEDBYTES]);

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
