/* gpu_roots.h - the roots of F and G of ECM stage 2 on an NVIDIA GPU.

Copyright 2026 Stefan Teleman and contributors.

This file is part of the ECM Library.

The ECM Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The ECM Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the ECM Library; see the file COPYING.LIB.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

/* The tables of finite differences of ecm_rootsF / ecm_rootsG (ecm2.c),
   nr progressions of S + 1 points each, kept on the GPU in projective
   coordinates on y^2 = x^3 + A x + b, with the complete addition formulas
   of Renes, Costello and Batina (2016): no inversion per addition, no
   special case. The roots (x-coordinates of the first point of each
   progression) are made affine at the end with batched inversions, so
   they are the same integers as ecm2.c's.

   Every addition also multiplies into an accumulator the value whose
   inverse ecm2.c needs (x2 - x1, or 2y for a doubling, in projective
   form), so a product that is not invertible mod N, which is when ecm2.c
   finds a factor while computing roots, is seen too: the caller then does
   that part on the CPU, from the table at its start, with ecm2.c's code.

   The modulus may have up to GPU_ROOTS_MAX_BITS bits. */

#ifndef _GPU_ROOTS_H
#define _GPU_ROOTS_H 1

#include <gmp.h>

#define GPU_ROOTS_MAX_BITS 4096

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_roots gpu_roots_t;

/* 1 if the GPU roots are in use: a device, ECM_GPU_ROOTS not 0, and n of at
   most GPU_ROOTS_MAX_BITS bits. */
int gpu_roots_active (const mpz_t n);

/* The table R[i] = coeffs[i] * P, i < nr * (S + 1) (as multiplyW2n), on
   the curve y^2 = x^3 + A x + b through P = (px, py), mod n; then, as
   ecm2.c does, the last point of progression 0 copied to the last point
   of every other progression. All integers are canonical (in [0, n)),
   coeffs may be negative. NULL if not done (no device, or an
   accumulator not invertible: *bad = 1). */
gpu_roots_t *gpu_roots_init (const mpz_t n, const mpz_t A, const mpz_t px,
                             const mpz_t py, mpz_t *coeffs, unsigned nr,
                             unsigned S, int *bad);

/* Runs a schedule: sched[k] = -1 for an update of the whole table (every
   progression advanced by one step), p >= 0 for "the next root is the
   first point of progression p". The nout = number of p >= 0 roots go to
   out[0..nout-1], canonical. Returns 0, or 1 if an accumulator or a root
   was not invertible (out not written; the table is back as it was before
   the call, see gpu_roots_table). */
int gpu_roots_run (gpu_roots_t *g, const int *sched, unsigned long nsched,
                   mpz_t *out, unsigned long nout);

/* The table as affine points (x[i], y[i]), canonical, the neutral element
   as (0, 0), as ecm2.c keeps it: for the CPU code to take over. Returns 0,
   or 1 if a point could not be made affine: then f is a divisor of n
   other than 1 (a factor, or n). */
int gpu_roots_table (gpu_roots_t *g, mpz_t *x, mpz_t *y, mpz_t f);

void gpu_roots_free (gpu_roots_t *g);

/* The same for C curves at once (one table of C nr (S+1) points), for the
   batched stage 2: A[c], px[c], py[c] the curves, the coefficients the
   same for all. bad[c] (C bytes, not cleared) is set for a curve whose
   accumulator, or a root, is not invertible: its results are not to be
   used. gpu_roots_batch_run returns the device address of the C * nout
   roots (affine, canonical, curve-major: curve c's at c * nout), each
   gpu_roots_bits(g) / 64 words (least significant first), valid until the
   next run. Free with gpu_roots_free. */
gpu_roots_t *gpu_roots_batch_init (const mpz_t n, unsigned C, const mpz_t *A,
                                   const mpz_t *px, const mpz_t *py,
                                   mpz_t *coeffs, unsigned nr, unsigned S,
                                   unsigned char *bad);
const void *gpu_roots_batch_run (gpu_roots_t *g, const int *sched,
                                 unsigned long nsched, unsigned long nout,
                                 unsigned char *bad);
unsigned gpu_roots_bits (gpu_roots_t *g);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_ROOTS_H */
