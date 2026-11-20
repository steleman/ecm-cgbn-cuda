/* gpu_ntt.h - the small-prime NTTs of stage 2 on an NVIDIA GPU.

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

/* The vectors of residues of stage 2 (mpzspv_t) are allocated as one block
   of CUDA managed memory (sp_num rows of len words), so the CPU code that
   is not ported keeps working on them, and the NTT products (forward DIF,
   pointwise product, inverse DIT) run on the GPU. They give the same words
   as the CPU code: the same butterflies, with the same roots of unity.

   ECM_GPU_NTT=0 in the environment turns this off (CPU only).
   ECM_GPU_NTT_CHECK=1 runs the CPU code on copies after each GPU operation
   and compares every word (slow; aborts on a difference). */

#ifndef _GPU_NTT_H
#define _GPU_NTT_H 1

#include "sp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if the GPU NTT is in use (built with GPU support, a device, and not
   turned off with ECM_GPU_NTT=0). */
int gpu_ntt_active (void);

/* sp_num rows of len words in one block of managed memory: x[i] = x[0] +
   i * len. NULL if the GPU NTT is not in use or the allocation failed. */
mpzspv_t gpu_mpzspv_init (spv_size_t len, mpzspm_t mpzspm);
/* 1 and frees x if gpu_mpzspv_init made it, else 0. */
int gpu_mpzspv_clear (mpzspv_t x, mpzspm_t mpzspm);

/* Device tables of an mpzspm (primes, reciprocals, roots of unity).
   Called by mpzspm_init / mpzspm_clear; mpzspm->gpu stays NULL if the GPU
   NTT is not in use. */
void gpu_mpzspm_init (mpzspm_t mpzspm);
void gpu_mpzspm_clear (mpzspm_t mpzspm);

/* As mpzspv_mul_ntt / mpzspv_to_ntt. Return 1 if done on the GPU, 0 if
   the caller must do it on the CPU (vectors not from gpu_mpzspv_init, or
   the GPU NTT not in use). */
int gpu_mpzspv_mul_ntt (mpzspv_t r, spv_size_t offsetr,
    mpzspv_t x, spv_size_t offsetx, spv_size_t lenx,
    mpzspv_t y, spv_size_t offsety, spv_size_t leny,
    spv_size_t ntt_size, int monic, spv_size_t monic_pos,
    mpzspm_t mpzspm, int steps);
int gpu_mpzspv_to_ntt (mpzspv_t x, spv_size_t offset, spv_size_t len,
    spv_size_t ntt_size, int monic, mpzspm_t mpzspm);
/* As mpzspv_from_mpzv / mpzspv_to_mpzv / mpzspv_normalise, same results
   word for word; 1 if done on the GPU, 0 if the caller must do it. */
int gpu_mpzspv_from_mpzv (mpzspv_t x, spv_size_t offset, const mpzv_t mpzv,
    spv_size_t len, mpzspm_t mpzspm);
int gpu_mpzspv_to_mpzv (mpzspv_t x, spv_size_t offset, mpzv_t mpzv,
    spv_size_t len, mpzspm_t mpzspm);
int gpu_mpzspv_normalise (mpzspv_t x, spv_size_t offset, spv_size_t len,
    mpzspm_t mpzspm);

/* r = prod_k (x + a[k]) (monic, the leading 1 not stored), len a power
   of 2, all on the GPU, the levels of the product tree in Tree if Tree is
   not NULL (as ntt_PolyFromRoots_Tree with dolvl = -1: Tree[lgk - 1] the
   roots, ..., Tree[0] the products of len/4 roots). The coefficients are
   correct mod N, not reduced. 1 if done, 0 if the caller must do it. */
int gpu_poly_from_roots (mpzv_t r, mpzv_t a, spv_size_t len, mpzv_t *Tree,
    mpzspm_t mpzspm);

/* The levels of ntt_polyevalT (the transposed product tree), all on the
   GPU: y (from gpu_mpzspv_init) holds rev(high(b * invF)) in its first
   len words; Tree as made by ntt_PolyFromRoots_Tree (not modified);
   T[0..len-1] receives the values (correct mod N, not reduced); y is
   destroyed. 1 if done, 0 if the caller must do it. */
int gpu_polyeval_levels (mpzspv_t y, spv_size_t len, mpzv_t *Tree,
    mpzv_t T, mpzspm_t mpzspm);

/* Batched stage 2 of C curves (same N and stage-2 parameters), all in
   residues on the GPU: gpu_s2batch_F with the C dF roots of F (device,
   curve-major, W 64-bit words each, as gpu_roots_batch_run gives them)
   builds F and its product tree and 1/F; gpu_s2batch_G with each block's
   roots of G makes H = prod (G - F) mod F (first: H = G - F);
   gpu_s2batch_finish evaluates H at the roots of F and puts the product
   of the values of each curve in prod[c] (correct mod N, not reduced):
   what stage2.c computes before its gcd. gpu_s2batch_bytes: the device
   memory it needs. NULL from gpu_s2batch_new if it cannot be had. */
typedef struct gpu_s2batch gpu_s2batch_t;
size_t gpu_s2batch_bytes (mpzspm_t mpzspm, unsigned C, unsigned long dF);
gpu_s2batch_t *gpu_s2batch_new (mpzspm_t mpzspm, unsigned C,
    unsigned long dF);
void gpu_s2batch_F (gpu_s2batch_t *b, const void *droots, unsigned W);
void gpu_s2batch_G (gpu_s2batch_t *b, const void *droots, unsigned W,
    int first);
void gpu_s2batch_finish (gpu_s2batch_t *b, mpz_t *prod);
void gpu_s2batch_free (gpu_s2batch_t *b);

/* Free device memory, in bytes. */
size_t gpu_free_memory (void);

/* 1 if ECM_GPU_NTT_CHECK=1 (and the GPU NTT is in use). */
int gpu_ntt_check (void);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_NTT_H */
