/* gpu_ntt.cu - the small-prime NTTs of stage 2 on an NVIDIA GPU.

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

/* See gpu_ntt.h. The transforms are those of ntt_gfp.c done stage by
   stage: the forward one (DIF) takes the data in natural order to
   bit-reversed order, the inverse one (DIT) back. A butterfly of a
   transform of length 2h uses the root of unity of order 2h of
   ntt_roots[] to the power i; ntt_gfp.c gets the same words, whether from
   its twiddle tables or computed on the fly, since every root of unity is
   a power of the largest one (nttdata_init squares it down). One table
   per prime, W[i] = r^i for i < K/2 (r of order K, the largest NTT
   size), serves every length: the twiddle of index i at half-length h is
   W[i * K / (2h)]. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <mutex>
#include <set>
#include <cuda_runtime.h>

#include "sp.h"
#include "gpu_ntt.h"

#if SP_NUMB_BITS != 62
#error "gpu_ntt.cu assumes 62-bit small primes (64-bit sp_t)"
#endif

/* ------------------------------------------------------------------ */
/* Arithmetic mod a small prime p < 2^62, as sp.h does it                */

__device__ __forceinline__ uint64_t
d_add (uint64_t a, uint64_t b, uint64_t p)
{
  uint64_t t = a + b;
  return t >= p ? t - p : t;
}

__device__ __forceinline__ uint64_t
d_sub (uint64_t a, uint64_t b, uint64_t p)
{
  return a >= b ? a - b : a - b + p;
}

/* sp_mul: x*y mod p with d = mul_c (sp_udiv_rem, small-modulus case). */
__device__ __forceinline__ uint64_t
d_mul (uint64_t x, uint64_t y, uint64_t p, uint64_t d)
{
  uint64_t nh = __umul64hi (x, y), nl = x * y;
  uint64_t q1 = (nh << 4) | (nl >> 60);
  uint64_t q2 = __umul64hi (q1, d);
  uint64_t r = nl - p * (q2 >> 1);
  return r >= p ? r - p : r;
}

__device__ uint64_t
d_pow (uint64_t x, uint64_t e, uint64_t p, uint64_t d)
{
  uint64_t r = 1;
  while (e)
    {
      if (e & 1)
        r = d_mul (r, x, p, d);
      e >>= 1;
      if (e)
        x = d_mul (x, x, p, d);
    }
  return r;
}

/* ------------------------------------------------------------------ */
/* Device tables of an mpzspm                                           */

struct gpu_ctx
{
  unsigned sp_num;
  unsigned logK;     /* largest NTT size K = 2^logK */
  size_t half;       /* K/2: words per prime in Wf, Wi */
  uint64_t *P, *D;   /* primes and their mul_c */
  uint64_t *Wf, *Wi; /* Wf[i*half + j] = r_i^j, Wi with the inverse root */
  /* The CRT tables of mpzspm.c, for the conversions and normalisation. */
  unsigned nl;       /* limbs of the modulus N */
  uint64_t *Bm;      /* 2^64 mod p_i */
  uint64_t *crt1;    /* crt1[i]: nl limbs each */
  uint64_t *crt2;    /* crt2[i], i < sp_num + 2: nl limbs each */
  uint64_t *crt3, *crt5;
  uint64_t *crt4;    /* crt4[i * sp_num + j] = mpzspm->crt4[i][j] */
};

/* ------------------------------------------------------------------ */
/* Kernels. Row i (prime p_i) of a vector is base + i * stride.         */
/*                                                                      */
/* Transforms come in batches: nt transforms of length L, transform t   */
/* at word t * bs of the row (bs >= L). Integers of a conversion are    */
/* placed by a view: integer k at word at (v, k) of the row.            */

struct view
{
  size_t seg;     /* integers per segment */
  size_t sstride; /* words between segments */
};

/* Contiguous: integer k at word k. */
static const struct view CONTIG = {(size_t) -1, 0};

__device__ __forceinline__ size_t
at (const struct view v, size_t k)
{
  return (k / v.seg) * v.sstride + k % v.seg;
}

/* For every transform, its L words from its len words: the words beyond L
   folded in (added; only with nt = 1), zeros beyond len, 1 added at
   len % L if monic (mpzspv_mul_ntt's step before a forward transform). */
__global__ void
k_prep (uint64_t *base, size_t stride, unsigned sp_num, size_t nt, size_t bs,
        size_t len, size_t L, int monic, const uint64_t *P)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  size_t per = nt * L;
  if (t >= (size_t) sp_num * per)
    return;
  unsigned i = t / per;
  size_t r = t % per, tix = r / L, j = r % L;
  uint64_t p = P[i];
  uint64_t *x = base + i * stride + tix * bs;
  uint64_t v = j < len ? x[j] : 0;
  for (size_t b = j + L; b < len; b += L)
    v = d_add (v, x[b], p);
  if (monic && j == len % L)
    v = d_add (v, 1, p);
  x[j] = v;
}

/* One DIF stage, half-length h, over every prime and transform. */
__global__ void
k_dif_stage (uint64_t *base, size_t stride, unsigned sp_num, size_t nt,
             size_t bs, size_t L, size_t h, const struct gpu_ctx c,
             const uint64_t *W)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  size_t per = nt * (L / 2);
  if (t >= (size_t) sp_num * per)
    return;
  unsigned i = t / per;
  size_t r = t % per, tix = r / (L / 2), j = r % (L / 2);
  size_t k = j % h, pos = tix * bs + (j / h) * 2 * h + k;
  uint64_t p = c.P[i], d = c.D[i];
  uint64_t *x = base + i * stride;
  uint64_t a = x[pos], b = x[pos + h];
  uint64_t w = W[i * c.half + k * (((size_t) 1 << c.logK) / (2 * h))];
  x[pos] = d_add (a, b, p);
  x[pos + h] = d_mul (d_sub (a, b, p), w, p, d);
}

/* One DIT stage, half-length h. */
__global__ void
k_dit_stage (uint64_t *base, size_t stride, unsigned sp_num, size_t nt,
             size_t bs, size_t L, size_t h, const struct gpu_ctx c,
             const uint64_t *W)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  size_t per = nt * (L / 2);
  if (t >= (size_t) sp_num * per)
    return;
  unsigned i = t / per;
  size_t r = t % per, tix = r / (L / 2), j = r % (L / 2);
  size_t k = j % h, pos = tix * bs + (j / h) * 2 * h + k;
  uint64_t p = c.P[i], d = c.D[i];
  uint64_t *x = base + i * stride;
  uint64_t w = W[i * c.half + k * (((size_t) 1 << c.logK) / (2 * h))];
  uint64_t a = x[pos], b = d_mul (x[pos + h], w, p, d);
  x[pos] = d_add (a, b, p);
  x[pos + h] = d_sub (a, b, p);
}

#define SEG 2048 /* words per block in shared memory (16 KB) */

/* The DIF stages h = S/2 down to 1, S = min(L, SEG), on segments of S
   words in shared memory: one block per segment, transform and prime. */
__global__ void
k_dif_tail (uint64_t *base, size_t stride, size_t nt, size_t bs, size_t L,
            const struct gpu_ctx c, const uint64_t *W)
{
  __shared__ uint64_t s[SEG];
  size_t S = L < SEG ? L : SEG;
  size_t nseg = L / S, per = nt * nseg;
  unsigned i = blockIdx.x / per;
  size_t r = blockIdx.x % per, tix = r / nseg, seg = r % nseg;
  uint64_t p = c.P[i], d = c.D[i];
  uint64_t *x = base + i * stride + tix * bs + seg * S;
  const uint64_t *w = W + i * c.half;
  for (size_t j = threadIdx.x; j < S; j += blockDim.x)
    s[j] = x[j];
  __syncthreads ();
  for (size_t h = S / 2; h >= 1; h /= 2)
    {
      size_t sh = ((size_t) 1 << c.logK) / (2 * h);
      for (size_t j = threadIdx.x; j < S / 2; j += blockDim.x)
        {
          size_t k = j % h, pos = (j / h) * 2 * h + k;
          uint64_t a = s[pos], b = s[pos + h];
          s[pos] = d_add (a, b, p);
          s[pos + h] = d_mul (d_sub (a, b, p), w[k * sh], p, d);
        }
      __syncthreads ();
    }
  for (size_t j = threadIdx.x; j < S; j += blockDim.x)
    x[j] = s[j];
}

/* The DIT stages h = 1 up to S/2, on segments in shared memory. */
__global__ void
k_dit_head (uint64_t *base, size_t stride, size_t nt, size_t bs, size_t L,
            const struct gpu_ctx c, const uint64_t *W)
{
  __shared__ uint64_t s[SEG];
  size_t S = L < SEG ? L : SEG;
  size_t nseg = L / S, per = nt * nseg;
  unsigned i = blockIdx.x / per;
  size_t r = blockIdx.x % per, tix = r / nseg, seg = r % nseg;
  uint64_t p = c.P[i], d = c.D[i];
  uint64_t *x = base + i * stride + tix * bs + seg * S;
  const uint64_t *w = W + i * c.half;
  for (size_t j = threadIdx.x; j < S; j += blockDim.x)
    s[j] = x[j];
  __syncthreads ();
  for (size_t h = 1; h < S; h *= 2)
    {
      size_t sh = ((size_t) 1 << c.logK) / (2 * h);
      for (size_t j = threadIdx.x; j < S / 2; j += blockDim.x)
        {
          size_t k = j % h, pos = (j / h) * 2 * h + k;
          uint64_t a = s[pos], b = d_mul (s[pos + h], w[k * sh], p, d);
          s[pos] = d_add (a, b, p);
          s[pos + h] = d_sub (a, b, p);
        }
      __syncthreads ();
    }
  for (size_t j = threadIdx.x; j < S; j += blockDim.x)
    x[j] = s[j];
}

/* r = x * y pointwise: nt pairs of L words, pair t at t * rbs, t * xbs,
   t * ybs. */
__global__ void
k_pwmul (uint64_t *r, size_t sr, size_t rbs, const uint64_t *x, size_t sx,
         size_t xbs, const uint64_t *y, size_t sy, size_t ybs,
         unsigned sp_num, size_t nt, size_t L, const uint64_t *P,
         const uint64_t *D)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  size_t per = nt * L;
  if (t >= (size_t) sp_num * per)
    return;
  unsigned i = t / per;
  size_t q = t % per, tix = q / L, j = q % L;
  r[i * sr + tix * rbs + j] = d_mul (x[i * sx + tix * xbs + j],
                                     y[i * sy + tix * ybs + j], P[i], D[i]);
}

/* After the inverse transforms: times 1/L, and 1 subtracted at
   monic_pos % L if monic_pos != 0. */
__global__ void
k_post (uint64_t *base, size_t stride, unsigned sp_num, size_t nt, size_t bs,
        size_t L, size_t monic_pos, const uint64_t *P, const uint64_t *D)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  size_t per = nt * L;
  if (t >= (size_t) sp_num * per)
    return;
  unsigned i = t / per;
  size_t q = t % per, tix = q / L, j = q % L;
  uint64_t p = P[i], d = D[i];
  uint64_t invn = p - (p - 1) / L;
  uint64_t *x = base + i * stride + tix * bs;
  uint64_t v = d_mul (x[j], invn, p, d);
  if (monic_pos && j == monic_pos % L)
    v = d_sub (v, 1, p);
  x[j] = v;
}

/* Residues of len integers: limbs of integer k at L + k * ls, n[k] of
   them (0 for 0), least significant first. Row i, word at (v, k) = that
   mod p_i (mpzspv_from_mpzv). One thread per integer and prime. */
__global__ void
k_from (uint64_t *base, size_t stride, unsigned sp_num, size_t len,
        const struct view v, const uint64_t *L, size_t ls, const uint32_t *n,
        const uint64_t *P, const uint64_t *D, const uint64_t *Bm)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (t >= (size_t) sp_num * len)
    return;
  unsigned i = t % sp_num;
  size_t k = t / sp_num;
  uint64_t p = P[i], d = D[i], b = Bm[i];
  const uint64_t *l = L + k * ls;
  uint64_t r = 0;
  for (int j = (int) n[k] - 1; j >= 0; j--)
    r = d_add (d_mul (r, b, p, d), l[j] % p, p);
  base[i * stride + at (v, k)] = r;
}

/* Explicit CRT (mpzspv_to_mpzv): integer k = sum_i t_i crt1[i] +
   crt2[floor(f)], t_i = x[i][at (v, k)] crt3[i] mod p_i, f = 0.5 + sum_i
   t_i / p_i in single precision, summed in the order of the CPU code with
   IEEE rounding and no fused multiply-add, so that floor(f) is the same.
   out: nl + 2 limbs per integer. */
__global__ void
k_to (const uint64_t *base, size_t stride, unsigned sp_num, size_t len,
      const struct view v, const struct gpu_ctx c, uint64_t *out)
{
  size_t k = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (k >= len)
    return;
  const unsigned nl = c.nl;
  const size_t w = at (v, k);
  uint64_t *acc = out + k * (nl + 2);
  for (unsigned j = 0; j < nl + 2; j++)
    acc[j] = 0;
  float f = 0.5f;
  for (unsigned i = 0; i < sp_num; i++)
    {
      uint64_t p = c.P[i], d = c.D[i];
      uint64_t t = d_mul (base[i * stride + w], c.crt3[i], p, d);
      float recip = __fdiv_rn (1.0f, __ull2float_rn (p));
      f = __fadd_rn (f, __fmul_rn (__ull2float_rn (t), recip));
      const uint64_t *m = c.crt1 + (size_t) i * nl;
      uint64_t cy = 0;
      for (unsigned j = 0; j < nl; j++)
        {
          uint64_t lo = t * m[j], hi = __umul64hi (t, m[j]);
          lo += cy;
          hi += lo < cy;
          uint64_t s = acc[j] + lo;
          hi += s < lo;
          acc[j] = s;
          cy = hi;
        }
      for (unsigned j = nl; cy && j < nl + 2; j++)
        {
          uint64_t s = acc[j] + cy;
          cy = s < cy;
          acc[j] = s;
        }
    }
  const uint64_t *m = c.crt2 + (size_t) ((unsigned) f) * nl;
  uint64_t cy = 0;
  for (unsigned j = 0; j < nl + 2; j++)
    {
      uint64_t a = j < nl ? m[j] : 0;
      uint64_t s = acc[j] + a;
      uint64_t c1 = s < a;
      s += cy;
      c1 += s < cy;
      acc[j] = s;
      cy = c1;
    }
}

/* mpzspv_normalise: the word at (v, k) of every row replaced by the
   residues of an integer congruent mod N, by the explicit CRT mod N mod
   p_i (Bernstein and Sorenson, Theorem 4.1), as the CPU code: t_j = x[j]
   crt3[j], f as in k_to, then x[i] = (crt5[i] floor(f) + sum_j t_j
   crt4[i][j]) mod p_i. tt: sp_num words of scratch per integer. */
__global__ void
k_norm (uint64_t *base, size_t stride, unsigned sp_num, size_t len,
        const struct view v, const struct gpu_ctx c, uint64_t *tt)
{
  size_t k = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (k >= len)
    return;
  const size_t w0 = at (v, k);
  uint64_t *t = tt + k * sp_num;
  float f = 0.5f;
  for (unsigned i = 0; i < sp_num; i++)
    {
      uint64_t p = c.P[i];
      t[i] = d_mul (base[i * stride + w0], c.crt3[i], p, c.D[i]);
      float recip = __fdiv_rn (1.0f, __ull2float_rn (p));
      f = __fadd_rn (f, __fmul_rn (__ull2float_rn (t[i]), recip));
    }
  uint64_t fi = __float2ull_rz (f);
  for (unsigned i = 0; i < sp_num; i++)
    {
      uint64_t p = c.P[i], d = c.D[i];
      /* d2:d1:d0 = crt5[i] fi + sum_j t_j crt4[i][j] (fits in 3 words) */
      uint64_t d0 = c.crt5[i] * fi, d1 = __umul64hi (c.crt5[i], fi), d2 = 0;
      const uint64_t *w = c.crt4 + (size_t) i * sp_num;
      for (unsigned j = 0; j < sp_num; j++)
        {
          uint64_t lo = t[j] * w[j], hi = __umul64hi (t[j], w[j]);
          d0 += lo;
          hi += d0 < lo;
          d1 += hi;
          d2 += d1 < hi;
        }
      /* (d2 B^2 + d1 B + d0) mod p, B = 2^64 */
      uint64_t b = c.Bm[i];
      uint64_t r = d2 % p;
      r = d_add (d_mul (r, b, p, d), d1 % p, p);
      r = d_add (d_mul (r, b, p, d), d0 % p, p);
      base[i * stride + w0] = r;
    }
}


/* For the transposed product tree (ntt_polyevalT): node q of a level with
   children of degree m, children A (z[2mq ..]) and B (z[2mq + m ..]):
   x[4mq + j] = rev-monic(A)[j], x[4mq + 2m + j] = rev-monic(B)[j], j < 2m,
   where rev-monic(A) = 1, A[m-1], ..., A[0], then zeros. */
__global__ void
k_revmonic (uint64_t *x, size_t sx, const uint64_t *z, size_t sz,
            unsigned sp_num, size_t len, size_t m)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  size_t per = 2 * len; /* words of x per prime */
  if (t >= (size_t) sp_num * per)
    return;
  unsigned i = t / per;
  size_t w = t % per;
  size_t q = w / (4 * m), r = w % (4 * m), ch = r / (2 * m), j = r % (2 * m);
  uint64_t v;
  if (j == 0)
    v = 1;
  else if (j <= m)
    v = z[i * sz + 2 * m * q + ch * m + (m - j)];
  else
    v = 0;
  x[i * sx + w] = v;
}

/* The middle halves of the two products of node q back into y:
   y[2mq + m + j] = x[4mq + m + j], y[2mq + j] = x[4mq + 3m + j]. */
__global__ void
k_mid (uint64_t *y, size_t sy, const uint64_t *x, size_t sx, unsigned sp_num,
       size_t len, size_t m)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (t >= (size_t) sp_num * len)
    return;
  unsigned i = t / len;
  size_t w = t % len;
  size_t q = w / (2 * m), r = w % (2 * m);
  const uint64_t *xs = x + i * sx + 4 * m * q;
  y[i * sy + w] = r < m ? xs[3 * m + r] : xs[r];
}

/* Residues of total integers in device memory, W words each (least
   significant first, non-negative), integer k to word at (v, k) of the
   rows. */
__global__ void
k_from_dev (uint64_t *base, size_t stride, unsigned sp_num, size_t total,
            const struct view v, const uint64_t *src, unsigned W,
            const uint64_t *P, const uint64_t *D, const uint64_t *Bm)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (t >= (size_t) sp_num * total)
    return;
  unsigned i = t % sp_num;
  size_t k = t / sp_num;
  uint64_t p = P[i], d = D[i], b = Bm[i];
  const uint64_t *l = src + k * W;
  uint64_t r = 0;
  for (int j = (int) W - 1; j >= 0; j--)
    r = d_add (d_mul (r, b, p, d), l[j] % p, p);
  base[i * stride + at (v, k)] = r;
}

/* For nt blocks: dst[t dbs + j], j < L, = 1 if one0 and j == 0; else
   (-)src[t sbs + a + dir j] for j0 <= j < j1; else 0. */
__global__ void
k_move (uint64_t *dst, size_t sd, size_t dbs, const uint64_t *src, size_t ss,
        size_t sbs, unsigned sp_num, size_t nt, size_t L, size_t j0,
        size_t j1, long a, int dir, int neg, int one0, const uint64_t *P)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  size_t per = nt * L;
  if (t >= (size_t) sp_num * per)
    return;
  unsigned i = t / per;
  size_t q = t % per, tix = q / L, j = q % L;
  uint64_t v = 0;
  if (one0 && j == 0)
    v = 1;
  else if (j >= j0 && j < j1)
    {
      v = src[i * ss + tix * sbs + (size_t) (a + (long) dir * (long) j)];
      if (neg && v)
        v = P[i] - v;
    }
  dst[i * sd + tix * dbs + j] = v;
}

/* dst[t dbs + j] = x[t xbs + j] + sign y[t ybs + j] (y only for j < cy),
   j < L. */
__global__ void
k_addsub (uint64_t *dst, size_t sd, size_t dbs, const uint64_t *x, size_t sx,
          size_t xbs, const uint64_t *y, size_t sy, size_t ybs,
          unsigned sp_num, size_t nt, size_t L, size_t cy, int sign,
          const uint64_t *P)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  size_t per = nt * L;
  if (t >= (size_t) sp_num * per)
    return;
  unsigned i = t / per;
  size_t q = t % per, tix = q / L, j = q % L;
  uint64_t p = P[i], a = x[i * sx + tix * xbs + j];
  if (j < cy)
    {
      uint64_t b = y[i * sy + tix * ybs + j];
      a = sign > 0 ? d_add (a, b, p) : d_sub (a, b, p);
    }
  dst[i * sd + tix * dbs + j] = a;
}

/* W[i*half + j] = root_i^j. */
__global__ void
k_twiddles (uint64_t *W, size_t half, unsigned sp_num, const uint64_t *R,
            const uint64_t *P, const uint64_t *D)
{
  size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (t >= (size_t) sp_num * half)
    return;
  unsigned i = t / half;
  size_t j = t % half;
  W[t] = d_pow (R[i], j, P[i], D[i]);
}

/* ------------------------------------------------------------------ */
/* Host side                                                            */

namespace {

int g_active = -1; /* -1 not decided yet */
int g_check = 0;
int g_stats = 0;
/* ECM_GPU_NTT_STATS=1: calls done on the GPU, declined (CPU), seconds. */
unsigned long g_n_gpu = 0, g_n_cpu = 0;
double g_t_gpu = 0.0;

double
now (void)
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

void
print_stats (void)
{
  fprintf (stderr, "GPU NTT: %lu operations on the GPU (%.3f s), %lu on the "
           "CPU\n", g_n_gpu, g_t_gpu, g_n_cpu);
}
std::mutex g_mutex;
std::set<uint64_t *> g_blocks; /* bases from gpu_mpzspv_init */

const unsigned TPB = 256;

inline unsigned
nblocks (size_t n)
{
  return (unsigned) ((n + TPB - 1) / TPB);
}

void
fail (const char *what, cudaError_t e)
{
  fprintf (stderr, "GPU NTT: %s: %s\n", what, cudaGetErrorString (e));
  abort ();
}

inline void
check_launch (const char *what)
{
  cudaError_t e = cudaGetLastError ();
  if (e != cudaSuccess)
    fail (what, e);
}

bool
owned (const uint64_t *b)
{
  std::lock_guard<std::mutex> lk (g_mutex);
  return g_blocks.count ((uint64_t *) b) != 0;
}

/* The row stride of a vector from gpu_mpzspv_init, or 0 if x is not one. */
size_t
stride_of (mpzspv_t x, unsigned sp_num)
{
  if (!owned (x[0]))
    return 0;
  if (sp_num < 2)
    return 1; /* any: one row */
  return (size_t) (x[1] - x[0]);
}

/* nt forward transforms of length L, transform t at base + t * bs. */
void
dif (const gpu_ctx *c, uint64_t *base, size_t stride, size_t nt, size_t bs,
     size_t L, cudaStream_t s)
{
  unsigned sp = c->sp_num;
  for (size_t h = L / 2; h >= SEG; h /= 2)
    {
      k_dif_stage<<<nblocks (sp * nt * (L / 2)), TPB, 0, s>>> (base, stride,
          sp, nt, bs, L, h, *c, c->Wf);
      check_launch ("dif stage");
    }
  size_t S = L < SEG ? L : SEG;
  if (S >= 2)
    {
      k_dif_tail<<<(unsigned) (sp * nt * (L / S)), S < 1024 ? S / 2 : 512, 0,
                   s>>> (base, stride, nt, bs, L, *c, c->Wf);
      check_launch ("dif tail");
    }
}

/* nt inverse transforms (without the 1/L). */
void
dit (const gpu_ctx *c, uint64_t *base, size_t stride, size_t nt, size_t bs,
     size_t L, cudaStream_t s)
{
  unsigned sp = c->sp_num;
  size_t S = L < SEG ? L : SEG;
  if (S >= 2)
    {
      k_dit_head<<<(unsigned) (sp * nt * (L / S)), S < 1024 ? S / 2 : 512, 0,
                   s>>> (base, stride, nt, bs, L, *c, c->Wi);
      check_launch ("dit head");
    }
  for (size_t h = SEG; h < L; h *= 2)
    {
      k_dit_stage<<<nblocks (sp * nt * (L / 2)), TPB, 0, s>>> (base, stride,
          sp, nt, bs, L, h, *c, c->Wi);
      check_launch ("dit stage");
    }
}

void
sync (cudaStream_t s)
{
  cudaError_t e = cudaStreamSynchronize (s);
  if (e != cudaSuccess)
    fail ("synchronize", e);
}

int
declined (void)
{
  if (g_stats)
    {
      std::lock_guard<std::mutex> lk (g_mutex);
      g_n_cpu++;
    }
  return 0;
}

void
counted (double t0)
{
  if (g_stats)
    {
      double t = now () - t0;
      std::lock_guard<std::mutex> lk (g_mutex);
      g_n_gpu++;
      g_t_gpu += t;
    }
}

/* Per-thread scratch: pinned host memory and device memory, grown as
   needed. */
struct scratch
{
  void *h = NULL, *d = NULL;
  size_t hs = 0, ds = 0;
  void *host (size_t n)
  {
    if (n > hs)
      {
        if (h) cudaFreeHost (h);
        hs = n + n / 4;
        cudaError_t e = cudaMallocHost (&h, hs);
        if (e != cudaSuccess)
          fail ("pinned memory", e);
      }
    return h;
  }
  void *dev (size_t n)
  {
    if (n > ds)
      {
        if (d) cudaFree (d);
        ds = n + n / 4;
        cudaError_t e = cudaMalloc (&d, ds);
        if (e != cudaSuccess)
          fail ("device memory", e);
      }
    return d;
  }
};
thread_local scratch g_s1, g_s2, g_s3;

/* ---- check mode: the CPU code on copies ---- */

struct copy_set
{
  unsigned sp_num;
  std::set<uint64_t *> bases;
  struct ent { uint64_t *orig; size_t words; uint64_t *copy; };
  ent e[3];
  int n = 0;

  /* A copy of the block of x (sp_num rows of stride words). */
  void add (mpzspv_t x, size_t stride)
  {
    for (int k = 0; k < n; k++)
      if (e[k].orig == x[0])
        return;
    size_t words = stride * sp_num;
    uint64_t *c = (uint64_t *) malloc (words * sizeof (uint64_t));
    if (!c)
      abort ();
    memcpy (c, x[0], words * sizeof (uint64_t));
    e[n++] = {x[0], words, c};
  }
  /* The rows of the copy of x, as an mpzspv_t (malloc'ed). */
  mpzspv_t rows (mpzspv_t x)
  {
    for (int k = 0; k < n; k++)
      if (e[k].orig == x[0])
        {
          mpzspv_t r = (mpzspv_t) malloc (sp_num * sizeof (spv_t));
          for (unsigned i = 0; i < sp_num; i++)
            r[i] = e[k].copy + (x[i] - x[0]);
          return r;
        }
    abort ();
  }
  void compare (const char *what)
  {
    for (int k = 0; k < n; k++)
      for (size_t j = 0; j < e[k].words; j++)
        if (e[k].orig[j] != e[k].copy[j])
          {
            fprintf (stderr, "GPU NTT check: %s differs from the CPU at "
                     "word %zu of %zu: %llu != %llu\n", what, j, e[k].words,
                     (unsigned long long) e[k].orig[j],
                     (unsigned long long) e[k].copy[j]);
            abort ();
          }
  }
  ~copy_set ()
  {
    for (int k = 0; k < n; k++)
      free (e[k].copy);
  }
};

} /* namespace */

extern "C" {

int
gpu_ntt_active (void)
{
  if (g_active < 0)
    {
      std::lock_guard<std::mutex> lk (g_mutex);
      if (g_active < 0)
        {
          const char *e = getenv ("ECM_GPU_NTT");
          int n = 0;
          if (e && strcmp (e, "0") == 0)
            g_active = 0;
          else
            g_active = cudaGetDeviceCount (&n) == cudaSuccess && n > 0;
          e = getenv ("ECM_GPU_NTT_CHECK");
          g_check = e && strcmp (e, "1") == 0;
          e = getenv ("ECM_GPU_NTT_STATS");
          g_stats = e && strcmp (e, "1") == 0;
          if (g_stats)
            atexit (print_stats);
        }
    }
  return g_active;
}

mpzspv_t
gpu_mpzspv_init (spv_size_t len, mpzspm_t mpzspm)
{
  if (!gpu_ntt_active () || mpzspm->gpu == NULL || len == 0)
    return NULL;
  unsigned sp_num = mpzspm->sp_num;
  uint64_t *b = NULL;
  if (cudaMallocManaged (&b, (size_t) sp_num * len * sizeof (uint64_t))
      != cudaSuccess)
    return NULL;
  mpzspv_t x = (mpzspv_t) malloc (sp_num * sizeof (spv_t));
  if (x == NULL)
    {
      cudaFree (b);
      return NULL;
    }
  for (unsigned i = 0; i < sp_num; i++)
    x[i] = b + (size_t) i * len;
  std::lock_guard<std::mutex> lk (g_mutex);
  g_blocks.insert (b);
  return x;
}

int
gpu_mpzspv_clear (mpzspv_t x, mpzspm_t mpzspm)
{
  (void) mpzspm;
  {
    std::lock_guard<std::mutex> lk (g_mutex);
    if (!g_blocks.erase (x[0]))
      return 0;
  }
  cudaFree (x[0]);
  free (x);
  return 1;
}

void
gpu_mpzspm_init (mpzspm_t mpzspm)
{
  mpzspm->gpu = NULL;
  if (!gpu_ntt_active ())
    return;
  unsigned sp_num = mpzspm->sp_num;
  size_t K = mpzspm->max_ntt_size;
  unsigned logK = 0;
  while (((size_t) 1 << logK) < K)
    logK++;
  if (((size_t) 1 << logK) != K || K < 2)
    return;
  gpu_ctx *c = (gpu_ctx *) malloc (sizeof (gpu_ctx));
  if (!c)
    return;
  c->sp_num = sp_num;
  c->logK = logK;
  c->half = K / 2;
  uint64_t *hP = (uint64_t *) malloc (sp_num * sizeof (uint64_t));
  uint64_t *hD = (uint64_t *) malloc (sp_num * sizeof (uint64_t));
  uint64_t *hRf = (uint64_t *) malloc (sp_num * sizeof (uint64_t));
  uint64_t *hRi = (uint64_t *) malloc (sp_num * sizeof (uint64_t));
  for (unsigned i = 0; i < sp_num; i++)
    {
      spm_t spm = mpzspm->spm[i];
      hP[i] = spm->sp;
      hD[i] = spm->mul_c;
      hRf[i] = spm->nttdata->ntt_roots[logK];
      hRi[i] = spm->inttdata->ntt_roots[logK];
    }
  uint64_t *dRf, *dRi;
  cudaError_t e = cudaSuccess;
  size_t sb = sp_num * sizeof (uint64_t);
  if ((e = cudaMalloc (&c->P, sb)) != cudaSuccess ||
      (e = cudaMalloc (&c->D, sb)) != cudaSuccess ||
      (e = cudaMalloc (&dRf, sb)) != cudaSuccess ||
      (e = cudaMalloc (&dRi, sb)) != cudaSuccess ||
      (e = cudaMalloc (&c->Wf, sb * c->half)) != cudaSuccess ||
      (e = cudaMalloc (&c->Wi, sb * c->half)) != cudaSuccess)
    fail ("allocating the NTT tables", e);
  cudaMemcpy (c->P, hP, sb, cudaMemcpyHostToDevice);
  cudaMemcpy (c->D, hD, sb, cudaMemcpyHostToDevice);
  cudaMemcpy (dRf, hRf, sb, cudaMemcpyHostToDevice);
  cudaMemcpy (dRi, hRi, sb, cudaMemcpyHostToDevice);
  size_t n = (size_t) sp_num * c->half;
  k_twiddles<<<nblocks (n), TPB>>> (c->Wf, c->half, sp_num, dRf, c->P, c->D);
  k_twiddles<<<nblocks (n), TPB>>> (c->Wi, c->half, sp_num, dRi, c->P, c->D);
  check_launch ("twiddles");
  if ((e = cudaDeviceSynchronize ()) != cudaSuccess)
    fail ("twiddles", e);
  cudaFree (dRf);
  cudaFree (dRi);

  /* CRT tables */
  unsigned nl = (unsigned) mpz_size (mpzspm->modulus);
  c->nl = nl;
  uint64_t *h1 = (uint64_t *) calloc ((size_t) sp_num * nl, 8);
  uint64_t *h2 = (uint64_t *) calloc ((size_t) (sp_num + 2) * nl, 8);
  uint64_t *h3 = (uint64_t *) malloc (sb), *h5 = (uint64_t *) malloc (sb);
  uint64_t *h4 = (uint64_t *) malloc ((size_t) sp_num * sb);
  uint64_t *hB = (uint64_t *) malloc (sb);
  for (unsigned i = 0; i < sp_num; i++)
    {
      mpz_export (h1 + (size_t) i * nl, NULL, -1, 8, 0, 0, mpzspm->crt1[i]);
      h3[i] = mpzspm->crt3[i];
      h5[i] = mpzspm->crt5[i];
      for (unsigned j = 0; j < sp_num; j++)
        h4[(size_t) i * sp_num + j] = mpzspm->crt4[i][j];
      /* 2^64 mod p = (2^64 - p) mod p */
      hB[i] = (0 - hP[i]) % hP[i];
    }
  for (unsigned i = 0; i < sp_num + 2; i++)
    mpz_export (h2 + (size_t) i * nl, NULL, -1, 8, 0, 0, mpzspm->crt2[i]);
  if ((e = cudaMalloc (&c->crt1, (size_t) sp_num * nl * 8)) != cudaSuccess ||
      (e = cudaMalloc (&c->crt2, (size_t) (sp_num + 2) * nl * 8))
      != cudaSuccess ||
      (e = cudaMalloc (&c->crt3, sb)) != cudaSuccess ||
      (e = cudaMalloc (&c->crt5, sb)) != cudaSuccess ||
      (e = cudaMalloc (&c->crt4, (size_t) sp_num * sb)) != cudaSuccess ||
      (e = cudaMalloc (&c->Bm, sb)) != cudaSuccess)
    fail ("allocating the CRT tables", e);
  cudaMemcpy (c->crt1, h1, (size_t) sp_num * nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy (c->crt2, h2, (size_t) (sp_num + 2) * nl * 8,
              cudaMemcpyHostToDevice);
  cudaMemcpy (c->crt3, h3, sb, cudaMemcpyHostToDevice);
  cudaMemcpy (c->crt5, h5, sb, cudaMemcpyHostToDevice);
  cudaMemcpy (c->crt4, h4, (size_t) sp_num * sb, cudaMemcpyHostToDevice);
  cudaMemcpy (c->Bm, hB, sb, cudaMemcpyHostToDevice);
  free (h1); free (h2); free (h3); free (h4); free (h5); free (hB);
  free (hP);
  free (hD);
  free (hRf);
  free (hRi);
  mpzspm->gpu = c;
}

void
gpu_mpzspm_clear (mpzspm_t mpzspm)
{
  gpu_ctx *c = (gpu_ctx *) mpzspm->gpu;
  if (!c)
    return;
  cudaFree (c->P);
  cudaFree (c->D);
  cudaFree (c->Wf);
  cudaFree (c->Wi);
  cudaFree (c->crt1);
  cudaFree (c->crt2);
  cudaFree (c->crt3);
  cudaFree (c->crt4);
  cudaFree (c->crt5);
  cudaFree (c->Bm);
  free (c);
  mpzspm->gpu = NULL;
}

int
gpu_mpzspv_mul_ntt (mpzspv_t r, spv_size_t offsetr,
    mpzspv_t x, spv_size_t offsetx, spv_size_t lenx,
    mpzspv_t y, spv_size_t offsety, spv_size_t leny,
    spv_size_t ntt_size, int monic, spv_size_t monic_pos,
    mpzspm_t mpzspm, int steps)
{
  const gpu_ctx *c = (const gpu_ctx *) mpzspm->gpu;
  if (!c || ntt_size < 2 || ntt_size > ((size_t) 1 << c->logK))
    return declined ();
  unsigned sp = c->sp_num;
  size_t sr = 0, sx = 0, sy = 0;
  if ((steps & (NTT_MUL_STEP_FFT1 | NTT_MUL_STEP_MUL)) &&
      !(sx = stride_of (x, sp)))
    return declined ();
  if ((steps & (NTT_MUL_STEP_FFT2 | NTT_MUL_STEP_MUL)) &&
      !(sy = stride_of (y, sp)))
    return declined ();
  if ((steps & (NTT_MUL_STEP_MUL | NTT_MUL_STEP_IFFT)) &&
      !(sr = stride_of (r, sp)))
    return declined ();
  double t0 = g_stats ? now () : 0.0;

  copy_set cs;
  cs.sp_num = sp;
  if (g_check)
    {
      if (sx) cs.add (x, sx);
      if (sy) cs.add (y, sy);
      if (sr) cs.add (r, sr);
    }

  cudaStream_t s = cudaStreamPerThread;
  size_t n = (size_t) sp * ntt_size;
  if (steps & NTT_MUL_STEP_FFT1)
    {
      k_prep<<<nblocks (n), TPB, 0, s>>> (x[0] + offsetx, sx, sp, 1,
          ntt_size, lenx, ntt_size, monic, c->P);
      check_launch ("prep");
      dif (c, x[0] + offsetx, sx, 1, ntt_size, ntt_size, s);
    }
  if (steps & NTT_MUL_STEP_FFT2)
    {
      k_prep<<<nblocks (n), TPB, 0, s>>> (y[0] + offsety, sy, sp, 1,
          ntt_size, leny, ntt_size, monic, c->P);
      check_launch ("prep");
      dif (c, y[0] + offsety, sy, 1, ntt_size, ntt_size, s);
    }
  if (steps & NTT_MUL_STEP_MUL)
    {
      k_pwmul<<<nblocks (n), TPB, 0, s>>> (r[0] + offsetr, sr, 0,
          x[0] + offsetx, sx, 0, y[0] + offsety, sy, 0, sp, 1, ntt_size,
          c->P, c->D);
      check_launch ("pwmul");
    }
  if (steps & NTT_MUL_STEP_IFFT)
    {
      dit (c, r[0] + offsetr, sr, 1, ntt_size, ntt_size, s);
      k_post<<<nblocks (n), TPB, 0, s>>> (r[0] + offsetr, sr, sp, 1, ntt_size,
                                          ntt_size, monic_pos, c->P, c->D);
      check_launch ("post");
    }
  sync (s);
  counted (t0);

  if (g_check)
    {
      /* The CPU code on the copies, then every word compared. */
      mpzspv_t rr = sr ? cs.rows (r) : r, xx = sx ? cs.rows (x) : x,
               yy = sy ? cs.rows (y) : y;
      mpzspv_mul_ntt_cpu (rr, offsetr, xx, offsetx, lenx, yy, offsety, leny,
                          ntt_size, monic, monic_pos, mpzspm, steps);
      cs.compare ("mpzspv_mul_ntt");
      if (sr) free (rr);
      if (sx) free (xx);
      if (sy) free (yy);
    }
  return 1;
}

int
gpu_mpzspv_to_ntt (mpzspv_t x, spv_size_t offset, spv_size_t len,
    spv_size_t ntt_size, int monic, mpzspm_t mpzspm)
{
  const gpu_ctx *c = (const gpu_ctx *) mpzspm->gpu;
  if (!c || ntt_size < 2 || ntt_size > ((size_t) 1 << c->logK))
    return declined ();
  unsigned sp = c->sp_num;
  size_t sx = stride_of (x, sp);
  if (!sx)
    return declined ();
  double t0 = g_stats ? now () : 0.0;
  copy_set cs;
  cs.sp_num = sp;
  if (g_check)
    cs.add (x, sx);
  cudaStream_t s = cudaStreamPerThread;
  size_t n = (size_t) sp * ntt_size;
  k_prep<<<nblocks (n), TPB, 0, s>>> (x[0] + offset, sx, sp, 1, ntt_size,
                                      len, ntt_size, monic, c->P);
  check_launch ("prep");
  dif (c, x[0] + offset, sx, 1, ntt_size, ntt_size, s);
  sync (s);
  counted (t0);
  if (g_check)
    {
      mpzspv_t xx = cs.rows (x);
      mpzspv_to_ntt_cpu (xx, offset, len, ntt_size, monic, mpzspm);
      cs.compare ("mpzspv_to_ntt");
      free (xx);
    }
  return 1;
}

int
gpu_mpzspv_from_mpzv (mpzspv_t x, spv_size_t offset, const mpzv_t mpzv,
    spv_size_t len, mpzspm_t mpzspm)
{
  const gpu_ctx *c = (const gpu_ctx *) mpzspm->gpu;
  if (!c || len == 0)
    return declined ();
  unsigned sp = c->sp_num;
  size_t sx = stride_of (x, sp);
  if (!sx)
    return declined ();
  double t0 = g_stats ? now () : 0.0;
  size_t ls = 1;
  for (size_t k = 0; k < len; k++)
    {
      if (mpz_sgn (mpzv[k]) < 0)
        return declined (); /* the CPU code asserts against this */
      if (mpz_size (mpzv[k]) > ls)
        ls = mpz_size (mpzv[k]);
    }
  size_t lb = len * ls * 8, nb = len * 4;
  char *h = (char *) g_s1.host (lb + nb);
  char *d = (char *) g_s1.dev (lb + nb);
  uint64_t *hl = (uint64_t *) h;
  uint32_t *hn = (uint32_t *) (h + lb);
  for (size_t k = 0; k < len; k++)
    {
      size_t n = mpz_size (mpzv[k]);
      hn[k] = (uint32_t) n;
      if (n)
        memcpy (hl + k * ls, mpz_limbs_read (mpzv[k]), n * 8);
    }
  cudaStream_t s = cudaStreamPerThread;
  cudaMemcpyAsync (d, h, lb + nb, cudaMemcpyHostToDevice, s);
  k_from<<<nblocks ((size_t) sp * len), TPB, 0, s>>> (x[0] + offset, sx, sp,
      len, CONTIG, (const uint64_t *) d, ls, (const uint32_t *) (d + lb), c->P, c->D,
      c->Bm);
  check_launch ("from_mpzv");
  sync (s);
  counted (t0);
  if (g_check)
    {
      mpzspv_t y = mpzspv_init_cpu (len, mpzspm);
      mpzspv_from_mpzv_cpu (y, 0, mpzv, len, mpzspm);
      for (unsigned i = 0; i < sp; i++)
        for (size_t k = 0; k < len; k++)
          if (y[i][k] != x[i][offset + k])
            {
              fprintf (stderr, "GPU NTT check: mpzspv_from_mpzv differs "
                       "from the CPU (prime %u, integer %zu)\n", i, k);
              abort ();
            }
      mpzspv_clear (y, mpzspm);
    }
  return 1;
}

int
gpu_mpzspv_to_mpzv (mpzspv_t x, spv_size_t offset, mpzv_t mpzv,
    spv_size_t len, mpzspm_t mpzspm)
{
  const gpu_ctx *c = (const gpu_ctx *) mpzspm->gpu;
  if (!c || len == 0)
    return declined ();
  unsigned sp = c->sp_num;
  size_t sx = stride_of (x, sp);
  if (!sx)
    return declined ();
  double t0 = g_stats ? now () : 0.0;
  const unsigned nl = c->nl;
  size_t ob = len * (nl + 2) * 8;
  uint64_t *d = (uint64_t *) g_s2.dev (ob);
  uint64_t *h = (uint64_t *) g_s2.host (ob);
  cudaStream_t s = cudaStreamPerThread;
  k_to<<<nblocks (len), TPB, 0, s>>> (x[0] + offset, sx, sp, len, CONTIG, *c,
                                     d);
  check_launch ("to_mpzv");
  cudaMemcpyAsync (h, d, ob, cudaMemcpyDeviceToHost, s);
  sync (s);
  for (size_t k = 0; k < len; k++)
    {
      mp_limb_t *l = mpz_limbs_write (mpzv[k], nl + 2);
      memcpy (l, h + k * (nl + 2), (nl + 2) * 8);
      mpz_limbs_finish (mpzv[k], nl + 2);
    }
  counted (t0);
  if (g_check)
    {
      mpz_t *v = (mpz_t *) malloc (len * sizeof (mpz_t));
      for (size_t k = 0; k < len; k++)
        mpz_init (v[k]);
      mpzspv_to_mpzv_cpu (x, offset, v, len, mpzspm);
      for (size_t k = 0; k < len; k++)
        {
          if (mpz_cmp (v[k], mpzv[k]) != 0)
            {
              fprintf (stderr, "GPU NTT check: mpzspv_to_mpzv differs from "
                       "the CPU (integer %zu)\n", k);
              abort ();
            }
          mpz_clear (v[k]);
        }
      free (v);
    }
  return 1;
}

int
gpu_mpzspv_normalise (mpzspv_t x, spv_size_t offset, spv_size_t len,
    mpzspm_t mpzspm)
{
  const gpu_ctx *c = (const gpu_ctx *) mpzspm->gpu;
  if (!c || len == 0)
    return declined ();
  unsigned sp = c->sp_num;
  size_t sx = stride_of (x, sp);
  if (!sx)
    return declined ();
  copy_set cs;
  cs.sp_num = sp;
  if (g_check)
    cs.add (x, sx);
  double t0 = g_stats ? now () : 0.0;
  uint64_t *tt = (uint64_t *) g_s3.dev (len * sp * 8);
  cudaStream_t s = cudaStreamPerThread;
  k_norm<<<nblocks (len), TPB, 0, s>>> (x[0] + offset, sx, sp, len, CONTIG,
                                       *c, tt);
  check_launch ("normalise");
  sync (s);
  counted (t0);
  if (g_check)
    {
      mpzspv_t xx = cs.rows (x);
      mpzspv_normalise_cpu (xx, offset, len, mpzspm);
      cs.compare ("mpzspv_normalise");
      free (xx);
    }
  return 1;
}

int
gpu_ntt_check (void)
{
  return gpu_ntt_active () && g_check;
}

} /* extern "C" */

namespace {

/* len integers (non-negative) to residues at view v of the rows at base. */
bool
upload (const gpu_ctx *c, const mpzv_t a, size_t len, uint64_t *base,
        size_t stride, struct view v, cudaStream_t s)
{
  size_t ls = 1;
  for (size_t k = 0; k < len; k++)
    {
      if (mpz_sgn (a[k]) < 0)
        return false;
      if (mpz_size (a[k]) > ls)
        ls = mpz_size (a[k]);
    }
  size_t lb = len * ls * 8, nb = len * 4;
  char *h = (char *) g_s1.host (lb + nb);
  char *d = (char *) g_s1.dev (lb + nb);
  uint64_t *hl = (uint64_t *) h;
  uint32_t *hn = (uint32_t *) (h + lb);
  for (size_t k = 0; k < len; k++)
    {
      size_t n = mpz_size (a[k]);
      hn[k] = (uint32_t) n;
      if (n)
        memcpy (hl + k * ls, mpz_limbs_read (a[k]), n * 8);
    }
  cudaMemcpyAsync (d, h, lb + nb, cudaMemcpyHostToDevice, s);
  k_from<<<nblocks ((size_t) c->sp_num * len), TPB, 0, s>>> (base, stride,
      c->sp_num, len, v, (const uint64_t *) d, ls, (const uint32_t *) (d + lb),
      c->P, c->D, c->Bm);
  check_launch ("upload");
  /* g_s1's host buffer is reused by the next upload: wait for the copy */
  sync (s);
  return true;
}

/* len integers from the residues at view v (explicit CRT). */
void
download (const gpu_ctx *c, const uint64_t *base, size_t stride,
          struct view v, mpzv_t r, size_t len, cudaStream_t s)
{
  const unsigned nl = c->nl;
  size_t ob = len * (nl + 2) * 8;
  uint64_t *d = (uint64_t *) g_s2.dev (ob);
  uint64_t *h = (uint64_t *) g_s2.host (ob);
  k_to<<<nblocks (len), TPB, 0, s>>> (base, stride, c->sp_num, len, v, *c, d);
  check_launch ("download");
  cudaMemcpyAsync (h, d, ob, cudaMemcpyDeviceToHost, s);
  sync (s);
  for (size_t k = 0; k < len; k++)
    {
      mp_limb_t *l = mpz_limbs_write (r[k], nl + 2);
      memcpy (l, h + k * (nl + 2), (nl + 2) * 8);
      mpz_limbs_finish (r[k], nl + 2);
    }
}

void
normalise (const gpu_ctx *c, uint64_t *base, size_t stride, struct view v,
           size_t len, cudaStream_t s)
{
  uint64_t *tt = (uint64_t *) g_s3.dev (len * c->sp_num * 8);
  k_norm<<<nblocks (len), TPB, 0, s>>> (base, stride, c->sp_num, len, v, *c,
                                        tt);
  check_launch ("normalise");
}

thread_local scratch g_tree;

} /* namespace */

extern "C" {

/* See gpu_ntt.h. The layout is that of ntt_PolyFromRoots: at level m the
   two factors of pair q are at words 4mq and 4mq + 2m, their product goes
   to 4mq, which is where the next level wants it. */
int
gpu_poly_from_roots (mpzv_t r, mpzv_t a, spv_size_t len, mpzv_t *Tree,
                     mpzspm_t mpzspm)
{
  const gpu_ctx *c = (const gpu_ctx *) mpzspm->gpu;
  if (!c || len < 2 || (len & (len - 1)) || 2 * len > ((size_t) 1 << c->logK))
    return declined ();
  double t0 = g_stats ? now () : 0.0;
  unsigned sp = c->sp_num;
  size_t stride = 2 * len;
  uint64_t *x = (uint64_t *) g_tree.dev ((size_t) sp * stride * 8);
  cudaStream_t s = cudaStreamPerThread;
  unsigned lgk = 0;
  while (((size_t) 1 << lgk) < len)
    lgk++;

  if (!upload (c, a, len, x, stride, (struct view) {1, 2}, s))
    return declined ();
  if (Tree)
    {
      /* the leaves: the roots themselves (list_set in the CPU code); now,
         since r may be a */
      for (size_t k = 0; k < len; k++)
        mpz_set (Tree[lgk - 1][k], a[k]);
    }
  for (size_t m = 1; m < len; m *= 2)
    {
      size_t L = 2 * m, nb = len / m; /* nb transforms of length L */
      k_prep<<<nblocks ((size_t) sp * nb * L), TPB, 0, s>>> (x, stride, sp,
          nb, L, m, L, 1, c->P);
      check_launch ("tree prep");
      dif (c, x, stride, nb, L, L, s);
      k_pwmul<<<nblocks ((size_t) sp * (nb / 2) * L), TPB, 0, s>>> (x,
          stride, 2 * L, x, stride, 2 * L, x + L, stride, 2 * L, sp, nb / 2,
          L, c->P, c->D);
      check_launch ("tree pwmul");
      dit (c, x, stride, nb / 2, 2 * L, L, s);
      k_post<<<nblocks ((size_t) sp * (nb / 2) * L), TPB, 0, s>>> (x, stride,
          sp, nb / 2, 2 * L, L, L, c->P, c->D);
      check_launch ("tree post");
      struct view lv = {L, 2 * L};
      if (2 * m < len)
        {
          if (Tree)
            {
              unsigned j = 0;
              while (((size_t) 1 << j) < m)
                j++;
              download (c, x, stride, lv, Tree[lgk - 2 - j], len, s);
            }
          normalise (c, x, stride, lv, len, s);
        }
    }
  download (c, x, stride, CONTIG, r, len, s);
  counted (t0);
  return 1;
}

/* See gpu_ntt.h. */
int
gpu_polyeval_levels (mpzspv_t y, spv_size_t len, mpzv_t *Tree, mpzv_t T,
                     mpzspm_t mpzspm)
{
  const gpu_ctx *c = (const gpu_ctx *) mpzspm->gpu;
  if (!c || len < 2 || (len & (len - 1)) || 2 * len > ((size_t) 1 << c->logK))
    return declined ();
  unsigned sp = c->sp_num;
  size_t sy = stride_of (y, sp);
  if (!sy)
    return declined ();
  double t0 = g_stats ? now () : 0.0;
  cudaStream_t s = cudaStreamPerThread;
  uint64_t *x = (uint64_t *) g_tree.dev ((size_t) sp * 3 * len * 8);
  uint64_t *z = x + (size_t) sp * 2 * len; /* sp rows of len words */
  unsigned lgk = 0;
  while (((size_t) 1 << lgk) < len)
    lgk++;
  for (unsigned lv = 0; lv < lgk; lv++)
    {
      size_t m = len >> (lv + 1), L = 2 * m, nodes = len / L;
      if (!upload (c, Tree[lv], len, z, len, CONTIG, s))
        return declined (); /* (nothing written to T yet) */
      k_revmonic<<<nblocks ((size_t) sp * 2 * len), TPB, 0, s>>> (x, 2 * len,
          z, len, sp, len, m);
      check_launch ("revmonic");
      dif (c, x, 2 * len, 2 * nodes, L, L, s);
      dif (c, y[0], sy, nodes, L, L, s);
      for (int ch = 0; ch < 2; ch++)
        {
          k_pwmul<<<nblocks ((size_t) sp * nodes * L), TPB, 0, s>>> (
              x + ch * L, 2 * len, 2 * L, x + ch * L, 2 * len, 2 * L, y[0], sy,
              L, sp, nodes, L, c->P, c->D);
          check_launch ("polyeval pwmul");
        }
      dit (c, x, 2 * len, 2 * nodes, L, L, s);
      k_post<<<nblocks ((size_t) sp * 2 * len), TPB, 0, s>>> (x, 2 * len, sp,
          2 * nodes, L, L, 0, c->P, c->D);
      check_launch ("polyeval post");
      k_mid<<<nblocks ((size_t) sp * len), TPB, 0, s>>> (y[0], sy, x, 2 * len,
          sp, len, m);
      check_launch ("polyeval mid");
      if (lv + 1 < lgk)
        normalise (c, y[0], sy, CONTIG, len, s);
    }
  download (c, y[0], sy, CONTIG, T, len, s);
  counted (t0);
  return 1;
}

} /* extern "C" */

/* ------------------------------------------------------------------ */
/* Batched stage 2 (gpu_ntt.h): C curves, every polynomial a block of a  */
/* row, curve-major (curve c's words at c * L in every row).            */

struct gpu_s2batch
{
  const gpu_ctx *c;
  unsigned C, lgk;
  size_t len;
  uint64_t *tree; /* lgk levels of C len words per row */
  uint64_t *F, *Fn, *R, *iF, *H, *G; /* C len (iF: C 2len) per row */
  uint64_t *x, *A, *B;               /* scratch, C 2len per row */
  uint64_t *tt;                      /* normalisation scratch */
  cudaStream_t s;
};

namespace {

typedef struct gpu_s2batch S2;

/* sp rows of n words */
uint64_t *
s2alloc (const S2 *b, size_t n, size_t *total)
{
  uint64_t *p = NULL;
  size_t bytes = (size_t) b->c->sp_num * n * 8;
  if (cudaMalloc (&p, bytes) != cudaSuccess)
    return NULL;
  *total += bytes;
  return p;
}

void
s2norm (S2 *b, uint64_t *base, size_t stride, struct view v, size_t count)
{
  k_norm<<<nblocks (count), TPB, 0, b->s>>> (base, stride, b->c->sp_num,
      count, v, *b->c, b->tt);
  check_launch ("batch normalise");
}

void
s2move (S2 *b, uint64_t *dst, size_t sd, size_t dbs, const uint64_t *src,
        size_t ss, size_t sbs, size_t nt, size_t L, size_t j0, size_t j1,
        long a, int dir, int neg, int one0)
{
  k_move<<<nblocks ((size_t) b->c->sp_num * nt * L), TPB, 0, b->s>>> (dst,
      sd, dbs, src, ss, sbs, b->c->sp_num, nt, L, j0, j1, a, dir, neg, one0,
      b->c->P);
  check_launch ("batch move");
}

/* r[t rbs ..] = x[t xbs ..] * y[t ybs ..] mod (X^L - 1) for t < nt: both
   already holding L words (zero-padded), transformed in place; r gets the
   cyclic product (1/L applied). */
void
s2cmul (S2 *b, uint64_t *r, size_t sr, size_t rbs, uint64_t *x, size_t sx,
        size_t xbs, uint64_t *y, size_t sy, size_t ybs, size_t nt, size_t L,
        bool ytransformed)
{
  const gpu_ctx *c = b->c;
  dif (c, x, sx, nt, xbs, L, b->s);
  if (!ytransformed)
    dif (c, y, sy, nt, ybs, L, b->s);
  k_pwmul<<<nblocks ((size_t) c->sp_num * nt * L), TPB, 0, b->s>>> (r, sr,
      rbs, x, sx, xbs, y, sy, ybs, c->sp_num, nt, L, c->P, c->D);
  check_launch ("batch pwmul");
  dit (c, r, sr, nt, rbs, L, b->s);
  k_post<<<nblocks ((size_t) c->sp_num * nt * L), TPB, 0, b->s>>> (r, sr,
      c->sp_num, nt, rbs, L, 0, c->P, c->D);
  check_launch ("batch post");
}

/* The product of the C len roots (W words each, at droots) per curve, the
   root product tree in residues: the result (normalised, monic implicit)
   to out (C len per row, curve-major); the levels to b->tree if keep. */
void
s2tree (S2 *b, const void *droots, unsigned W, uint64_t *out, bool keep)
{
  const gpu_ctx *c = b->c;
  const unsigned sp = c->sp_num;
  const size_t len = b->len, Ctot = (size_t) b->C * len, sx = 2 * Ctot;
  uint64_t *x = b->x;
  k_from_dev<<<nblocks ((size_t) sp * Ctot), TPB, 0, b->s>>> (x, sx, sp,
      Ctot, (struct view) {1, 2}, (const uint64_t *) droots, W, c->P, c->D,
      c->Bm);
  check_launch ("batch roots");
  if (keep) /* the leaves: the roots */
    s2move (b, b->tree + (size_t) (b->lgk - 1) * sp * Ctot, Ctot, 1, x, sx,
            2, Ctot, 1, 0, 1, 0, 1, 0, 0);
  for (size_t m = 1; m < len; m *= 2)
    {
      size_t L = 2 * m, nb = 2 * Ctot / L;
      k_prep<<<nblocks ((size_t) sp * nb * L), TPB, 0, b->s>>> (x, sx, sp,
          nb, L, m, L, 1, c->P);
      check_launch ("tree prep");
      dif (c, x, sx, nb, L, L, b->s);
      k_pwmul<<<nblocks ((size_t) sp * (nb / 2) * L), TPB, 0, b->s>>> (x, sx,
          2 * L, x, sx, 2 * L, x + L, sx, 2 * L, sp, nb / 2, L, c->P, c->D);
      check_launch ("tree pwmul");
      dit (c, x, sx, nb / 2, 2 * L, L, b->s);
      k_post<<<nblocks ((size_t) sp * (nb / 2) * L), TPB, 0, b->s>>> (x, sx,
          sp, nb / 2, 2 * L, L, L, c->P, c->D);
      check_launch ("tree post");
      struct view lv = {L, 2 * L};
      s2norm (b, x, sx, lv, Ctot);
      if (keep && 2 * m < len)
        {
          unsigned j = 0;
          while (((size_t) 1 << j) < m)
            j++;
          /* the products of this level, contiguous: Tree[lgk - 2 - j] */
          s2move (b, b->tree + (size_t) (b->lgk - 2 - j) * sp * Ctot, Ctot,
                  L, x, sx, 2 * L, Ctot / L, L, 0, L, 0, 1, 0, 0);
        }
    }
  /* the root product of curve c is at x[c 2len ..] */
  s2move (b, out, Ctot, len, x, sx, 2 * len, b->C, len, 0, len, 0, 1, 0, 0);
}

} /* namespace */

extern "C" {

size_t
gpu_s2batch_bytes (mpzspm_t mpzspm, unsigned C, unsigned long len)
{
  unsigned lgk = 0;
  while (((size_t) 1 << lgk) < len)
    lgk++;
  /* tree lgk, F Fn R H G 5, iF x A B 8 (C len words per row), tt */
  return (size_t) mpzspm->sp_num * C * len * 8 * (lgk + 13 + 1);
}

gpu_s2batch_t *
gpu_s2batch_new (mpzspm_t mpzspm, unsigned C, unsigned long len)
{
  const gpu_ctx *c = (const gpu_ctx *) mpzspm->gpu;
  if (!c || len < 2 || (len & (len - 1)) || 2 * len > ((size_t) 1 << c->logK))
    return NULL;
  S2 *b = (S2 *) calloc (1, sizeof (S2));
  b->c = c;
  b->C = C;
  b->len = len;
  b->s = cudaStreamPerThread;
  while (((size_t) 1 << b->lgk) < len)
    b->lgk++;
  size_t Ctot = (size_t) C * len, tot = 0;
  b->tree = s2alloc (b, b->lgk * Ctot, &tot);
  b->F = s2alloc (b, Ctot, &tot);
  b->Fn = s2alloc (b, Ctot, &tot);
  b->R = s2alloc (b, Ctot, &tot);
  b->H = s2alloc (b, Ctot, &tot);
  b->G = s2alloc (b, Ctot, &tot);
  b->iF = s2alloc (b, 2 * Ctot, &tot);
  b->x = s2alloc (b, 2 * Ctot, &tot);
  b->A = s2alloc (b, 2 * Ctot, &tot);
  b->B = s2alloc (b, 2 * Ctot, &tot);
  b->tt = s2alloc (b, 2 * Ctot, &tot);
  if (!b->tree || !b->F || !b->Fn || !b->R || !b->H || !b->G || !b->iF ||
      !b->x || !b->A || !b->B || !b->tt)
    {
      gpu_s2batch_free (b);
      return NULL;
    }
  return b;
}

void
gpu_s2batch_free (gpu_s2batch_t *b)
{
  if (!b)
    return;
  cudaFree (b->tree);
  cudaFree (b->F);
  cudaFree (b->Fn);
  cudaFree (b->R);
  cudaFree (b->H);
  cudaFree (b->G);
  cudaFree (b->iF);
  cudaFree (b->x);
  cudaFree (b->A);
  cudaFree (b->B);
  cudaFree (b->tt);
  free (b);
}

void
gpu_s2batch_F (gpu_s2batch_t *b, const void *droots, unsigned W)
{
  const unsigned sp = b->c->sp_num;
  const size_t len = b->len, Ctot = (size_t) b->C * len;
  const size_t s1 = Ctot, s2 = 2 * Ctot; /* row strides */
  double t0 = g_stats ? now () : 0.0;
  s2tree (b, droots, W, b->F, true);

  /* Fn: F mod (X^len - 1) with its leading 1 (at X^0), transformed */
  s2move (b, b->Fn, s1, len, b->F, s1, len, b->C, len, 0, len, 0, 1, 0, 0);
  k_prep<<<nblocks ((size_t) sp * Ctot), TPB, 0, b->s>>> (b->Fn, s1, sp,
      b->C, len, len, len, 1, b->c->P);
  check_launch ("Fn prep");
  dif (b->c, b->Fn, s1, b->C, len, len, b->s);

  /* R = rev(F)^-1 mod X^len by Newton, rev(F) = 1 + F[len-1] X + ...:
     R_2m = R_m - X^m (R_m e1 mod X^m), rev(F) R_m = 1 + X^m e1 + ... */
  s2move (b, b->R, s1, len, b->R, s1, len, b->C, len, 0, 0, 0, 1, 0, 1);
  for (size_t m = 1; m < len; m *= 2)
    {
      size_t L = 2 * m;
      /* A = rev(F) mod X^2m, B = R_m, cyclic product of length 2m: its
         words m .. 2m-1 are those of the true product */
      s2move (b, b->A, s2, 2 * len, b->F, s1, len, b->C, L, 1, L, (long) len,
              -1, 0, 1);
      s2move (b, b->B, s2, 2 * len, b->R, s1, len, b->C, L, 0, m, 0, 1, 0, 0);
      s2cmul (b, b->A, s2, 2 * len, b->A, s2, 2 * len, b->B, s2, 2 * len,
              b->C, L, false);
      s2norm (b, b->A + m, s2, (struct view) {m, 2 * len}, (size_t) b->C * m);
      /* B = e1 (words m.. of A), A = R_m; their product mod X^m */
      s2move (b, b->B, s2, 2 * len, b->A, s2, 2 * len, b->C, L, 0, m,
              (long) m, 1, 0, 0);
      s2move (b, b->A, s2, 2 * len, b->R, s1, len, b->C, L, 0, m, 0, 1, 0, 0);
      s2cmul (b, b->A, s2, 2 * len, b->A, s2, 2 * len, b->B, s2, 2 * len,
              b->C, L, false);
      s2norm (b, b->A, s2, (struct view) {m, 2 * len}, (size_t) b->C * m);
      /* R[m .. 2m) = -A[0 .. m) */
      s2move (b, b->R + m, s1, len, b->A, s2, 2 * len, b->C, m, 0, m, 0, 1,
              1, 0);
    }

  /* iF = invF = Quo(X^(2len-1), F) = rev(R): invF[j] = R[len-1-j],
     transformed at length 2len (for ntt_polyevalT's first product) */
  s2move (b, b->iF, s2, 2 * len, b->R, s1, len, b->C, 2 * len, 0, len,
          (long) len - 1, -1, 0, 0);
  dif (b->c, b->iF, s2, b->C, 2 * len, 2 * len, b->s);
  sync (b->s);
  counted (t0);
}

void
gpu_s2batch_G (gpu_s2batch_t *b, const void *droots, unsigned W, int first)
{
  const unsigned sp = b->c->sp_num;
  const size_t len = b->len, Ctot = (size_t) b->C * len;
  const size_t s1 = Ctot, s2 = 2 * Ctot;
  double t0 = g_stats ? now () : 0.0;
  (void) sp;
  s2tree (b, droots, W, b->G, false);
  /* G - F: degree < len (both monic) */
  k_addsub<<<nblocks ((size_t) b->c->sp_num * Ctot), TPB, 0, b->s>>> (
      first ? b->H : b->G, s1, len, b->G, s1, len, b->F, s1, len,
      b->c->sp_num, b->C, len, len, -1, b->c->P);
  check_launch ("G - F");
  if (first)
    {
      s2norm (b, b->H, s1, CONTIG, Ctot);
      sync (b->s);
      counted (t0);
      return;
    }
  s2norm (b, b->G, s1, CONTIG, Ctot);
  /* T = H (G - F), 2len - 1 words, in A */
  s2move (b, b->A, s2, 2 * len, b->H, s1, len, b->C, 2 * len, 0, len, 0, 1,
          0, 0);
  s2move (b, b->B, s2, 2 * len, b->G, s1, len, b->C, 2 * len, 0, len, 0, 1,
          0, 0);
  s2cmul (b, b->A, s2, 2 * len, b->A, s2, 2 * len, b->B, s2, 2 * len, b->C,
          2 * len, false);
  s2norm (b, b->A, s2, (struct view) {2 * len - 1, 2 * len},
          (size_t) b->C * (2 * len - 1));
  /* q = Quo(T, F): rev(q) = rev(T) R mod X^(len-1), rev(T)[j] = T[2len-2-j] */
  s2move (b, b->B, s2, 2 * len, b->A, s2, 2 * len, b->C, 2 * len, 0, len - 1,
          (long) (2 * len - 2), -1, 0, 0);
  s2move (b, b->x, s2, 2 * len, b->R, s1, len, b->C, 2 * len, 0, len - 1, 0,
          1, 0, 0);
  s2cmul (b, b->B, s2, 2 * len, b->B, s2, 2 * len, b->x, s2, 2 * len, b->C,
          2 * len, false);
  s2norm (b, b->B, s2, (struct view) {len - 1, 2 * len},
          (size_t) b->C * (len - 1));
  /* q (len - 1 words) at x, padded to len: q[j] = rev(q)[len-2-j] */
  s2move (b, b->x, s2, 2 * len, b->B, s2, 2 * len, b->C, len, 0, len - 1,
          (long) (len - 2), -1, 0, 0);
  /* q F mod (X^len - 1) */
  s2cmul (b, b->x, s2, 2 * len, b->x, s2, 2 * len, b->Fn, s1, len, b->C, len,
          true);
  /* H = (T mod (X^len - 1)) - q F mod (X^len - 1) */
  k_addsub<<<nblocks ((size_t) b->c->sp_num * Ctot), TPB, 0, b->s>>> (b->H,
      s1, len, b->A, s2, 2 * len, b->A + len, s2, 2 * len, b->c->sp_num,
      b->C, len, len - 1, 1, b->c->P);
  check_launch ("fold T");
  k_addsub<<<nblocks ((size_t) b->c->sp_num * Ctot), TPB, 0, b->s>>> (b->H,
      s1, len, b->H, s1, len, b->x, s2, 2 * len, b->c->sp_num, b->C, len,
      len, -1, b->c->P);
  check_launch ("H");
  s2norm (b, b->H, s1, CONTIG, Ctot);
  sync (b->s);
  counted (t0);
}

void
gpu_s2batch_finish (gpu_s2batch_t *b, mpz_t *prod)
{
  const gpu_ctx *c = b->c;
  const unsigned sp = c->sp_num;
  const size_t len = b->len, Ctot = (size_t) b->C * len;
  const size_t s1 = Ctot, s2 = 2 * Ctot;
  double t0 = g_stats ? now () : 0.0;
  /* y = rev(high(H invF)): the words len-1 .. 2len-2 of H iF (length 2len) */
  s2move (b, b->A, s2, 2 * len, b->H, s1, len, b->C, 2 * len, 0, len, 0, 1,
          0, 0);
  s2cmul (b, b->A, s2, 2 * len, b->A, s2, 2 * len, b->iF, s2, 2 * len, b->C,
          2 * len, true);
  s2norm (b, b->A + len - 1, s2, (struct view) {len, 2 * len}, Ctot);
  uint64_t *y = b->G; /* C len per row */
  s2move (b, y, s1, len, b->A, s2, 2 * len, b->C, len, 0, len,
          (long) (2 * len - 2), -1, 0, 0);
  /* the levels of the transposed product tree */
  uint64_t *x = b->x;
  for (unsigned lv = 0; lv < b->lgk; lv++)
    {
      size_t m = len >> (lv + 1), L = 2 * m, nodes = Ctot / L;
      const uint64_t *z = b->tree + (size_t) lv * sp * Ctot;
      k_revmonic<<<nblocks ((size_t) sp * 2 * Ctot), TPB, 0, b->s>>> (x, s2,
          z, s1, sp, Ctot, m);
      check_launch ("batch revmonic");
      dif (c, x, s2, 2 * nodes, L, L, b->s);
      dif (c, y, s1, nodes, L, L, b->s);
      for (int ch = 0; ch < 2; ch++)
        {
          k_pwmul<<<nblocks ((size_t) sp * nodes * L), TPB, 0, b->s>>> (
              x + ch * L, s2, 2 * L, x + ch * L, s2, 2 * L, y, s1, L, sp,
              nodes, L, c->P, c->D);
          check_launch ("batch polyeval pwmul");
        }
      dit (c, x, s2, 2 * nodes, L, L, b->s);
      k_post<<<nblocks ((size_t) sp * 2 * Ctot), TPB, 0, b->s>>> (x, s2, sp,
          2 * nodes, L, L, 0, c->P, c->D);
      check_launch ("batch polyeval post");
      k_mid<<<nblocks ((size_t) sp * Ctot), TPB, 0, b->s>>> (y, s1, x, s2,
          sp, Ctot, m);
      check_launch ("batch polyeval mid");
      s2norm (b, y, s1, CONTIG, Ctot);
    }
  /* the product of the len values of each curve */
  for (size_t h = len / 2; h >= 1; h /= 2)
    {
      k_pwmul<<<nblocks ((size_t) sp * b->C * h), TPB, 0, b->s>>> (y, s1, len,
          y, s1, len, y + h, s1, len, sp, b->C, h, c->P, c->D);
      check_launch ("batch product");
      s2norm (b, y, s1, (struct view) {h, len}, (size_t) b->C * h);
    }
  download (c, y, s1, (struct view) {1, len}, prod, b->C, b->s);
  counted (t0);
}

size_t
gpu_free_memory (void)
{
  size_t fr = 0, tot = 0;
  if (cudaMemGetInfo (&fr, &tot) != cudaSuccess)
    return 0;
  return fr;
}

} /* extern "C" */
