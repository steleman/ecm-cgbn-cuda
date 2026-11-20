/* gpu_roots.cu - the roots of F and G of ECM stage 2 on an NVIDIA GPU.

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

/* See gpu_roots.h. One CGBN instance per point operation; every value in
   Montgomery form and fully reduced (below n), so n may fill the width. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <gmp.h>
#include <cgbn.h>
#include <cuda_runtime.h>

#include "gpu_roots.h"

namespace {

template <uint32_t tpi, uint32_t bits>
struct rparams
{
  static const uint32_t TPB = 0;
  static const uint32_t MAX_ROTATION = 4;
  static const uint32_t SHM_LIMIT = 0;
  static const bool CONSTANT_TIME = false;
  static const uint32_t TPI = tpi;
  static const uint32_t BITS = bits;
};

const uint32_t TPB = 128;

void
fail (const char *what, cudaError_t e)
{
  fprintf (stderr, "GPU roots: %s: %s\n", what, cudaGetErrorString (e));
  abort ();
}

void
ck (cudaError_t e, const char *what)
{
  if (e != cudaSuccess)
    fail (what, e);
}

/* The modulus, and the curves' a, b3 = 3b (Montgomery form), in device
   memory: curve c's at a[c], b3[c]. */
template <uint32_t BITS>
struct dcurve
{
  cgbn_mem_t<BITS> n;
};

/* ------------------------------------------------------------------ */
/* Device arithmetic                                                    */

template <class env_t>
struct ar
{
  typedef typename env_t::cgbn_t bn;
  env_t &e;
  bn n;
  uint32_t np0;

  __device__ ar (env_t &env, const cgbn_mem_t<env_t::BITS> *nm) : e (env)
  {
    bn one;
    cgbn_load (e, n, (cgbn_mem_t<env_t::BITS> *) nm);
    cgbn_set_ui32 (e, one, 1);
    np0 = cgbn_bn2mont (e, one, one, n);
  }
  __device__ void mul (bn &r, const bn &a, const bn &b)
  {
    cgbn_mont_mul (e, r, a, b, n, np0);
    if (cgbn_compare (e, r, n) >= 0)
      cgbn_sub (e, r, r, n);
  }
  __device__ void add (bn &r, const bn &a, const bn &b)
  {
    int c = cgbn_add (e, r, a, b);
    if (c || cgbn_compare (e, r, n) >= 0)
      cgbn_sub (e, r, r, n);
  }
  __device__ void sub (bn &r, const bn &a, const bn &b)
  {
    if (cgbn_sub (e, r, a, b))
      cgbn_add (e, r, r, n);
  }
  __device__ void one (bn &r)
  {
    cgbn_set_ui32 (e, r, 1);
    cgbn_bn2mont (e, r, r, n);
  }
};

/* P3 = P1 + P2, complete (Renes, Costello, Batina 2016, Algorithm 1),
   a general, b3 = 3b. P3 may alias neither input. */
template <class env_t>
__device__ void
padd (ar<env_t> &m, typename env_t::cgbn_t &X3, typename env_t::cgbn_t &Y3,
      typename env_t::cgbn_t &Z3, const typename env_t::cgbn_t &X1,
      const typename env_t::cgbn_t &Y1, const typename env_t::cgbn_t &Z1,
      const typename env_t::cgbn_t &X2, const typename env_t::cgbn_t &Y2,
      const typename env_t::cgbn_t &Z2, const typename env_t::cgbn_t &a,
      const typename env_t::cgbn_t &b3)
{
  typename env_t::cgbn_t t0, t1, t2, t3, t4, t5;
  m.mul (t0, X1, X2);
  m.mul (t1, Y1, Y2);
  m.mul (t2, Z1, Z2);
  m.add (t3, X1, Y1);
  m.add (t4, X2, Y2);
  m.mul (t3, t3, t4);
  m.add (t4, t0, t1);
  m.sub (t3, t3, t4);
  m.add (t4, X1, Z1);
  m.add (t5, X2, Z2);
  m.mul (t4, t4, t5);
  m.add (t5, t0, t2);
  m.sub (t4, t4, t5);
  m.add (t5, Y1, Z1);
  m.add (X3, Y2, Z2);
  m.mul (t5, t5, X3);
  m.add (X3, t1, t2);
  m.sub (t5, t5, X3);
  m.mul (Z3, a, t4);
  m.mul (X3, b3, t2);
  m.add (Z3, X3, Z3);
  m.sub (X3, t1, Z3);
  m.add (Z3, t1, Z3);
  m.mul (Y3, X3, Z3);
  m.add (t1, t0, t0);
  m.add (t1, t1, t0);
  m.mul (t2, a, t2);
  m.mul (t4, b3, t4);
  m.add (t1, t1, t2);
  m.sub (t2, t0, t2);
  m.mul (t2, a, t2);
  m.add (t4, t4, t2);
  m.mul (t0, t1, t4);
  m.add (Y3, Y3, t0);
  m.mul (t0, t5, t4);
  m.mul (X3, t3, X3);
  m.sub (X3, X3, t0);
  m.mul (t0, t3, t1);
  m.mul (Z3, t5, Z3);
  m.add (Z3, Z3, t0);
}

/* ------------------------------------------------------------------ */
/* Kernels. C curves; curve c's table is points [c cnt, (c+1) cnt) of   */
/* 3 words (X, Y, Z) each, cnt = nr (S + 1).                            */

#define ENV                                                                 \
  typedef cgbn_context_t<TPI, rparams<TPI, BITS> > context_t;               \
  typedef cgbn_env_t<context_t, BITS> env_t;                                \
  typedef typename env_t::cgbn_t bn;                                        \
  uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;            \
  context_t ctx (cgbn_no_checks);                                           \
  env_t env (ctx);

/* pw[c][t] = 2^t P_c, t <= maxbit (Montgomery, projective); acc[c] = the
   product of 2 Y over the doublings (ecm2.c inverts 2 s.y). One instance
   per curve; P_c affine canonical. */
template <uint32_t TPI, uint32_t BITS>
__global__ void
k_powers (const dcurve<BITS> *c, uint32_t C, const cgbn_mem_t<BITS> *ca,
          const cgbn_mem_t<BITS> *cb3, const cgbn_mem_t<BITS> *px,
          const cgbn_mem_t<BITS> *py, uint32_t maxbit, cgbn_mem_t<BITS> *pw,
          cgbn_mem_t<BITS> *acc)
{
  ENV
  if (inst >= C)
    return;
  ar<env_t> m (env, &c->n);
  bn X, Y, Z, X2, Y2, Z2, a, b3, ac, t;
  cgbn_load (env, a, (cgbn_mem_t<BITS> *) &ca[inst]);
  cgbn_load (env, b3, (cgbn_mem_t<BITS> *) &cb3[inst]);
  cgbn_load (env, X, (cgbn_mem_t<BITS> *) &px[inst]);
  cgbn_load (env, Y, (cgbn_mem_t<BITS> *) &py[inst]);
  cgbn_bn2mont (env, X, X, m.n);
  cgbn_bn2mont (env, Y, Y, m.n);
  m.one (Z);
  m.one (ac);
  cgbn_mem_t<BITS> *w = pw + (size_t) inst * 3 * (maxbit + 1);
  for (uint32_t t2 = 0; t2 <= maxbit; t2++)
    {
      cgbn_store (env, &w[3 * t2], X);
      cgbn_store (env, &w[3 * t2 + 1], Y);
      cgbn_store (env, &w[3 * t2 + 2], Z);
      if (t2 == maxbit)
        break;
      m.add (t, Y, Y);
      m.mul (ac, ac, t);
      padd (m, X2, Y2, Z2, X, Y, Z, X, Y, Z, a, b3);
      cgbn_set (env, X, X2);
      cgbn_set (env, Y, Y2);
      cgbn_set (env, Z, Z2);
    }
  cgbn_mont2bn (env, ac, ac, m.n, m.np0);
  cgbn_store (env, &acc[inst], ac);
}

/* R[c][i] = q_i P_c = sum of 2^t P_c over the bits t of |q_i| (q: w words
   each, least significant first, the same for every curve), negated if
   neg[i]; acc[c][i] *= x(2^t P) - x(partial) for every addition to a
   partial result (multiplyW2n's denominators), projectively. */
template <uint32_t TPI, uint32_t BITS>
__global__ void
k_mulw (const dcurve<BITS> *c, uint32_t C, uint32_t cnt,
        const cgbn_mem_t<BITS> *ca, const cgbn_mem_t<BITS> *cb3,
        const uint32_t *q, uint32_t w, const uint8_t *neg, uint32_t maxbit,
        const cgbn_mem_t<BITS> *pw, cgbn_mem_t<BITS> *R,
        cgbn_mem_t<BITS> *acc)
{
  ENV
  if (inst >= C * cnt)
    return;
  uint32_t cv = inst / cnt, ii = inst % cnt;
  ar<env_t> m (env, &c->n);
  bn X, Y, Z, X2, Y2, Z2, X3, Y3, Z3, a, b3, ac, t, u;
  cgbn_load (env, a, (cgbn_mem_t<BITS> *) &ca[cv]);
  cgbn_load (env, b3, (cgbn_mem_t<BITS> *) &cb3[cv]);
  cgbn_set_ui32 (env, X, 0);
  m.one (Y);
  cgbn_set_ui32 (env, Z, 0); /* the neutral element (0:1:0) */
  m.one (ac);
  const uint32_t *qi = q + (size_t) ii * w;
  const cgbn_mem_t<BITS> *p = pw + (size_t) cv * 3 * (maxbit + 1);
  bool have = false;
  for (uint32_t k = 0; k < w; k++)
    {
      uint32_t word = qi[k];
      while (word)
        {
          uint32_t b = __ffs (word) - 1;
          word &= word - 1;
          uint32_t t2 = 32 * k + b;
          cgbn_load (env, X2, (cgbn_mem_t<BITS> *) &p[3 * t2]);
          cgbn_load (env, Y2, (cgbn_mem_t<BITS> *) &p[3 * t2 + 1]);
          cgbn_load (env, Z2, (cgbn_mem_t<BITS> *) &p[3 * t2 + 2]);
          if (have)
            {
              m.mul (t, X2, Z);
              m.mul (u, X, Z2);
              m.sub (t, t, u);
              m.mul (ac, ac, t);
            }
          padd (m, X3, Y3, Z3, X, Y, Z, X2, Y2, Z2, a, b3);
          cgbn_set (env, X, X3);
          cgbn_set (env, Y, Y3);
          cgbn_set (env, Z, Z3);
          have = true;
        }
    }
  if (neg[ii])
    {
      cgbn_set_ui32 (env, t, 0);
      m.sub (Y, t, Y);
    }
  cgbn_store (env, &R[3 * (size_t) inst], X);
  cgbn_store (env, &R[3 * (size_t) inst + 1], Y);
  cgbn_store (env, &R[3 * (size_t) inst + 2], Z);
  cgbn_load (env, u, &acc[inst]);
  m.mul (u, u, ac);
  cgbn_store (env, &acc[inst], u);
}

/* The last point of progression 0 to the last point of every other one,
   for every curve (one block per copy). */
template <uint32_t BITS>
__global__ void
k_copytop (cgbn_mem_t<BITS> *T, uint32_t C, uint32_t nr, uint32_t S)
{
  uint32_t b = blockIdx.x; /* C (nr - 1) blocks */
  if (b >= C * (nr - 1))
    return;
  uint32_t cv = b / (nr - 1), i = b % (nr - 1) + 1;
  size_t base = (size_t) cv * nr * (S + 1);
  const uint32_t *src = (const uint32_t *) &T[3 * (base + S)];
  uint32_t *dst = (uint32_t *) &T[3 * (base + (size_t) i * (S + 1) + S)];
  for (uint32_t k = threadIdx.x; k < 3 * BITS / 32; k += blockDim.x)
    dst[k] = src[k];
}

/* One step of every progression of every curve: nxt[e] = cur[e] + cur[e+1]
   for e = c cnt + i (S+1) + j, j < S; nxt[... + S] = cur[...].
   acc[e] *= X1 Z2 - X2 Z1 unless a point is the neutral element. */
template <uint32_t TPI, uint32_t BITS>
__global__ void
k_update (const dcurve<BITS> *c, uint32_t total, uint32_t cnt, uint32_t S,
          const cgbn_mem_t<BITS> *ca, const cgbn_mem_t<BITS> *cb3,
          const cgbn_mem_t<BITS> *cur, cgbn_mem_t<BITS> *nxt,
          cgbn_mem_t<BITS> *acc)
{
  ENV
  if (inst >= total)
    return;
  uint32_t j = (inst % cnt) % (S + 1);
  if (j == S)
    {
      for (int k = 0; k < 3; k++)
        nxt[3 * (size_t) inst + k] = cur[3 * (size_t) inst + k];
      return;
    }
  uint32_t cv = inst / cnt;
  ar<env_t> m (env, &c->n);
  bn X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, a, b3, t, u;
  cgbn_load (env, a, (cgbn_mem_t<BITS> *) &ca[cv]);
  cgbn_load (env, b3, (cgbn_mem_t<BITS> *) &cb3[cv]);
  const cgbn_mem_t<BITS> *q = cur + 3 * (size_t) inst;
  cgbn_load (env, X1, (cgbn_mem_t<BITS> *) &q[0]);
  cgbn_load (env, Y1, (cgbn_mem_t<BITS> *) &q[1]);
  cgbn_load (env, Z1, (cgbn_mem_t<BITS> *) &q[2]);
  cgbn_load (env, X2, (cgbn_mem_t<BITS> *) &q[3]);
  cgbn_load (env, Y2, (cgbn_mem_t<BITS> *) &q[4]);
  cgbn_load (env, Z2, (cgbn_mem_t<BITS> *) &q[5]);
  if (!cgbn_equals_ui32 (env, Z1, 0) && !cgbn_equals_ui32 (env, Z2, 0))
    {
      m.mul (t, X2, Z1);
      m.mul (u, X1, Z2);
      m.sub (t, t, u);
      cgbn_load (env, u, &acc[inst]);
      m.mul (u, u, t);
      cgbn_store (env, &acc[inst], u);
    }
  padd (m, X3, Y3, Z3, X1, Y1, Z1, X2, Y2, Z2, a, b3);
  cgbn_store (env, &nxt[3 * (size_t) inst], X3);
  cgbn_store (env, &nxt[3 * (size_t) inst + 1], Y3);
  cgbn_store (env, &nxt[3 * (size_t) inst + 2], Z3);
}

/* For every curve c and k < count: roots (X, Z)[c nout + r0 + k] = the
   first point of progression p[k] (one block each). */
template <uint32_t BITS>
__global__ void
k_extract (const cgbn_mem_t<BITS> *T, uint32_t C, uint32_t cnt, uint32_t S,
           const int *p, uint32_t count, size_t nout, size_t r0,
           cgbn_mem_t<BITS> *rx, cgbn_mem_t<BITS> *rz)
{
  uint32_t b = blockIdx.x;
  if (b >= C * count)
    return;
  uint32_t cv = b / count, k = b % count;
  size_t e = (size_t) cv * cnt + (size_t) p[k] * (S + 1);
  const uint32_t *src = (const uint32_t *) &T[3 * e];
  const uint32_t *srz = (const uint32_t *) &T[3 * e + 2];
  size_t o = (size_t) cv * nout + r0 + k;
  uint32_t *dx = (uint32_t *) &rx[o], *dz = (uint32_t *) &rz[o];
  for (uint32_t i = threadIdx.x; i < BITS / 32; i += blockDim.x)
    {
      dx[i] = src[i];
      dz[i] = srz[i];
    }
}

/* x[k] = X[k] / Z[k] (and y[k] = Y[k] / Z[k] if Y), canonical, not
   Montgomery; 0 (and 0) where Z[k] is 0 exactly (the neutral element).
   count values per curve, chunks of C2 within a curve, one inversion
   each; bad[curve] = 1 if an inversion fails. */
template <uint32_t TPI, uint32_t BITS>
__global__ void
k_affine (const dcurve<BITS> *c, uint32_t C, uint32_t count, uint32_t C2,
          const cgbn_mem_t<BITS> *X, const cgbn_mem_t<BITS> *Y,
          const cgbn_mem_t<BITS> *Z, size_t stride, cgbn_mem_t<BITS> *pre,
          cgbn_mem_t<BITS> *x, cgbn_mem_t<BITS> *y, uint8_t *bad)
{
  ENV
  uint32_t per = (count + C2 - 1) / C2;
  if (inst >= C * per)
    return;
  uint32_t cv = inst / per, ch = inst % per;
  size_t k0 = (size_t) cv * count + (size_t) ch * C2;
  size_t k1 = k0 + C2;
  if (k1 > (size_t) (cv + 1) * count)
    k1 = (size_t) (cv + 1) * count;
  ar<env_t> m (env, &c->n);
  bn p, z, inv, t;
  m.one (p);
  for (size_t k = k0; k < k1; k++)
    {
      cgbn_store (env, &pre[k], p); /* product of the Z before k */
      cgbn_load (env, z, (cgbn_mem_t<BITS> *) &Z[k * stride]);
      if (!cgbn_equals_ui32 (env, z, 0))
        m.mul (p, p, z);
    }
  cgbn_mont2bn (env, t, p, m.n, m.np0);
  if (!cgbn_modular_inverse (env, inv, t, m.n))
    {
      if ((threadIdx.x % TPI) == 0)
        bad[cv] = 1;
      return;
    }
  cgbn_bn2mont (env, inv, inv, m.n); /* inv = 1 / (Z_k0 ... Z_{k1-1}) */
  for (size_t k = k1; k-- > k0;)
    {
      cgbn_load (env, z, (cgbn_mem_t<BITS> *) &Z[k * stride]);
      if (cgbn_equals_ui32 (env, z, 0))
        {
          cgbn_set_ui32 (env, t, 0);
          cgbn_store (env, &x[k], t);
          if (Y)
            cgbn_store (env, &y[k], t);
          continue;
        }
      bn w, v;
      cgbn_load (env, w, &pre[k]);
      m.mul (w, w, inv); /* 1 / Z_k */
      m.mul (inv, inv, z);
      cgbn_load (env, v, (cgbn_mem_t<BITS> *) &X[k * stride]);
      m.mul (v, v, w);
      cgbn_mont2bn (env, v, v, m.n, m.np0);
      cgbn_store (env, &x[k], v);
      if (Y)
        {
          cgbn_load (env, v, (cgbn_mem_t<BITS> *) &Y[k * stride]);
          m.mul (v, v, w);
          cgbn_mont2bn (env, v, v, m.n, m.np0);
          cgbn_store (env, &y[k], v);
        }
    }
}

/* part[c per + k] = product of acc[c cnt + k C2 ..] (not Montgomery);
   acc reset to 1. */
template <uint32_t TPI, uint32_t BITS>
__global__ void
k_accprod (const dcurve<BITS> *c, uint32_t C, uint32_t cnt, uint32_t C2,
           cgbn_mem_t<BITS> *acc, cgbn_mem_t<BITS> *part)
{
  ENV
  uint32_t per = (cnt + C2 - 1) / C2;
  if (inst >= C * per)
    return;
  uint32_t cv = inst / per, ch = inst % per;
  size_t k0 = (size_t) cv * cnt + (size_t) ch * C2;
  size_t k1 = k0 + C2;
  if (k1 > (size_t) (cv + 1) * cnt)
    k1 = (size_t) (cv + 1) * cnt;
  ar<env_t> m (env, &c->n);
  bn p, v, one;
  m.one (p);
  m.one (one);
  for (size_t k = k0; k < k1; k++)
    {
      cgbn_load (env, v, &acc[k]);
      m.mul (p, p, v);
      cgbn_store (env, &acc[k], one);
    }
  cgbn_mont2bn (env, p, p, m.n, m.np0);
  cgbn_store (env, &part[inst], p);
}

/* acc[k] = 1 (Montgomery). */
template <uint32_t TPI, uint32_t BITS>
__global__ void
k_accinit (const dcurve<BITS> *c, size_t count, cgbn_mem_t<BITS> *acc)
{
  ENV
  if (inst >= count)
    return;
  ar<env_t> m (env, &c->n);
  bn one;
  m.one (one);
  cgbn_store (env, &acc[inst], one);
}

/* ------------------------------------------------------------------ */
/* Host side                                                            */

template <uint32_t BITS>
void
to_mem (cgbn_mem_t<BITS> &d, const mpz_t v)
{
  memset (&d, 0, sizeof d);
  mpz_export (d._limbs, NULL, -1, 4, 0, 0, v);
}

template <uint32_t BITS>
void
from_mem (mpz_t v, const cgbn_mem_t<BITS> &d)
{
  mpz_import (v, BITS / 32, -1, 4, 0, 0, d._limbs);
}

struct base_impl
{
  virtual ~base_impl () {}
  virtual void run (const int *sched, unsigned long nsched,
                    unsigned long nout, uint8_t *bad) = 0;
  virtual const void *roots () = 0;
  virtual void download (mpz_t *out, unsigned long nout) = 0;
  virtual int table (mpz_t *x, mpz_t *y, mpz_t f) = 0;
  virtual void restore () = 0;
  unsigned bits;
};

template <uint32_t TPI, uint32_t BITS>
struct impl : base_impl
{
  typedef cgbn_mem_t<BITS> mem;
  mpz_t n;
  unsigned C, nr, S, cnt; /* cnt = nr * (S + 1) points per curve */
  dcurve<BITS> *dc = NULL;
  mem *ca = NULL, *cb3 = NULL;
  mem *cur = NULL, *nxt = NULL, *snap = NULL, *acc = NULL, *part = NULL;
  mem *rx = NULL; /* the last run's roots: C nout affine canonical */
  size_t rcap = 0;
  cudaStream_t s = cudaStreamPerThread;

  static unsigned blocks (size_t inst)
  {
    return (unsigned) ((inst * TPI + TPB - 1) / TPB);
  }

  impl () { mpz_init (n); bits = BITS; }
  ~impl ()
  {
    cudaFree (dc);
    cudaFree (ca);
    cudaFree (cb3);
    cudaFree (cur);
    cudaFree (nxt);
    cudaFree (snap);
    cudaFree (acc);
    cudaFree (part);
    cudaFree (rx);
    mpz_clear (n);
  }

  /* per curve: product of the accumulators invertible mod n? bad[c] |= */
  void acc_check (uint8_t *bad)
  {
    const unsigned C2 = 64, per = (cnt + C2 - 1) / C2;
    k_accprod<TPI, BITS><<<blocks ((size_t) C * per), TPB, 0, s>>> (dc, C,
        cnt, C2, acc, part);
    std::vector<mem> h ((size_t) C * per);
    ck (cudaMemcpyAsync (h.data (), part, h.size () * sizeof (mem),
                         cudaMemcpyDeviceToHost, s), "accumulators");
    ck (cudaStreamSynchronize (s), "accumulators");
    mpz_t p, v;
    mpz_init (p);
    mpz_init (v);
    for (unsigned cv = 0; cv < C; cv++)
      {
        mpz_set_ui (p, 1);
        for (unsigned k = 0; k < per; k++)
          {
            from_mem<BITS> (v, h[(size_t) cv * per + k]);
            mpz_mul (p, p, v);
            mpz_mod (p, p, n);
          }
        mpz_gcd (v, p, n);
        if (mpz_cmp_ui (v, 1) != 0)
          bad[cv] = 1;
      }
    mpz_clear (p);
    mpz_clear (v);
  }

  bool init (const mpz_t N, unsigned C_, const mpz_t *A, const mpz_t *px,
             const mpz_t *py, mpz_t *coeffs, unsigned nr_, unsigned S_,
             uint8_t *bad)
  {
    mpz_set (n, N);
    C = C_;
    nr = nr_;
    S = S_;
    cnt = nr * (S + 1);
    size_t tot = (size_t) C * cnt;
    /* per curve: b3 = 3 (py^2 - px^3 - A px) mod n; a, b3 in Montgomery
       form (R = 2^BITS) */
    std::vector<mem> ha (C), hb (C), hx (C), hy (C);
    mpz_t b, t, R;
    mpz_inits (b, t, R, NULL);
    mpz_setbit (R, BITS);
    for (unsigned cv = 0; cv < C; cv++)
      {
        mpz_mul (b, py[cv], py[cv]);
        mpz_powm_ui (t, px[cv], 3, n);
        mpz_sub (b, b, t);
        mpz_mul (t, A[cv], px[cv]);
        mpz_sub (b, b, t);
        mpz_mul_ui (b, b, 3);
        mpz_mul (b, b, R);
        mpz_mod (b, b, n);
        to_mem<BITS> (hb[cv], b);
        mpz_mul (t, A[cv], R);
        mpz_mod (t, t, n);
        to_mem<BITS> (ha[cv], t);
        to_mem<BITS> (hx[cv], px[cv]);
        to_mem<BITS> (hy[cv], py[cv]);
      }
    mpz_clears (b, t, R, NULL);
    dcurve<BITS> hc;
    to_mem<BITS> (hc.n, n);
    ck (cudaMalloc (&dc, sizeof hc), "curve");
    ck (cudaMemcpy (dc, &hc, sizeof hc, cudaMemcpyHostToDevice), "curve");
    ck (cudaMalloc (&ca, C * sizeof (mem)), "curves");
    ck (cudaMalloc (&cb3, C * sizeof (mem)), "curves");
    ck (cudaMemcpy (ca, ha.data (), C * sizeof (mem),
                    cudaMemcpyHostToDevice), "curves");
    ck (cudaMemcpy (cb3, hb.data (), C * sizeof (mem),
                    cudaMemcpyHostToDevice), "curves");
    ck (cudaMalloc (&cur, 3 * tot * sizeof (mem)), "table");
    ck (cudaMalloc (&nxt, 3 * tot * sizeof (mem)), "table");
    ck (cudaMalloc (&acc, tot * sizeof (mem)), "accumulators");
    ck (cudaMalloc (&part, ((size_t) C * (cnt / 64 + 2)) * sizeof (mem)),
        "acc");
    k_accinit<TPI, BITS><<<blocks (tot), TPB, 0, s>>> (dc, tot, acc);

    /* the multipliers (the same for all curves): |q_i| as w words, signs */
    size_t maxbit = 0;
    for (unsigned i = 0; i < cnt; i++)
      {
        size_t b2 = mpz_sgn (coeffs[i]) ? mpz_sizeinbase (coeffs[i], 2) : 0;
        if (b2 > maxbit)
          maxbit = b2;
      }
    if (maxbit == 0)
      maxbit = 1;
    uint32_t w = (uint32_t) ((maxbit + 31) / 32);
    std::vector<uint32_t> hq ((size_t) cnt * w, 0);
    std::vector<uint8_t> hn (cnt, 0);
    for (unsigned i = 0; i < cnt; i++)
      {
        hn[i] = mpz_sgn (coeffs[i]) < 0;
        mpz_export (hq.data () + (size_t) i * w, NULL, -1, 4, 0, 0,
                    coeffs[i]);
      }
    uint32_t *dq;
    uint8_t *dn;
    mem *pw, *dpx, *dpy, *pacc;
    ck (cudaMalloc (&dq, hq.size () * 4), "multipliers");
    ck (cudaMalloc (&dn, cnt), "signs");
    ck (cudaMalloc (&pw, (size_t) C * 3 * maxbit * sizeof (mem)), "powers");
    ck (cudaMalloc (&dpx, C * sizeof (mem)), "P");
    ck (cudaMalloc (&dpy, C * sizeof (mem)), "P");
    ck (cudaMalloc (&pacc, C * sizeof (mem)), "P");
    ck (cudaMemcpy (dq, hq.data (), hq.size () * 4, cudaMemcpyHostToDevice),
        "multipliers");
    ck (cudaMemcpy (dn, hn.data (), cnt, cudaMemcpyHostToDevice), "signs");
    ck (cudaMemcpy (dpx, hx.data (), C * sizeof (mem),
                    cudaMemcpyHostToDevice), "P");
    ck (cudaMemcpy (dpy, hy.data (), C * sizeof (mem),
                    cudaMemcpyHostToDevice), "P");
    uint32_t mb = (uint32_t) (maxbit - 1); /* highest bit index */
    k_powers<TPI, BITS><<<blocks (C), TPB, 0, s>>> (dc, C, ca, cb3, dpx, dpy,
                                                     mb, pw, pacc);
    ck (cudaGetLastError (), "powers");
    k_mulw<TPI, BITS><<<blocks (tot), TPB, 0, s>>> (dc, C, cnt, ca, cb3, dq,
        w, dn, mb, pw, cur, acc);
    ck (cudaGetLastError (), "multiples");
    std::vector<mem> hp (C);
    ck (cudaMemcpyAsync (hp.data (), pacc, C * sizeof (mem),
                         cudaMemcpyDeviceToHost, s), "powers acc");
    ck (cudaStreamSynchronize (s), "init");
    cudaFree (dq);
    cudaFree (dn);
    cudaFree (pw);
    cudaFree (dpx);
    cudaFree (dpy);
    cudaFree (pacc);
    mpz_t v;
    mpz_init (v);
    for (unsigned cv = 0; cv < C; cv++)
      {
        from_mem<BITS> (v, hp[cv]);
        mpz_gcd (v, v, n);
        if (mpz_cmp_ui (v, 1) != 0)
          bad[cv] = 1;
      }
    mpz_clear (v);
    acc_check (bad);
    if (nr > 1)
      k_copytop<BITS><<<C * (nr - 1), 32, 0, s>>> (cur, C, nr, S);
    ck (cudaStreamSynchronize (s), "init");
    return true;
  }

  void run (const int *sched, unsigned long nsched, unsigned long nout,
            uint8_t *bad)
  {
    size_t tot = (size_t) C * cnt;
    if (C == 1)
      {
        if (!snap)
          ck (cudaMalloc (&snap, 3 * tot * sizeof (mem)), "table");
        ck (cudaMemcpyAsync (snap, cur, 3 * tot * sizeof (mem),
                             cudaMemcpyDeviceToDevice, s), "snapshot");
      }
    std::vector<int> plist;
    plist.reserve (nout);
    for (unsigned long k = 0; k < nsched; k++)
      if (sched[k] >= 0)
        plist.push_back (sched[k]);
    if (plist.size () != nout)
      abort ();
    size_t nb = (size_t) C * (nout ? nout : 1);
    if (nb > rcap)
      {
        cudaFree (rx);
        ck (cudaMalloc (&rx, nb * sizeof (mem)), "roots");
        rcap = nb;
      }
    int *dp = NULL;
    mem *rz = NULL, *pre = NULL, *raw = NULL;
    uint8_t *dbad = NULL;
    ck (cudaMalloc (&dp, (nout ? nout : 1) * sizeof (int)), "plist");
    ck (cudaMalloc (&raw, nb * sizeof (mem)), "roots");
    ck (cudaMalloc (&rz, nb * sizeof (mem)), "roots");
    ck (cudaMalloc (&pre, nb * sizeof (mem)), "roots");
    ck (cudaMalloc (&dbad, C), "flags");
    ck (cudaMemsetAsync (dbad, 0, C, s), "flags");
    if (nout)
      ck (cudaMemcpyAsync (dp, plist.data (), nout * sizeof (int),
                           cudaMemcpyHostToDevice, s), "plist");
    unsigned long r0 = 0, r = 0;
    for (unsigned long k = 0; k <= nsched; k++)
      {
        if (k == nsched || sched[k] < 0)
          {
            if (r > r0)
              k_extract<BITS><<<(unsigned) (C * (r - r0)), 32, 0, s>>> (cur,
                  C, cnt, S, dp + r0, (uint32_t) (r - r0), nout, r0, raw, rz);
            r0 = r;
            if (k < nsched)
              {
                k_update<TPI, BITS><<<blocks (tot), TPB, 0, s>>> (dc,
                    (uint32_t) tot, cnt, S, ca, cb3, cur, nxt, acc);
                mem *tmp = cur;
                cur = nxt;
                nxt = tmp;
              }
          }
        else
          r++;
      }
    ck (cudaGetLastError (), "updates");
    acc_check (bad);
    if (nout)
      {
        const uint32_t C2 = 32;
        uint32_t per = (uint32_t) ((nout + C2 - 1) / C2);
        k_affine<TPI, BITS><<<blocks ((size_t) C * per), TPB, 0, s>>> (dc, C,
            (uint32_t) nout, C2, raw, NULL, rz, 1, pre, rx, NULL, dbad);
        ck (cudaGetLastError (), "affine");
      }
    std::vector<uint8_t> hb (C);
    ck (cudaMemcpyAsync (hb.data (), dbad, C, cudaMemcpyDeviceToHost, s),
        "flags");
    ck (cudaStreamSynchronize (s), "run");
    for (unsigned cv = 0; cv < C; cv++)
      bad[cv] |= hb[cv];
    cudaFree (dp);
    cudaFree (raw);
    cudaFree (rz);
    cudaFree (pre);
    cudaFree (dbad);
  }

  const void *roots () { return rx; }

  void download (mpz_t *out, unsigned long nout)
  {
    std::vector<mem> h ((size_t) C * nout);
    ck (cudaMemcpy (h.data (), rx, h.size () * sizeof (mem),
                    cudaMemcpyDeviceToHost), "roots");
    for (size_t k = 0; k < h.size (); k++)
      from_mem<BITS> (out[k], h[k]);
  }

  void restore ()
  {
    ck (cudaMemcpy (cur, snap, 3 * (size_t) cnt * sizeof (mem),
                    cudaMemcpyDeviceToDevice), "restore");
  }

  int table (mpz_t *x, mpz_t *y, mpz_t f) /* C == 1 */
  {
    mem *pre, *dx, *dy;
    uint8_t *dbad, hbad = 0;
    ck (cudaMalloc (&pre, cnt * sizeof (mem)), "table");
    ck (cudaMalloc (&dx, cnt * sizeof (mem)), "table");
    ck (cudaMalloc (&dy, cnt * sizeof (mem)), "table");
    ck (cudaMalloc (&dbad, 1), "flag");
    ck (cudaMemsetAsync (dbad, 0, 1, s), "flag");
    const uint32_t C2 = 32;
    uint32_t per = (cnt + C2 - 1) / C2;
    k_affine<TPI, BITS><<<blocks (per), TPB, 0, s>>> (dc, 1, cnt, C2, cur,
        cur + 1, cur + 2, 3, pre, dx, dy, dbad);
    std::vector<mem> hx (cnt), hy (cnt);
    ck (cudaMemcpyAsync (&hbad, dbad, 1, cudaMemcpyDeviceToHost, s), "flag");
    ck (cudaMemcpyAsync (hx.data (), dx, cnt * sizeof (mem),
                         cudaMemcpyDeviceToHost, s), "table");
    ck (cudaMemcpyAsync (hy.data (), dy, cnt * sizeof (mem),
                         cudaMemcpyDeviceToHost, s), "table");
    ck (cudaStreamSynchronize (s), "table");
    cudaFree (pre);
    cudaFree (dx);
    cudaFree (dy);
    cudaFree (dbad);
    if (hbad)
      {
        /* a Z not invertible (Montgomery form: same gcd with n) */
        std::vector<mem> hp (3 * (size_t) cnt);
        ck (cudaMemcpy (hp.data (), cur, hp.size () * sizeof (mem),
                        cudaMemcpyDeviceToHost), "table");
        for (unsigned k = 0; k < cnt; k++)
          {
            from_mem<BITS> (f, hp[3 * k + 2]);
            if (mpz_sgn (f) == 0)
              continue;
            mpz_gcd (f, f, n);
            if (mpz_cmp_ui (f, 1) != 0)
              return 1;
          }
        mpz_set (f, n);
        return 1;
      }
    for (unsigned k = 0; k < cnt; k++)
      {
        from_mem<BITS> (x[k], hx[k]);
        from_mem<BITS> (y[k], hy[k]);
      }
    return 0;
  }
};

/* The widths: the narrowest that holds n (n may fill it). */
#define GPU_ROOTS_WIDTHS(X)                                                 \
  X (4, 512) X (8, 1024) X (8, 1536) X (8, 2048) X (16, 3072) X (16, 4096)

base_impl *
make (const mpz_t N, unsigned C, const mpz_t *A, const mpz_t *px,
      const mpz_t *py, mpz_t *coeffs, unsigned nr, unsigned S, uint8_t *bad)
{
  size_t bits = mpz_sizeinbase (N, 2);
#define X(T, B)                                                             \
  if (bits <= B)                                                            \
    {                                                                       \
      impl<T, B> *p = new impl<T, B> ();                                    \
      p->init (N, C, A, px, py, coeffs, nr, S, bad);                        \
      return p;                                                             \
    }
  GPU_ROOTS_WIDTHS (X)
#undef X
  return NULL;
}

int g_active = -1;

} /* namespace */

struct gpu_roots
{
  base_impl *p;
  unsigned C;
};

extern "C" {

int
gpu_roots_active (const mpz_t n)
{
  if (g_active < 0)
    {
      const char *e = getenv ("ECM_GPU_ROOTS");
      int d = 0;
      g_active = !(e && strcmp (e, "0") == 0) &&
                 cudaGetDeviceCount (&d) == cudaSuccess && d > 0;
    }
  return g_active && mpz_sizeinbase (n, 2) <= GPU_ROOTS_MAX_BITS &&
         mpz_odd_p (n);
}

gpu_roots_t *
gpu_roots_init (const mpz_t n, const mpz_t A, const mpz_t px, const mpz_t py,
                mpz_t *coeffs, unsigned nr, unsigned S, int *bad)
{
  uint8_t b = 0;
  *bad = 0;
  base_impl *p = make (n, 1, (const mpz_t *) A, (const mpz_t *) px,
                       (const mpz_t *) py, coeffs, nr, S, &b);
  if (!p)
    return NULL;
  if (b)
    {
      delete p;
      *bad = 1;
      return NULL;
    }
  gpu_roots_t *g = (gpu_roots_t *) malloc (sizeof (gpu_roots_t));
  g->p = p;
  g->C = 1;
  return g;
}

int
gpu_roots_run (gpu_roots_t *g, const int *sched, unsigned long nsched,
               mpz_t *out, unsigned long nout)
{
  uint8_t b = 0;
  g->p->run (sched, nsched, nout, &b);
  if (b)
    {
      g->p->restore ();
      return 1;
    }
  if (nout)
    g->p->download (out, nout);
  return 0;
}

int
gpu_roots_table (gpu_roots_t *g, mpz_t *x, mpz_t *y, mpz_t f)
{
  return g->p->table (x, y, f);
}

void
gpu_roots_free (gpu_roots_t *g)
{
  if (g)
    {
      delete g->p;
      free (g);
    }
}

gpu_roots_t *
gpu_roots_batch_init (const mpz_t n, unsigned C, const mpz_t *A,
                      const mpz_t *px, const mpz_t *py, mpz_t *coeffs,
                      unsigned nr, unsigned S, unsigned char *bad)
{
  base_impl *p = make (n, C, A, px, py, coeffs, nr, S, bad);
  if (!p)
    return NULL;
  gpu_roots_t *g = (gpu_roots_t *) malloc (sizeof (gpu_roots_t));
  g->p = p;
  g->C = C;
  return g;
}

const void *
gpu_roots_batch_run (gpu_roots_t *g, const int *sched, unsigned long nsched,
                     unsigned long nout, unsigned char *bad)
{
  g->p->run (sched, nsched, nout, bad);
  return g->p->roots ();
}

unsigned
gpu_roots_bits (gpu_roots_t *g)
{
  return g->p->bits;
}

} /* extern "C" */
