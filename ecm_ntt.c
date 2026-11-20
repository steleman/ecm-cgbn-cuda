/* ecm_ntt.c - high level poly functions to interface between ecm and sp

Copyright 2005, 2006, 2007, 2008, 2009, 2011, 2012 Dave Newman,
Paul Zimmermann, Alexander Kruppa.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sp.h"
#include "ecm-impl.h"
#if defined(WITH_GPU) && !defined(TUNE)
#include "gpu_ntt.h"
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h> /* for unlink */
#endif

#define UNUSED 0

/* memory: 4 * len mpspv coeffs */
void
ntt_mul (mpzv_t r, mpzv_t x, mpzv_t y, spv_size_t len, mpzv_t t,
    int monic, mpzspm_t mpzspm)
{
  mpzspv_t u, v;
	
  if (len < MUL_NTT_THRESHOLD)
    {
      list_mul (r, x, len, y, len, monic, t);
      return;
    }

  u = mpzspv_init (2 * len, mpzspm);
  v = mpzspv_init (2 * len, mpzspm);
  
  mpzspv_from_mpzv (v, 0, y, len, mpzspm);
  mpzspv_from_mpzv (u, 0, x, len, mpzspm);

  mpzspv_mul_ntt(u, 0, u, 0, len, v, 0, len, 2 * len, monic, 
    monic ? 2 * len : 0, mpzspm, 
    NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_FFT2 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
  mpzspv_to_mpzv (u, 0, r, 2 * len - 1 + monic, mpzspm);
  
  mpzspv_clear (u, mpzspm);
  mpzspv_clear (v, mpzspm);
}

static void ntt_PolyFromRoots_cpu (mpzv_t, mpzv_t, spv_size_t, mpzv_t,
    mpzspm_t);
static int ntt_PolyFromRoots_Tree_cpu (mpzv_t, mpzv_t, spv_size_t, mpzv_t,
    int, mpzspm_t, mpzv_t *, FILE *);

#if defined(WITH_GPU) && !defined(TUNE)
/* ECM_GPU_NTT_CHECK=1: the GPU product tree (r, Tree) against the CPU's,
   mod N. a are the roots (saved before the GPU run, as r may be a). */
static void
check_tree (mpzv_t r, mpzv_t a, spv_size_t len, mpzv_t *Tree,
    mpzspm_t mpzspm)
{
  unsigned int lgk = ceil_log2 (len), i;
  spv_size_t k;
  listz_t rr = init_list2 (len, 0), tt = init_list2 (4 * len, 0);
  listz_t *T = NULL;
  mpz_t u, v;

  if (Tree)
    {
      T = (listz_t *) malloc (lgk * sizeof (listz_t));
      for (i = 0; i < lgk; i++)
        T[i] = init_list2 (len, 0);
      ntt_PolyFromRoots_Tree_cpu (rr, a, len, tt, -1, mpzspm, T, NULL);
    }
  else
    ntt_PolyFromRoots_cpu (rr, a, len, tt, mpzspm);
  mpz_init (u);
  mpz_init (v);
  for (k = 0; k < len; k++)
    {
      mpz_mod (u, r[k], mpzspm->modulus);
      mpz_mod (v, rr[k], mpzspm->modulus);
      if (mpz_cmp (u, v))
        {
          fprintf (stderr, "GPU NTT check: product of %lu roots, "
                   "coefficient %lu differs from the CPU\n", len, k);
          abort ();
        }
    }
  if (Tree)
    for (i = 0; i < lgk; i++)
      for (k = 0; k < len; k++)
        {
          mpz_mod (u, Tree[i][k], mpzspm->modulus);
          mpz_mod (v, T[i][k], mpzspm->modulus);
          if (mpz_cmp (u, v))
            {
              fprintf (stderr, "GPU NTT check: product tree level %u, "
                       "coefficient %lu differs from the CPU\n", i, k);
              abort ();
            }
        }
  mpz_clear (u);
  mpz_clear (v);
  clear_list (rr, len);
  clear_list (tt, 4 * len);
  if (Tree)
    {
      for (i = 0; i < lgk; i++)
        clear_list (T[i], len);
      free (T);
    }
}

/* The GPU product tree, checked if ECM_GPU_NTT_CHECK=1. */
static int
gpu_tree (mpzv_t r, mpzv_t a, spv_size_t len, mpzv_t *Tree, mpzspm_t mpzspm)
{
  listz_t sa = NULL;
  int done;

  if (!gpu_ntt_active ())
    return 0;
  if (gpu_ntt_check ())
    {
      sa = init_list2 (len, 0);
      list_set (sa, a, len);
    }
  done = gpu_poly_from_roots (r, a, len, Tree, mpzspm);
  if (sa != NULL)
    {
      if (done)
        check_tree (r, sa, len, Tree, mpzspm);
      clear_list (sa, len);
    }
  return done;
}
#endif

/* memory: 2 * len mpzspv coeffs */
void
ntt_PolyFromRoots (mpzv_t r, mpzv_t a, spv_size_t len, mpzv_t t,
    mpzspm_t mpzspm)
{
#if defined(WITH_GPU) && !defined(TUNE)
  if (gpu_tree (r, a, len, NULL, mpzspm))
    return;
#endif
  ntt_PolyFromRoots_cpu (r, a, len, t, mpzspm);
}

static void
ntt_PolyFromRoots_cpu (mpzv_t r, mpzv_t a, spv_size_t len, mpzv_t t,
    mpzspm_t mpzspm)
{
  mpzspv_t x;
  spv_size_t i, m;
  
  ASSERT (len == ((spv_size_t)1) << ceil_log2 (len));

  if (len <= MUL_NTT_THRESHOLD)
  {
    PolyFromRoots (r, a, len, t, mpzspm->modulus);
    return;
  }
  
  x = mpzspv_init (2 * len, mpzspm);
  
  for (i = 0; i < len; i += MUL_NTT_THRESHOLD)
    {
      PolyFromRoots (r, a + i, MUL_NTT_THRESHOLD, t, mpzspm->modulus);
      mpzspv_from_mpzv (x, 2 * i, r, MUL_NTT_THRESHOLD, mpzspm);
    }
  
  for (m = MUL_NTT_THRESHOLD; m < len; m *= 2)
    {
      for (i = 0; i < 2 * len; i += 4 * m)
        {
	  mpzspv_mul_ntt (x, i, x, i, m, x, i + 2 * m, m, 2 * m, 1, 2 * m, mpzspm,
            NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_FFT2 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
	  
	  if (2 * m < len)
	    mpzspv_normalise (x, i, 2 * m, mpzspm);
	}	  
    }
      
  mpzspv_to_mpzv (x, 0, r, len, mpzspm);

  mpzspv_clear (x, mpzspm);
}

  
/* memory: 2 * len mpzspv coeffs */
int
ntt_PolyFromRoots_Tree (mpzv_t r, mpzv_t a, spv_size_t len, mpzv_t t,
    int dolvl, mpzspm_t mpzspm, mpzv_t *Tree, FILE *TreeFile)
{
#if defined(WITH_GPU) && !defined(TUNE)
  if (dolvl == -1 && TreeFile == NULL && gpu_tree (r, a, len, Tree, mpzspm))
    return 0;
#endif
  return ntt_PolyFromRoots_Tree_cpu (r, a, len, t, dolvl, mpzspm, Tree,
                                     TreeFile);
}

static int
ntt_PolyFromRoots_Tree_cpu (mpzv_t r, mpzv_t a, spv_size_t len, mpzv_t t,
    int dolvl, mpzspm_t mpzspm, mpzv_t *Tree, FILE *TreeFile)
{
  mpzspv_t x;
  spv_size_t i, m, m_max;
  mpzv_t src;
  mpzv_t *dst = Tree + ceil_log2 (len) - 1;

  ASSERT (len == ((spv_size_t)1) << ceil_log2 (len));
  
  x = mpzspv_init (2 * len, mpzspm);
  
  if (dolvl >= 0)
    {
      src = a;
      dst = &r;
    }
  else  
    {
      /* Copy the roots into the destination level of the tree (negating
	 if so desired), set the source to this level (which now contains 
	 the possibly negated roots), and advance the destination level 
	 of the tree to the next level */
      src = *dst;
      /* we consider x + root[i], which means we consider negated roots */
      list_set (*dst--, a, len);
    }
  
  m = (dolvl == -1) ? 1 : 1 << (ceil_log2 (len) - 1 - dolvl);
  m_max = (dolvl == -1) ? len : 2 * m;
  
  for (; m < m_max && m < MUL_NTT_THRESHOLD; m *= 2)
    {
      /* dst = &r anyway for dolvl != -1 */
      if (m == len / 2)
	dst = &r;
      
      if (TreeFile && list_out_raw (TreeFile, src, len) == ECM_ERROR)
        {
          outputf (OUTPUT_ERROR, "Error writing product tree of F\n");
          return ECM_ERROR;
        }

      for (i = 0; i < len; i += 2 * m)
	list_mul (t + i, src + i, m, src + i + m, m, 1, t + len);

      list_mod (*dst, t, len, mpzspm->modulus);
      
      src = *dst--;
    }
  
  for (; m < m_max; m *= 2)
    {
      ASSERT (m > 1); /* This code does not do the sign change. Let's assume
			 MUL_NTT_THRESHOLD is always large enough that the
			 degree 1 product are done in the above loop */
      /* dst = &r anyway for dolvl != -1 */
      if (m == len / 2)
        dst = &r;
      
      for (i = 0; i < 2 * len; i += 4 * m)
        {
 	  if (TreeFile &&
	      list_out_raw (TreeFile, src + i / 2, 2 * m) == ECM_ERROR)
	    return ECM_ERROR;
	  
	  mpzspv_from_mpzv (x, i, src + i / 2, m, mpzspm);
	  mpzspv_from_mpzv (x, i + 2 * m, src + i / 2 + m, m, mpzspm);
          mpzspv_mul_ntt (x, i, x, i, m, x, i + 2 * m, m, 2 * m, 1, 2 * m, mpzspm,
            NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_FFT2 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
          mpzspv_to_mpzv (x, i, *dst + i / 2, 2 * m, mpzspm);

          /* we only do the mod reduction to reduce the file size a bit */
	  if (TreeFile)
	    list_mod (*dst + i / 2, *dst + i / 2, 2 * m, mpzspm->modulus);
	}
    
      src = *dst--;
    }

  mpzspv_clear (x, mpzspm);

  return 0;
}


/* 2 NTTs of size 2 * len
 * 2 NTTs of size len
 *
 * memory: 2 * len mpzspv coeffs */
void
ntt_PrerevertDivision (mpzv_t a, mpzv_t b, mpzv_t invb, mpzspv_t sp_b,
    mpzspv_t sp_invb, spv_size_t len, mpzv_t t, mpzspm_t mpzspm)
{
  mpzspv_t x;
  
  if (len < PREREVERTDIVISION_NTT_THRESHOLD)
    {
      PrerevertDivision (a, b, invb, len, t, mpzspm->modulus);
      return;
    }
  
  x = mpzspv_init (2 * len, mpzspm);

  /* y = TOP (TOP (a) * invb) */
  mpzspv_set_sp (x, 0, 0, len + 1, mpzspm);
  mpzspv_from_mpzv (x, len + 1, a + len, len - 1, mpzspm);
  mpzspv_mul_ntt (x, 0, x, 0, 2 * len, sp_invb, 0, UNUSED, 2 * len, 0, 0, mpzspm,
    NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
  mpzspv_normalise (x, 0, len, mpzspm);
  
  mpzspv_mul_ntt (x, 0, x, 0, len, sp_b, 0, UNUSED, len, 0, 0, mpzspm, 
    NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
  mpzspv_to_mpzv (x, 0, t, len, mpzspm);
  
  mpzspv_clear (x, mpzspm);
 
  list_sub (t, t, a + len, len - 1);
  list_sub (a, a, t, len);
  /* can we avoid this mod without risking overflow later? */
  list_mod (a, a, len, mpzspm->modulus);
}

/* memory: 7/2 * len mpzspv coeffs */
void ntt_PolyInvert (mpzv_t q, mpzv_t b, spv_size_t len, mpzv_t t,
    mpzspm_t mpzspm)
{
  spv_size_t k = POLYINVERT_NTT_THRESHOLD / 2;
  mpzspv_t w, x, y, z;
  
  if (len < POLYINVERT_NTT_THRESHOLD)
    {
      PolyInvert (q, b, len, t, mpzspm->modulus);
      return;
    }

  PolyInvert (q + len - k, b + len - k, k, t, mpzspm->modulus);
  
  w = mpzspv_init (len / 2, mpzspm);
  x = mpzspv_init (len, mpzspm);
  y = mpzspv_init (len, mpzspm);
  z = mpzspv_init (len, mpzspm);
  
  mpzspv_from_mpzv (x, 0, q + len - k - 1, k + 1, mpzspm);
  mpzspv_from_mpzv (y, 0, b, len - 1, mpzspm);
  
  for (; k < len; k *= 2)
    {
      mpzspv_set (w, 0, x, 1, k, mpzspm);
      mpzspv_set (z, 0, y, len - 2 * k, 2 * k - 1, mpzspm);
      mpzspv_mul_ntt (z, 0, z, 0, 2 * k - 1, x, 0, k + 1, 2 * k, 0, 0, mpzspm, 
        NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_FFT2 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
      mpzspv_normalise (z, k, k, mpzspm);
      mpzspv_neg (z, 0, z, k, k, mpzspm);
      
      mpzspv_mul_ntt (x, 0, x, 0, 0, z, 0, k, 2 * k, 0, 0, mpzspm, 
        NTT_MUL_STEP_FFT2 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
      if (2 * k < len)
	mpzspv_normalise (x, k, k, mpzspm);
      mpzspv_set (x, 1, x, k, k, mpzspm); /* legal overlap */
      mpzspv_set (x, k + 1, w, 0, MIN(k, len / 2 - 1), mpzspm);
    }

  mpzspv_to_mpzv (x, 1, q, len - POLYINVERT_NTT_THRESHOLD / 2, mpzspm);
 
#if defined DEBUG
  ntt_mul (t, q, b, len, NULL, 0, mpzspm);
  list_mod (t, t, 2 * len - 1, mpzspm->modulus);
  
  spv_size_t i;
  for (i = len - 1; i < 2 * len - 2; i++)
    if (mpz_cmp_ui (t[i], 0))
      printf ("error in ntt_PolyInvert\n");
  if (mpz_cmp_ui (t[2 * len - 2], 1))
    printf ("error in ntt_PolyInvert-\n");
#endif

  mpzspv_clear (w, mpzspm);
  mpzspv_clear (x, mpzspm);
  mpzspv_clear (y, mpzspm);
  mpzspv_clear (z, mpzspm);
}


static int ntt_polyevalT_cpu (mpzv_t, spv_size_t, mpzv_t *, mpzv_t,
    mpzspv_t, mpzspm_t, char *);

#if defined(WITH_GPU) && !defined(TUNE)
/* ntt_polyevalT with every level on the GPU (no tree files). 1 if done. */
static int
ntt_polyevalT_gpu (mpzv_t b, spv_size_t len, mpzv_t *Tree, mpzv_t T,
                   mpzspv_t sp_invF, mpzspm_t mpzspm)
{
  spv_size_t i;
  mpzspv_t x = mpzspv_init (2 * len, mpzspm);
  mpzspv_t y = mpzspv_init (2 * len, mpzspm);
  int done;

  mpzspv_from_mpzv (x, 0, b, len, mpzspm);
  mpzspv_mul_ntt(x, 0, x, 0, len, sp_invF, 0, UNUSED, 2 * len, 0, 0, mpzspm,
    NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
  mpzspv_normalise (x, len - 1, len, mpzspm);
  mpzspv_set (y, 0, x, len - 1, len, mpzspm); /* y = high (b * invF) */
  mpzspv_reverse (y, 0, len, mpzspm); /* y = rev (high (b * invF)) */
  mpzspv_clear (x, mpzspm);
  done = gpu_polyeval_levels (y, len, Tree, T, mpzspm);
  mpzspv_clear (y, mpzspm);
  if (!done)
    return 0;
  for (i = 0; i < len; i++)
    mpz_mod (T[i], T[i], mpzspm->modulus);
  list_swap (b, T, len);
  return 1;
}
#endif

/* memory: 4 * len mpzspv coeffs */
int
ntt_polyevalT (mpzv_t b, spv_size_t len, mpzv_t *Tree, mpzv_t T,
                   mpzspv_t sp_invF, mpzspm_t mpzspm, char *TreeFilenameStem)
{
#if defined(WITH_GPU) && !defined(TUNE)
  if (TreeFilenameStem == NULL && gpu_ntt_active ())
    {
      unsigned int lgk = ceil_log2 (len), l;
      spv_size_t k;
      listz_t cb = NULL, cT = NULL, *cTree = NULL;
      int done;

      if (gpu_ntt_check ())
        {
          /* copies for the CPU code, which reverses the tree's levels */
          cb = init_list2 (len, 0);
          list_set (cb, b, len);
          cT = init_list2 (4 * len, 0);
          cTree = (listz_t *) malloc (lgk * sizeof (listz_t));
          for (l = 0; l < lgk; l++)
            {
              cTree[l] = init_list2 (len, 0);
              list_set (cTree[l], Tree[l], len);
            }
        }
      done = ntt_polyevalT_gpu (b, len, Tree, T, sp_invF, mpzspm);
      if (cb != NULL)
        {
          if (done)
            {
              mpz_t u;
              ntt_polyevalT_cpu (cb, len, cTree, cT, sp_invF, mpzspm, NULL);
              mpz_init (u);
              for (k = 0; k < len; k++)
                {
                  mpz_mod (u, cb[k], mpzspm->modulus);
                  if (mpz_cmp (u, b[k]))
                    {
                      fprintf (stderr, "GPU NTT check: polyevalT value %lu "
                               "differs from the CPU\n", k);
                      abort ();
                    }
                }
              mpz_clear (u);
            }
          clear_list (cb, len);
          clear_list (cT, 4 * len);
          for (l = 0; l < lgk; l++)
            clear_list (cTree[l], len);
          free (cTree);
        }
      if (done)
        return 0;
    }
#endif
  return ntt_polyevalT_cpu (b, len, Tree, T, sp_invF, mpzspm,
                            TreeFilenameStem);
}

static int
ntt_polyevalT_cpu (mpzv_t b, spv_size_t len, mpzv_t *Tree, mpzv_t T,
                   mpzspv_t sp_invF, mpzspm_t mpzspm, char *TreeFilenameStem)
{
  spv_size_t m, i;
  FILE *TreeFile = NULL;
  /* assume this "small" malloc will not fail in normal usage */
  char *TreeFilename = NULL;
  mpzv_t *Tree_orig = Tree;
  int level = 0; /* = ceil_log2 (len / m) - 1 */
  mpzspv_t x = mpzspv_init (2 * len, mpzspm);
  mpzspv_t y = mpzspv_init (2 * len, mpzspm);

  if (TreeFilenameStem)
    {
      TreeFilename = (char *) malloc (strlen (TreeFilenameStem) + 1 + 2 + 1);
      if (TreeFilename == NULL)
        {
          fprintf (stderr, "Cannot allocate memory in ntt_polyevalT\n");
          exit (1);
        }
    }
  
  mpzspv_from_mpzv (x, 0, b, len, mpzspm);
  mpzspv_mul_ntt(x, 0, x, 0, len, sp_invF, 0, UNUSED, 2 * len, 0, 0, mpzspm,
    NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
  mpzspv_normalise (x, len - 1, len, mpzspm);
  mpzspv_set (y, 0, x, len - 1, len, mpzspm); /* y = high (b * invF) */
  mpzspv_reverse (y, 0, len, mpzspm); /* y = rev (high (b * invF)) */
    
  for (m = len / 2; m >= POLYEVALT_NTT_THRESHOLD; m /= 2)
    {
      if (TreeFilenameStem)
        {
          Tree = &T;
	  
          sprintf (TreeFilename, "%s.%d", TreeFilenameStem, level);
          
	  TreeFile = fopen (TreeFilename, "rb");
          if (TreeFile == NULL)
            {
              outputf (OUTPUT_ERROR,
		  "Error opening file %s for product tree of F\n",
                        TreeFilename);
              mpzspv_clear (x, mpzspm);
	      mpzspv_clear (y, mpzspm);
	      return ECM_ERROR;
            }

	  list_inp_raw (*Tree, TreeFile, len);

	  fclose (TreeFile);
	  unlink (TreeFilename);
	}

      for (i = 0; i < len; i += 2 * m)
        {
	  
	  list_revert (*Tree + i, m);
          mpzspv_set_sp (x, 0, 1, 1, mpzspm);
          mpzspv_from_mpzv (x, 1, *Tree + i, m, mpzspm);
	  /* x contains reversed monic poly */
          mpzspv_mul_ntt (x, 0, x, 0, m + 1, y, i, 2 * m, 2 * m, 0, 0, mpzspm, 
            NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_FFT2 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
          if (m > POLYEVALT_NTT_THRESHOLD)
	    mpzspv_normalise (x, m, m, mpzspm);
	    
	  list_revert (*Tree + i + m, m);
	  mpzspv_set_sp (x, 2 * m, 1, 1, mpzspm);
	  mpzspv_from_mpzv (x, 2 * m + 1, *Tree + i + m, m, mpzspm);
          mpzspv_mul_ntt(x, 2 * m, x, 2 * m, m + 1, y, i, UNUSED, 2 * m, 0, 0, mpzspm, 
            NTT_MUL_STEP_FFT1 + NTT_MUL_STEP_MUL + NTT_MUL_STEP_IFFT);
	  if (m > POLYEVALT_NTT_THRESHOLD)
	    mpzspv_normalise (x, 3 * m, m, mpzspm);
	  
	  mpzspv_set (y, i, x, 3 * m, m, mpzspm);
	  mpzspv_set (y, i + m, x, m, m, mpzspm);
        }
      
      Tree++;
      level++;
    }
    
  mpzspv_clear (x, mpzspm);
  mpzspv_to_mpzv (y, 0, T, len, mpzspm); /* T = rev (high (b * invF)) */
  mpzspv_clear (y, mpzspm);
  for (i = 0; i < len; i++)
    mpz_mod (T[i], T[i], mpzspm->modulus);

  for (; m >= 1; m /= 2)
    {
      if (TreeFilenameStem)
        {
          sprintf (TreeFilename, "%s.%d", TreeFilenameStem, level);

          TreeFile = fopen (TreeFilename, "rb");
          if (TreeFile == NULL)
            {
              outputf (OUTPUT_ERROR,
		  "Error opening file %s for product tree of F\n",
                        TreeFilename);
	      return ECM_ERROR;
            }
	}
      
      TUpTree (T, Tree_orig, len, T + len, level++, 0,
	  mpzspm->modulus, TreeFile);

      if (TreeFilenameStem)
        {
	  fclose (TreeFile);
	  unlink (TreeFilename);
	}
    }
  
  if (TreeFilenameStem)
    free (TreeFilename);
  list_swap (b, T, len);
  return 0;
}
