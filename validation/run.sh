#!/bin/bash
# Phase 4 validation of ecmcgbn (installed RPM) against stock gmp-ecm (Fedora)
export PATH=/usr/bin:/bin
G=${G:-/usr/bin/ecmcgbn}; C=/usr/bin/gmp-ecm
fail=0
note() { echo "$@"; }
bad() { echo "FAIL: $@"; fail=$((fail+1)); }

echo "### A. factor agreement: GPU (batched / per-curve) vs stock CPU gmp-ecm"
# -gpu (as upstream) skips stage 2 of every curve when one finds a factor in
# stage 1: such a sigma range only tests stage 1. Take the first range
# (bases 1000, 3000, ...) whose factors are all found in stage 2.
for nb in ${A_SIZES:-256 512 1024 1536 2048 3072 4096}; do
  read p N < <(python3 mknum.py $nb 68 $nb)
  # stock with the NTT stage 2 above 1920 bits too, as ecmcgbn's default
  # (patch 0012): the stage 2 parameters (B2, dF, d) are then the same
  CO=""; [ $nb -gt 1920 ] && CO="-ntt"
  for base in 1000 3000 5000 7000 9000; do
    for mode in batch percurve; do
      env=""; [ $mode = percurve ] && env="ECM_GPU_BATCH=0"
      echo $N | env $env $G -gpu -gpucurves 512 -sigma 3:$base 5e4 > A_${nb}_$mode.log 2>&1
      grep -oE "found in Step [12] with curve [0-9]+" A_${nb}_$mode.log | sort -t' ' -k6n > A_${nb}_$mode.hits
    done
    grep -q "Step 1" A_${nb}_batch.hits || break
  done
  if ! cmp -s A_${nb}_batch.hits A_${nb}_percurve.hits; then bad "$nb bits: batched and per-curve GPU disagree"; fi
  s1=0; grep -q "Step 1" A_${nb}_batch.hits && s1=1
  nh=$(wc -l < A_${nb}_batch.hits)
  ok=0; checked=0
  while read -r _ _ _ st _ _ cv; do
    s=$((base+cv)); out=$(echo $N | $C $CO -param 3 -sigma 3:$s 5e4 2>&1)
    checked=$((checked+1))
    if echo "$out" | grep -q "Factor found in step $st: $p"; then ok=$((ok+1)); else bad "$nb bits sigma 3:$s: GPU found $p in step $st, stock: $(echo "$out" | grep -i 'factor found' | head -1)"; fi
  done < A_${nb}_batch.hits
  neg=0
  for cv in $(seq 1 41 500); do
    grep -qE "curve $cv\$" A_${nb}_batch.hits && continue
    s=$((base+cv)); out=$(echo $N | $C $CO -param 3 -sigma 3:$s 5e4 2>&1)
    pat="factor found"; [ $s1 = 1 ] && pat="factor found in step 1"
    if echo "$out" | grep -qi "$pat"; then bad "$nb bits sigma 3:$s: stock finds a factor ($pat), GPU did not"; fi
    neg=$((neg+1))
  done
  note "$nb bits, sigma 3:$((base+1))..: $nh of 512 curves found the 68-bit factor$([ $s1 = 1 ] && echo ' (a stage-1 hit: stage 2 not run)'), GPU batched == per-curve, stock agrees on $ok/$checked, $neg non-finding curves confirmed"
done

echo "### B. GPU stage 2 products compared with the CPU's (ECM_GPU_BATCH_CHECK=1)"
for nb in 256 1024 2048 3072 4096; do
  read p N < <(python3 mknum.py $nb $((nb/2)) $((nb+1)))
  for B1 in 1e6 3e6; do
    echo $N | ECM_GPU_BATCH_CHECK=1 $G -v -gpu -gpucurves 32 -sigma 3:77 $B1 > B_${nb}_$B1.log 2>&1
    rc=$?; dF=$(grep -oE "dF=[0-9]+" B_${nb}_$B1.log | head -1)
    if [ $rc -gt 1 ] || grep -qiE "differ|mismatch|abort" B_${nb}_$B1.log; then bad "$nb bits B1=$B1: batch check (rc=$rc)"; fi
    note "$nb bits B1=$B1 $dF: rc=$rc (abort on any difference), $(grep -E 'Step 2 took' B_${nb}_$B1.log)"
  done
done

echo "### C. one curve, large dF: every NTT and root compared (ECM_GPU_NTT_CHECK, ECM_GPU_ROOTS_CHECK)"
for nb in 1024 4096; do
  read p N < <(python3 mknum.py $nb $((nb/2)) $((nb+2)))
  echo $N | ECM_GPU_NTT_CHECK=1 ECM_GPU_ROOTS_CHECK=1 $G -v -sigma 3:5 1e4 1e11 > C_$nb.log 2>&1
  rc=$?; [ $rc -gt 1 ] && bad "$nb bits large-dF check rc=$rc"
  note "$nb bits B2=1e11 $(grep -oE 'dF=[0-9]+' C_$nb.log | head -1): rc=$rc, $(grep -E 'Step 2 took' C_$nb.log)"
done

echo "### D. P-1 and P+1 (their stage 2 uses the same GPU NTT)"
read p N < <(python3 mknum.py 1024 512 9)
for m in pm1 pp1; do
  echo $N | ECM_GPU_NTT_CHECK=1 $G -v -$m 1e5 1e10 > D_$m.log 2>&1; rc=$?
  [ $rc -gt 1 ] && bad "$m with NTT check rc=$rc"
  a=$(echo $N | $G -$m -x0 3 1e6 1e11 2>&1 | grep -iE "factor found" ); b=$(echo $N | $C -$m -x0 3 1e6 1e11 2>&1 | grep -iE "factor found")
  [ "$a" = "$b" ] || bad "$m: GPU '$a' vs stock '$b'"
  note "$m: NTT check rc=$rc; B1=1e6 B2=1e11 x0=3 GPU vs stock: same result ('${a:-no factor}')"
done
echo "### failures: $fail"
