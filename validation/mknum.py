# N = p * q, p of pbits bits, N of nbits bits (q prime), reproducible
import random, sys
def isprime(n):
    if n < 2: return False
    for s in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n % s == 0: return n == s
    d, r = n - 1, 0
    while d % 2 == 0: d //= 2; r += 1
    for a in (2,3,5,7,11,13,17,19,23,29,31,37,41):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1: break
        else: return False
    return True
def prime(bits, rng):
    while True:
        x = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if isprime(x): return x
nbits, pbits, seed = map(int, sys.argv[1:4])
rng = random.Random(seed)
p = prime(pbits, rng); q = prime(nbits - pbits, rng)
print(p, p * q)
