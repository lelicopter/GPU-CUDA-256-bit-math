#ifndef _U256_CUH
#define _U256_CUH

#include "u64.cuh"

__forceinline__ __device__ void u256Copy(u64 *r, const u64 *a) {
  r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3];
}

__forceinline__ __device__ void u256SetZero(u64 *a) {
  a[0] = 0; a[1] = 0; a[2] = 0; a[3] = 0;
}

__forceinline__ __device__ bool u256IsZero(const u64 *a) {
  return (a[0] == 0) && (a[1] == 0) && (a[2] == 0) && (a[3] == 0);
}

__forceinline__ __device__ bool u256GetBit(const u64 *a, int index) {
  int word = index >> 6;
  int bit = index & 63;
  return (a[word] >> bit) & 1;
}

__forceinline__ __device__ void u256SetBit(u64 *output, int index, bool value) {
  int word = index >> 6;
  u64 mask = 1ULL << (index & 63);
  if (value) output[word] |= mask;
  else output[word] &= ~mask;
}

__forceinline__ __device__ int u256Compare(const u64 *a, const u64 *b) {
  if (a[3] != b[3]) return a[3] < b[3] ? -1 : 1;
  if (a[2] != b[2]) return a[2] < b[2] ? -1 : 1;
  if (a[1] != b[1]) return a[1] < b[1] ? -1 : 1;
  if (a[0] != b[0]) return a[0] < b[0] ? -1 : 1;
  return 0;
}

__forceinline__ __device__ void u256And(u64 *r, const u64 *a, const u64 *b) {
  asm("and.b64 %0, %1, %2;" : "=l"(r[0]) : "l"(a[0]), "l"(b[0]));
  asm("and.b64 %0, %1, %2;" : "=l"(r[1]) : "l"(a[1]), "l"(b[1]));
  asm("and.b64 %0, %1, %2;" : "=l"(r[2]) : "l"(a[2]), "l"(b[2]));
  asm("and.b64 %0, %1, %2;" : "=l"(r[3]) : "l"(a[3]), "l"(b[3]));
}

__forceinline__ __device__ bool u256Add(u64 *r, const u64 *a, const u64 *b) {
  u64 carry64;
  asm volatile(
    "add.cc.u64 %0, %5, %9;"
    "addc.cc.u64 %1, %6, %10;"
    "addc.cc.u64 %2, %7, %11;"
    "addc.u64 %3, %8, %12;"
    "addc.u64 %4, 0, 0;"
    : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3]), "=l"(carry64)
    : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
      "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
  );
  return carry64 != 0;
}

__forceinline__ __device__ bool u256Sub(u64 *r, const u64 *a, const u64 *b) {
  u64 borrow64;
  asm volatile(
    "sub.cc.u64 %0, %5, %9;"
    "subc.cc.u64 %1, %6, %10;"
    "subc.cc.u64 %2, %7, %11;"
    "subc.u64 %3, %8, %12;"
    "subc.u64 %4, 0, 0;"
    : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3]), "=l"(borrow64)
    : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
      "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
  );
  return borrow64 != 0;
}

__forceinline__ __device__ void u256Mul(u64 *r, const u64 *a, const u64 *b) {
  u64 t[8] = {0};
  for (int i = 0; i < 4; i++) {
    u64 carry = 0;
    for (int j = 0; j < 4; j++) {
      u64 lo, hi;
      asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a[i]), "l"(b[j]));
      asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a[i]), "l"(b[j]));

      u64 carry_inner = carry;
      u64 sum1;
      unsigned int c1 = u64Add(&sum1, t[i+j], lo, 0);
      u64 sum2;
      unsigned int c2 = u64Add(&sum2, sum1, carry_inner, 0);

      t[i+j] = sum2;
      carry = hi + c1 + c2;
    }
    t[i+4] = carry;
  }
  r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
}

__forceinline__ __device__ void u256RShift(u64 *r, const u64 *a, uint64_t shift) {
  if (shift >= 256) {
    u256SetZero(r);
    return;
  }
  
  const uint64_t words = shift / 64;
  const uint64_t bits = shift % 64;
  
  if (bits == 0) {
    for (int i = 0; i < 4 - words; i++) r[i] = a[i + words];
    for (int i = 4 - words; i < 4; i++) r[i] = 0;
  } else {
    const uint64_t inv_bits = 64 - bits;
    for (int i = 0; i < 3; i++) {
      if (i < 3 - words) {
        r[i] = (a[i + words] >> bits) | (a[i + words + 1] << inv_bits);
      }
    }
    if (3 >= words) r[3 - words] = a[3] >> bits;
    for (int i = 0; i < words; i++) r[i] = 0;
  }
}

__forceinline__ __device__ void u256LShift(u64 *r, const u64 *a, uint64_t shift) {
  if (shift >= 256) {
    u256SetZero(r);
    return;
  }
  
  const uint64_t words = shift / 64;
  const uint64_t bits = shift % 64;
  
  if (bits == 0) {
    for (int i = 3; i >= (int)words; i--) r[i] = a[i - words];
    for (int i = 0; i < (int)words; i++) r[i] = 0;
  } else {
    const uint64_t inv_bits = 64 - bits;
    for (int i = 3; i >= 0; i--) {
      if (i > words) {
        r[i] = (a[i - words] << bits) | (a[i - words - 1] >> inv_bits);
      } else if (i == words) {
        r[i] = a[0] << bits;
      } else {
        r[i] = 0;
      }
    }
  }
}

__forceinline__ __device__ void barrett_reduce(u64 *x, const u64 *mu, const u64 *p) {
  u64 q[5] = {0};
  u256Mul(q, &x[4], mu);
  
  u64 t[8] = {0};
  u256Mul(t, q, p);
  
  u64 borrow = 0;
  borrow = u64Sub(&x[0], x[0], t[0], borrow);
  borrow = u64Sub(&x[1], x[1], t[1], borrow);
  borrow = u64Sub(&x[2], x[2], t[2], borrow);
  borrow = u64Sub(&x[3], x[3], t[3], borrow);
  
  if (borrow || u256Compare(x, p) >= 0) {
    u256Sub(x, x, p);
  }
}

__forceinline__ __device__ void u256Div(u64 *output, const u64 *a, const u64 *b) {
  u64 remainder[5] = {0};
  u64 quotient[4] = {0};

  for (int i = 255; i >= 0; i--) {
    u64 carry = (remainder[0] >> 63);
    remainder[0] = (remainder[0] << 1);
    for (int j = 1; j < 5; j++) {
      u64 next_carry = (remainder[j] >> 63);
      remainder[j] = (remainder[j] << 1) | carry;
      carry = next_carry;
    }

    if (u256GetBit(a, i)) {
      remainder[0] |= 1;
    }

    bool cmp = (remainder[4] > 0) || u256Compare(remainder, b) >= 0;

    if (cmp) {
      u64 borrow = 0;
      borrow = u64Sub(&remainder[0], remainder[0], b[0], borrow);
      borrow = u64Sub(&remainder[1], remainder[1], b[1], borrow);
      borrow = u64Sub(&remainder[2], remainder[2], b[2], borrow);
      borrow = u64Sub(&remainder[3], remainder[3], b[3], borrow);
      remainder[4] -= borrow;
      u256SetBit(quotient, i, true);
    }
  }

  u256Copy(output, quotient);
}

#endif
