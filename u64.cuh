#ifndef _U64_CUH
#define _U64_CUH

typedef unsigned long long u64;

__forceinline__ __device__ bool u64Add(u64 *output, u64 a, u64 b, u64 carry) {
    u64 result;
    u64 carryOut;
    asm volatile(
        "add.cc.u64 %0, %2, %3;"
        "addc.u64 %0, %0, %4;"
        "addc.u64 %1, 0, 0;"
        : "=l"(result), "=l"(carryOut)
        : "l"(a), "l"(b), "l"(carry)
    );
    *output = result;
    return carryOut != 0;
}

__forceinline__ __device__ bool u64Sub(u64 *output, u64 a, u64 b, u64 borrow) {
    u64 result;
    u64 borrowOut;
    asm volatile(
        "sub.cc.u64 %0, %2, %3;"
        "subc.u64 %0, %0, %4;"
        "subc.u64 %1, 0, 0;"
        : "=l"(result), "=l"(borrowOut)
        : "l"(a), "l"(b), "l"(borrow)
    );
    *output = result;
    return borrowOut != 0;
}

__forceinline__ __device__ u64 u64Mul(u64 *output, u64 a, u64 b, u64 carry) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 c = (lo < carry) ? 1ULL : 0ULL;
    *output = lo - carry;
    return hi - c;
}

#endif //_U64_CUH