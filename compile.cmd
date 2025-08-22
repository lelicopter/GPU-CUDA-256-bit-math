@echo off

del u256_benchmark.exe

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

nvcc -rdc=true -use_fast_math --expt-relaxed-constexpr -Xcompiler "/wd 4819" -Xptxas=-v,-O3 -O3 -arch=sm_75 --extended-lambda -std=c++20 -o u256_benchmark.exe u256_benchmark.cu

del u256_benchmark.exp
del u256_benchmark.lib
