#!/usr/bin/env bash

python golden_model.py --input fp8:e4m3 --input-rounding zero --prod-mant-bits 7 --acc bf16 --acc-rounding q_bf16_rne --scaled-spec bf16 --scale-spec fpe8m0 --scale-exp 2 --M 32 --K 32 --N 32 --tile 16 --header-path include/matmul_data_mx_fp8.h
# python golden_model.py --input fp6:e3m2 --input-rounding zero --prod-mant-bits 7 --acc bf16 --acc-rounding q_bf16_rne --scaled-spec bf16 --scale-spec fpe8m0 --scale-exp 2 --M 32 --K 32 --N 32 --tile 32 --lut-index-bits 4 --header-path include/matmul_data_mx_fp6.h
# python golden_model.py --input fp4:e2m1 --input-rounding zero --prod-mant-bits 7 --acc bf16 --acc-rounding q_bf16_rne --scaled-spec bf16 --scale-spec fpe8m0 --scale-exp 2 --M 32 --K 32 --N 32 --tile 32 --header-path include/matmul_data_mx_fp4.h