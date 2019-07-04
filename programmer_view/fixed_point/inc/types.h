#pragma once

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

#define INT_W_LENGTH 16
#define FIXED_W_LENGTH 16
#define FIXED_I_LENGTH 7

#define LDATA_T int
#define IDATA_T short
#define FDATA_T ap_fixed<FIXED_W_LENGTH, FIXED_I_LENGTH>

#define TOFLOAT(a) a.to_float()
#define TOINT(a) int(a)

#define MALLOC malloc
#define MFREE free
