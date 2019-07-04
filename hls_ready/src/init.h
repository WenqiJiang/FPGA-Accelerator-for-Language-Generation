#pragma once

#include "types.h"

template <typename DT, typename LT>
void linear_init(DT* input, DT lower_bound, DT upper_bound, LT length);

template <typename DT, typename LT>
void zero_init(DT* input, LT length);

template<typename DT, typename LT>
void init_array(DT* array, DT value, LT length);

void init_float_array(FDATA_T* array, FDATA_T value, LDATA_T length);

void init_int_array(IDATA_T* array, IDATA_T value, LDATA_T length);
