#pragma once

#include "constants.h"
#include "types.h"

template <typename DT, typename LT>
void load_data(char const* fname, DT* array, LT length);

template <typename DT, typename LT>
void copy_data(DT* copy_from, DT* copy_to, LT length);

template <typename DT, typename LT>
void print_data(DT* input, LT length);

// template <typename IT>
// FDATA_T** malloc_2d_array(IT row, IT col);

// template <typename DT, typename IT>
// void free_2d_array(DT** arr, IT row, IT col);

FDATA_T** malloc_2d_array(IDATA_T row, IDATA_T col);

void free_2d_array(FDATA_T** arr, IDATA_T row, IDATA_T col);

template <typename DT, typename IT>
void transpose(DT* src, DT* dst, const IT row, const IT col);

// given a sequence with a batch size, print the first batch
void print_sequence(IDATA_T* sequence);

void verify_correctness(IDATA_T program_result[COMPUTE_TIME * BATCH_SIZE],
                        const char* real_result_file_name);
