#include "utils.h"

#include <cstdio>
#include <stdlib.h>

#include "constants.h"

template <>
void load_data(char const* fname, FDATA_T* array, LDATA_T length) {

    FILE *data_file;
    data_file = fopen(fname, "r");

    if (data_file == NULL) {
        printf("ERROR: cannot open file: %s\n", fname);
        exit(1);
    }

    // Read floating point values from file and convert them to FDATA_T.
    float *flt_array = (float*) malloc(length * sizeof(double));

    for(LDATA_T i = 0; i < length; i++)
    {
        LDATA_T r = fscanf(data_file,"%40f", &flt_array[i]);
        (void)r; // suppress warning unused variable

        array[i] = FDATA_T(flt_array[i]);
    }

    free(flt_array);

    fclose(data_file);
}

template <>
void load_data(char const* fname, IDATA_T* array, LDATA_T length) {

    FILE *data_file;
    data_file = fopen(fname, "r");

    if (data_file == NULL) {
        printf("ERROR: cannot open file: %s\n", fname);
        exit(1);
    }

    // Read int values from file and convert them to IDATA_T.
    int *int_array = (int*) malloc(length * sizeof(int));

    for(LDATA_T i = 0; i < length; i++)
    {
        LDATA_T r = fscanf(data_file,"%d", &int_array[i]);
        (void)r; // suppress warning unused variable

        array[i] = int_array[i] < 6144? IDATA_T(int_array[i]):IDATA_T(0);
    }

    free(int_array);

    fclose(data_file);
}

template <>
void copy_data(FDATA_T* copy_from, FDATA_T* copy_to, LDATA_T length) {
    for (LDATA_T i = 0; i < length; i++) {
        copy_to[i] = copy_from[i];
    }
}

template <>
void copy_data(IDATA_T* copy_from, IDATA_T* copy_to, LDATA_T length) {
    for (LDATA_T i = 0; i < length; i++) {
        copy_to[i] = copy_from[i];
    }
}

template <>
void print_data(FDATA_T* input, LDATA_T length) {
    for (LDATA_T i = 0; i < length; i ++) {
        printf("%.10f\n", TOFLOAT(input[i]));
    }
}

template <>
void print_data(IDATA_T* input, LDATA_T length) {
    for (LDATA_T i = 0; i < length; i ++) {
        printf("%d\n", TOINT(input[i]));
    }
}

FDATA_T** malloc_2d_array(IDATA_T row, IDATA_T col) {
    FDATA_T** arr = (FDATA_T**) MALLOC(row * sizeof(FDATA_T*));
    for (int i = 0; i < row; i++) {
        arr[i] = (FDATA_T*) MALLOC(col * sizeof(FDATA_T));
    }

    return arr;
}

void free_2d_array(FDATA_T** arr, IDATA_T row, IDATA_T col) {
    for (int i = 0; i < row; i++) {
        MFREE(arr[i]);
    }
    MFREE(arr);
}

template <>
void transpose(FDATA_T* src, FDATA_T* dst, const LDATA_T ROW, const LDATA_T COL)
{
  // transpose array
  // the source array has shape of (row, col)

  for (IDATA_T row = 0; row < ROW; row++) {
    for (IDATA_T col = 0; col < COL; col++)
      dst[col * ROW + row] = src[row * COL + col];
  }
}

void print_sequence(IDATA_T sequence[BATCH_SIZE * COMPUTE_TIME]) {
  // given a sequence with a batch size, print the first batch

  for (LDATA_T i = 0; i < COMPUTE_TIME; i++) {
    LDATA_T idx = i * BATCH_SIZE;
    printf("%d\t", TOINT(sequence[idx]));
  }
}

void verify_correctness(IDATA_T program_result[COMPUTE_TIME * BATCH_SIZE],
                        const char* real_result_file_name) {
  // given the result computed by the program,
  // and the file name of the true result (of the first sample in the batch),
  // the timesteps for verification,
  // i.e. min(COMPUTE_TIME, RESULT_FILE_COMPUTE_TIME)

  IDATA_T* real_result =
      (IDATA_T*) malloc(sizeof(IDATA_T) * RESULT_FILE_COMPUTE_TIME);
  load_data(real_result_file_name, real_result, RESULT_FILE_COMPUTE_TIME);

  const LDATA_T compare_time = COMPUTE_TIME < RESULT_FILE_COMPUTE_TIME?
                               COMPUTE_TIME : RESULT_FILE_COMPUTE_TIME;

  LDATA_T correct_time= 0;
  for (LDATA_T i = 0; i < compare_time; i++) {
    if (program_result[i * BATCH_SIZE] == real_result[i]) {
      correct_time++;
    }
  }

  if (correct_time == compare_time) {
    printf("INFO: Computation is CORRECT!\n");
  }
  else {
    printf("INFO: Computation has errors, correct rate: %f\n",
           float(correct_time) / float(compare_time));
  }

  free(real_result);
}
