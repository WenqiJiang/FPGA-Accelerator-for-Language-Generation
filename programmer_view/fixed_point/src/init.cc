#include "init.h"

#include "types.h"

template<>
void linear_init(FDATA_T* input, FDATA_T lower_bound,
                 FDATA_T upper_bound, LDATA_T length)
{
    FDATA_T dif = upper_bound - lower_bound;
    LDATA_T pieces = length - 1;
    FDATA_T val_assigned = lower_bound;

    for(LDATA_T idx = 0; idx < length; idx++)
    {
        input[idx] = val_assigned;
        val_assigned += dif / pieces;
    }
}

template<>
void zero_init(FDATA_T* input, LDATA_T length)
{
    for(LDATA_T idx = 0; idx < length; idx++)
        input[idx] = 0;
}

// template<typename DT, typename LT>
// void init_array(DT* array, DT value, LT length) {
  // for (LDATA_T i = 0; i < length; i++) {
    // array[i] = value;
  // }
// }
void init_float_array(FDATA_T* array, FDATA_T value, LDATA_T length) {
  for (LDATA_T i = 0; i < length; i++) {
    array[i] = value;
  }
}
void init_int_array(IDATA_T* array, IDATA_T value, LDATA_T length) {
  for (LDATA_T i = 0; i < length; i++) {
    array[i] = value;
  }
}
