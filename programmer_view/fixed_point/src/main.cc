#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "activation.h"
#include "config.h"
#include "constants.h"
#include "fc.h"
#include "init.h"
#include "rnn.h"
#include "softmax.h"
#include "types.h"
#include "utils.h"
#include "wrapper.h"

int main(int argc, char *argv[]) {

  // Declare weights
  // Embedding
  FDATA_T* word_embedding =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * WORD_NUM * WORD_SIZE);

  // RNN
  FDATA_T* rnn_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_STATE_SIZE);
  FDATA_T* rnn_kernel =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_INPUT_SIZE * RNN_STATE_SIZE);
  FDATA_T* rnn_kernel_transpose =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_INPUT_SIZE * RNN_STATE_SIZE);
  FDATA_T* rnn_recurrent_kernel =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_STATE_SIZE * RNN_STATE_SIZE);
  FDATA_T* rnn_recurrent_kernel_transpose =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_STATE_SIZE * RNN_STATE_SIZE);

  FDATA_T* rnn_input_state_cache =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * RNN_INPUT_SIZE);

  // Ping-pong buffers, serves as input and output states alternatively
  FDATA_T* rnn_state0 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * RNN_STATE_SIZE);
  FDATA_T* rnn_state1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * RNN_STATE_SIZE);
  init_float_array (rnn_state0, 0, BATCH_SIZE * RNN_STATE_SIZE);
  init_float_array (rnn_state1, 0, BATCH_SIZE * RNN_STATE_SIZE);
  load_data<FDATA_T, LDATA_T>(INIT_STATES_FILE, rnn_state0,
                              BATCH_SIZE * RNN_STATE_SIZE);

  // FC
  FDATA_T* fc_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE);
  FDATA_T* fc_kernel =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_INPUT_SIZE * FC_OUTPUT_SIZE);
  FDATA_T* fc_kernel_transpose =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_INPUT_SIZE * FC_OUTPUT_SIZE);

  FDATA_T* fc_output_cache =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * FC_OUTPUT_SIZE);

  // result indexes of one single time step and all steps
  IDATA_T* result_idx_one_step0 =
      (IDATA_T*) MALLOC(sizeof(IDATA_T) * BATCH_SIZE);
  IDATA_T* result_idx_one_step1 =
      (IDATA_T*) MALLOC(sizeof(IDATA_T) * BATCH_SIZE);
  IDATA_T* result_idx_all =
      (IDATA_T*) MALLOC(sizeof(IDATA_T) * COMPUTE_TIME * BATCH_SIZE);
  init_int_array(result_idx_one_step0, 0, BATCH_SIZE);
  init_int_array(result_idx_one_step1, 0, BATCH_SIZE);
  init_int_array(result_idx_all, 0, COMPUTE_TIME * BATCH_SIZE);
  load_data<IDATA_T, LDATA_T>(INIT_WORD_IDX_FILE, result_idx_one_step0,
                              BATCH_SIZE);

  // load model in
  load_data<FDATA_T, LDATA_T>(EMBEDDINGS_FILE, word_embedding,
                              WORD_NUM * WORD_SIZE);
  load_data<FDATA_T, LDATA_T>(SIMPLE_RNN_BIAS_FILE, rnn_bias, RNN_STATE_SIZE);
  load_data<FDATA_T, LDATA_T>(SIMPLE_RNN_KERNEL_FILE, rnn_kernel,
                              RNN_INPUT_SIZE * RNN_STATE_SIZE);
  load_data<FDATA_T, LDATA_T>(SIMPLE_RNN_RECURRENT_KERNEL_FILE,
                              rnn_recurrent_kernel,
                              RNN_STATE_SIZE * RNN_STATE_SIZE);
  load_data<FDATA_T, LDATA_T>(DENSE_BIAS_FILE, fc_bias, FC_OUTPUT_SIZE);
  load_data<FDATA_T, LDATA_T>(DENSE_KERNEL_FILE, fc_kernel,
                              FC_INPUT_SIZE * FC_OUTPUT_SIZE);

  // transpose kernels
  transpose(rnn_kernel, rnn_kernel_transpose, RNN_INPUT_SIZE, RNN_STATE_SIZE);
  transpose(rnn_recurrent_kernel, rnn_recurrent_kernel_transpose,
            RNN_STATE_SIZE, RNN_STATE_SIZE);
  transpose(fc_kernel, fc_kernel_transpose, FC_INPUT_SIZE, FC_OUTPUT_SIZE);

  // MFREE untransposed kernel
  MFREE(rnn_kernel);
  MFREE(rnn_recurrent_kernel);
  MFREE(fc_kernel);
#ifdef DEBUG
  print_data<FDATA_T, LDATA_T>(fc_kernel, FC_INPUT_SIZE * FC_OUTPUT_SIZE);
#endif

  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME / 2;
       compute_time++) {

    // Use ping-pong buffer
    LDATA_T result_idx_all_idx = 2 * compute_time * BATCH_SIZE;
    wrapper_rnn_fc(
        word_embedding, rnn_kernel_transpose, rnn_recurrent_kernel_transpose, rnn_bias,
        fc_kernel_transpose, fc_bias, /* input_word_idx = */result_idx_one_step0,
        rnn_input_state_cache, /* rnn_last_state = */rnn_state0,
        /* rnn_output_state = */rnn_state1, fc_output_cache,
        /* result_idx = */result_idx_one_step1);
    memcpy(result_idx_all + result_idx_all_idx, result_idx_one_step1,
           sizeof(FDATA_T) * BATCH_SIZE);

    // result_idx_all_idx = (2 * compute_time + 1) * BATCH_SIZE;
    // wrapper_rnn_fc(
        // word_embedding, rnn_kernel_transpose, rnn_recurrent_kernel_transpose, rnn_bias,
        // fc_kernel_transpose, fc_bias, [> input_word_idx = <]result_idx_one_step1,
        // rnn_input_state_cache, [> rnn_last_state = <]rnn_state1,
        // [> rnn_output_state = <]rnn_state0, fc_output_cache,
        // [> result_idx = <]result_idx_one_step0);
    // memcpy(result_idx_all + result_idx_all_idx, result_idx_one_step0,
           // sizeof(FDATA_T) * BATCH_SIZE);
  }

#define VERBOSE
#ifdef VERBOSE
  print_sequence(result_idx_all);
#endif

  // Embedding
  MFREE(word_embedding);

  // RNN
  MFREE(rnn_bias);
  MFREE(rnn_kernel_transpose);
  MFREE(rnn_recurrent_kernel_transpose);
  MFREE(rnn_input_state_cache);
  MFREE(rnn_state0);
  MFREE(rnn_state1);

  // FC
  MFREE(fc_bias);
  MFREE(fc_kernel_transpose);
  MFREE(fc_output_cache);

  // Indexes
  MFREE(result_idx_one_step0);
  MFREE(result_idx_one_step1);
  MFREE(result_idx_all);

  return 0;
}
