#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "config.h"
#include "constants.h"
#include "init.h"
#include "types.h"
#include "utils.h"
#include "wrapper.h"

int main(int argc, char *argv[]) {


  printf("Starting memory allocation and loading data\n");
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

  // FDATA_T* rnn_input_state_cache =
      // (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * RNN_INPUT_SIZE);

  // Ping-pong buffers, serves as input and output states alternatively
  // FDATA_T* rnn_state0 =
      // (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * RNN_STATE_SIZE);
  // FDATA_T* rnn_state1 =
      // (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * RNN_STATE_SIZE);
  // init_float_array (rnn_state0, 0, BATCH_SIZE * RNN_STATE_SIZE);
  // init_float_array (rnn_state1, 0, BATCH_SIZE * RNN_STATE_SIZE);
  // load_data<FDATA_T, LDATA_T>(INIT_STATES_FILE, rnn_state0,
                              // BATCH_SIZE * RNN_STATE_SIZE);

  FDATA_T* rnn_init_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * RNN_STATE_SIZE);
  load_data<FDATA_T, LDATA_T>(INIT_STATES_FILE, rnn_init_state,
                              BATCH_SIZE * RNN_STATE_SIZE);
  // FC
  FDATA_T* fc_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE);
  FDATA_T* fc_kernel =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_INPUT_SIZE * FC_OUTPUT_SIZE);
  FDATA_T* fc_kernel_transpose =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_INPUT_SIZE * FC_OUTPUT_SIZE);

  // FDATA_T* fc_output_cache =
      // (FDATA_T*) MALLOC(sizeof(FDATA_T) * BATCH_SIZE * FC_OUTPUT_SIZE);

  // // result indexes of one single time step and all steps
  // IDATA_T* result_idx_one_step0 =
      // (IDATA_T*) MALLOC(sizeof(IDATA_T) * BATCH_SIZE);
  // IDATA_T* result_idx_one_step1 =
      // (IDATA_T*) MALLOC(sizeof(IDATA_T) * BATCH_SIZE);
  IDATA_T* result_idx_all =
      (IDATA_T*) MALLOC(sizeof(IDATA_T) * COMPUTE_TIME * BATCH_SIZE);
  // init_int_array(result_idx_one_step0, 0, BATCH_SIZE);
  // init_int_array(result_idx_one_step1, 0, BATCH_SIZE);
  // init_int_array(result_idx_all, 0, COMPUTE_TIME * BATCH_SIZE);
  // load_data<IDATA_T, LDATA_T>(INIT_WORD_IDX_FILE, result_idx_one_step0,
                              // BATCH_SIZE);

  IDATA_T* rnn_init_idx =
      (IDATA_T*) MALLOC(sizeof(IDATA_T) * BATCH_SIZE);
  load_data<IDATA_T, LDATA_T>(INIT_WORD_IDX_FILE, rnn_init_idx, BATCH_SIZE);

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

  printf("Ends memory allocation and loading data\n");
#ifdef DEBUG
  print_data<IDATA_T, LDATA_T>(result_idx_one_step0, BATCH_SIZE);
#endif

  printf("Start Inference\n");

#ifdef __SDSCC__
  perf_counter f_ctr;
#endif

#ifdef PROFILING
  struct timespec start, finish;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

#ifdef __SDSCC__
  f_ctr.start();
#endif

wrapper_text_generation(
     word_embedding, rnn_kernel_transpose, rnn_recurrent_kernel_transpose,
     rnn_bias, fc_kernel_transpose, fc_bias, rnn_init_state, rnn_init_idx,
     result_idx_all);

#ifdef __SDSCC__
  f_ctr.stop();
#endif

#ifdef PROFILING
  clock_gettime(CLOCK_REALTIME, &finish);

  long seconds = finish.tv_sec - start.tv_sec;
  long ns = finish.tv_nsec - start.tv_nsec;

  if (start.tv_nsec > finish.tv_nsec) { // clock underflow
    --seconds;
    ns += 1000000000;
  }

  printf("seconds: %ld\n", seconds);
  printf("nanoseconds: %ld\n", ns);
  printf("total seconds: %e\n", (double)seconds + (double)ns/(double)1000000000);
#endif

#ifdef __SDSCC__
  printf("INFO:   cpu cycles %lu\n\r", f_ctr.avg_cpu_cycles());
  f_ctr.reset();
#endif

#ifdef VERBOSE
  print_sequence(result_idx_all);
#endif

  verify_correctness(result_idx_all, CORRECT_RESULT_FILE);

  // Embedding
  MFREE(word_embedding);

  // RNN
  MFREE(rnn_bias);
  MFREE(rnn_kernel_transpose);
  MFREE(rnn_recurrent_kernel_transpose);
  // MFREE(rnn_input_state_cache);
  MFREE(rnn_init_state);
  // MFREE(rnn_state0);
  // MFREE(rnn_state1);

  // FC
  MFREE(fc_bias);
  MFREE(fc_kernel_transpose);
  // MFREE(fc_output_cache);

  // Indexes
  // MFREE(result_idx_one_step0);
  // MFREE(result_idx_one_step1);
  MFREE(rnn_init_idx);
  MFREE(result_idx_all);

  return 0;
}
