#include "wrapper.h"

// #include <cstring>
// #include <cstdio>

// #include "activation.h"
#include "constants.h"
// #include "fc.h"
// #include "rnn.h"
// #include "softmax.h"
#include "types.h"

////////////////////         TOP-LEVEL FUNCTION             ////////////////////

// weights
#pragma SDS data copy(word_embedding[0: WORD_NUM * WORD_SIZE])
#pragma SDS data copy(rnn_kernel[0: RNN_INPUT_SIZE * RNN_STATE_SIZE])
#pragma SDS data copy( \
    rnn_recurrent_kernel[0: RNN_STATE_SIZE * RNN_STATE_SIZE])
#pragma SDS data copy(rnn_bias[0: RNN_STATE_SIZE])
#pragma SDS data copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
#pragma SDS data copy(fc_bias[0: FC_OUTPUT_SIZE])

// input states and indexes
#pragma SDS data copy(rnn_init_state[0: BATCH_SIZE * RNN_INPUT_SIZE])
#pragma SDS data copy(rnn_init_idx[0: BATCH_SIZE])

// result indexes
#pragma SDS data copy(result_idx_all[0: COMPUTE_TIME * BATCH_SIZE])

// data access pattern
#pragma SDS data access_pattern( \
  word_embedding: SEQUENTIAL, \
  rnn_kernel: SEQUENTIAL, \
  rnn_recurrent_kernel: SEQUENTIAL, \
  rnn_bias: SEQUENTIAL, \
  fc_kernel: SEQUENTIAL, \
  fc_bias: SEQUENTIAL, \
  rnn_init_state: SEQUENTIAL, \
  rnn_init_idx: SEQUENTIAL, \
  result_idx_all: SEQUENTIAL)

void wrapper_text_generation(
    FDATA_T word_embedding[WORD_NUM * WORD_SIZE],
    FDATA_T rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_bias[RNN_STATE_SIZE],
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    FDATA_T fc_bias[FC_OUTPUT_SIZE],
    FDATA_T rnn_init_state[BATCH_SIZE * RNN_STATE_SIZE],
    IDATA_T rnn_init_idx[BATCH_SIZE],
    IDATA_T result_idx_all[COMPUTE_TIME * BATCH_SIZE]) {

  // declare arrays
  FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE];
  FDATA_T rnn_kernel_BRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE];
  FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
  FDATA_T fc_bias_BRAM[FC_OUTPUT_SIZE];

  FDATA_T fc_output_cache[BATCH_SIZE * FC_OUTPUT_SIZE];

  FDATA_T rnn_input_state_BRAM[BATCH_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_state0_BRAM[BATCH_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_state1_BRAM[BATCH_SIZE * RNN_STATE_SIZE];
  IDATA_T result_idx_one_step0[BATCH_SIZE];
  IDATA_T result_idx_one_step1[BATCH_SIZE];

  // copy all inputs from DRAM to BRAM
  copy_word_embedding(word_embedding_BRAM, word_embedding);
  copy_rnn_kernel(rnn_kernel_BRAM, rnn_kernel);
  copy_rnn_recurrent_kernel(rnn_recurrent_kernel_BRAM, rnn_recurrent_kernel);
  copy_rnn_bias(rnn_bias_BRAM, rnn_bias);
  copy_fc_kernel(fc_kernel_BRAM, fc_kernel);
  copy_fc_bias(fc_kernel_BRAM, fc_bias);
  copy_rnn_init_state(rnn_state0_BRAM, rnn_init_state);
  copy_rnn_init_idx(result_idx_one_step0, rnn_init_idx);

  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME / 2;
       compute_time++) {

    // Use ping-pong buffer
    LDATA_T result_idx_all_idx = 2 * compute_time * BATCH_SIZE;
    wrapper_rnn_fc(
        word_embedding_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM,
        rnn_bias_BRAM, fc_kernel_BRAM, fc_bias_BRAM,
        /* input_word_idx = */result_idx_one_step0, rnn_input_state_BRAM,
        /* rnn_last_state = */rnn_state0_BRAM,
		/* rnn_output_state = */rnn_state1_BRAM,
        fc_output_cache, /* result_idx = */result_idx_one_step1);
    result_to_DRAM(result_idx_one_step1, result_idx_all + result_idx_all_idx);

    result_idx_all_idx = (2 * compute_time + 1) * BATCH_SIZE;
    wrapper_rnn_fc(
        word_embedding_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM,
        rnn_bias_BRAM, fc_kernel_BRAM, fc_bias_BRAM,
        /* input_word_idx = */result_idx_one_step1, rnn_input_state_BRAM,
        /* rnn_last_state = */rnn_state1_BRAM,
		/* rnn_output_state = */rnn_state0_BRAM,
        fc_output_cache, /* result_idx = */result_idx_one_step0);
    result_to_DRAM(result_idx_one_step0, result_idx_all + result_idx_all_idx);
  }
}

////////////////////           Layer Wrapper                ////////////////////

// finish 1 batch, e.g. 64, of computation, return the result indexes
void wrapper_rnn_fc(
    FDATA_T word_embedding[WORD_NUM * WORD_SIZE],
    FDATA_T rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_bias[RNN_STATE_SIZE],
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    FDATA_T fc_bias[FC_OUTPUT_SIZE],
    IDATA_T input_word_idx[BATCH_SIZE],
    FDATA_T rnn_input_state_cache[BATCH_SIZE * RNN_INPUT_SIZE],
    FDATA_T rnn_last_state[BATCH_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_output_state[BATCH_SIZE * RNN_STATE_SIZE],
    FDATA_T fc_output_cache[BATCH_SIZE * FC_OUTPUT_SIZE],
    IDATA_T result_idx[BATCH_SIZE]) {
  // input:
  //  word_embedding, rnn weights, and fc weights
  //  last state, input word_idx
  // output:
  //  rnn_output_state, current generated word index
  // cache:
  //  fc_output_cache, avoid malloc every time we call this function

  init_rnn_state(rnn_output_state);
  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
    LDATA_T rnn_input_state_cache_idx = i * RNN_INPUT_SIZE;
    copy_word_vector(rnn_input_state_cache + rnn_input_state_cache_idx,
                     &word_embedding[input_word_idx[i] * RNN_INPUT_SIZE]);
  }

  rnn(rnn_last_state, rnn_input_state_cache,
      rnn_bias, rnn_kernel, rnn_recurrent_kernel, rnn_output_state);

  init_fc_state(fc_output_cache);
  // the output state feed to fc layer
  fc(/* input_feature_map = */rnn_output_state, fc_bias, fc_kernel,
     /* output_feature_map = */fc_output_cache);

  argmax (fc_output_cache, result_idx);
}

////////////////////           Layer Functions              ////////////////////

void rnn(FDATA_T last_state[BATCH_SIZE * RNN_STATE_SIZE],
         FDATA_T input_state[BATCH_SIZE * RNN_INPUT_SIZE],
         FDATA_T bias[RNN_STATE_SIZE],
         FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
         FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
         FDATA_T output_state[BATCH_SIZE * RNN_STATE_SIZE]) {
  // please do INITIALIZATION before input output_state
  // ------- DIMENSION SETTING  ---------- *
  //
  //   input_state: BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
  //   last_state: BATCH_SIZE * RNN_STATE_SIZE (None * 128)
  //   bias: RNN_STATE_SIZE (128)
  //   kernel: transposed -> RNN_STATE_SIZE * RNN_INPUT_SIZE (128 * 100)
  //   recurrent_kernel: transposed -> RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128)
  //   output_state: BATCH_SIZE * RNN_STATE_SIZE (None, 128)

  //  computation:
  //
  //    for each sample in batch:
  //    output_state = input_state mul kernel +
  //                   last_state mul recurrent_kernel +
  //                   bias

  for (LDATA_T batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
    // placeholder: loop naming
    // compute each sample in a batch

    for (LDATA_T output_state_index = 0; output_state_index < RNN_STATE_SIZE;
         output_state_index++) {
      // placeholder: loop naming
      // compute output_state[batch_index][output_state_index]

      // each output_state state has STATE_SIZE elements, compute each of them
      // * each computation is a vector vector multiplication
      // * vector 1: last_state concatenate input_state
      // * vector 2: a row of weights

      // output_state[batch_index][output_state_index]
      LDATA_T current_output_state_index =
          batch_index * RNN_STATE_SIZE + output_state_index;

      // initialize to 0
      output_state[current_output_state_index] = 0;

      // do multiplication: weights by last state
      for (LDATA_T last_state_index = 0; last_state_index < RNN_STATE_SIZE;
           last_state_index++) {
        // placeholder: loop naming

        // output_state[batch_index][output_state_index] +=
        //                 last_state[batch_index][last_state_index] *
        //                recurrent_kernel[output_state_index][last_state_index]

        // last_state[batch_index][last_state_index]
        LDATA_T current_last_state_index =
            batch_index * RNN_STATE_SIZE + last_state_index;

        // recurrent_kernel[output_state_index][last_state_index]
        LDATA_T current_recurrent_kernel_index =
            output_state_index * RNN_STATE_SIZE + last_state_index;

        // do multiplication, add to previous value
        // pr f("%f", last_state[current_last_state_index]);
        output_state[current_output_state_index] +=
            last_state[current_last_state_index] *
            recurrent_kernel[current_recurrent_kernel_index];
      }

      // do multiplication: weights by input_state
      for(LDATA_T input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
          input_state_index++) {
        // placeholder: loop naming

        // output_state[batch_index][output_state_index] +=
        //                input_state[batch_index][input_state_index] *
        //                kernel[output_state_index][input_state_index]

        // input_state[batch_index][input_state_index]
        LDATA_T current_input_state_index =
            batch_index * RNN_INPUT_SIZE + input_state_index;

        // kernel[output_state_index][input_state_index]
        LDATA_T current_kernel_index = output_state_index * RNN_INPUT_SIZE +
            input_state_index;

        // do multiplication, add to previous value
        output_state[current_output_state_index] +=
            input_state[current_input_state_index] *
            kernel[current_kernel_index];
      }

      // add bias
      // bias[output_state_index]
      FDATA_T tmp = output_state[current_output_state_index] +
                    bias[output_state_index];
      output_state[current_output_state_index] = FDATA_T(tanh(TOFLOAT(tmp)));
    }
  }
}

void fc(FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
        FDATA_T bias[FC_OUTPUT_SIZE],
        FDATA_T kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
        FDATA_T output_feature_map[BATCH_SIZE * FC_OUTPUT_SIZE]) {

  // please do INITIALIZATION before input output_feature_map
  // ------- DIMENSION SETTING  ----------

  //  input_feature_map: BATCH_SIZE * FC_INPUT_SIZE (None * 128)
  //  bias: FC_OUTPUT_SIZE (16192)
  //  kernel: tranposed -> FC_OUTPUT_SIZE * FC_INPUT_SIZE  (16192 * 128)
  //  output_feature_map: BATCH_SIZE * FC_OUTPUT_SIZE (None * 16192)

  for (LDATA_T batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
    // compute each sample in a batch

    for (LDATA_T output_feature_map_index = 0;
         output_feature_map_index < FC_OUTPUT_SIZE;
         output_feature_map_index++) {

      // compute output_feature_map[batch_index][output_feature_map_index]
      // each output_feature_map has FC_OUTPUT_SIZE elements, compute each of them
      //  * each computation is a vector vector multiplication
      //  * vector 1: input_feature_map
      //  * vector 2: a row of weights

      // output_feature_map[batch_index][output_feature_map_index]
      LDATA_T current_output_feature_map_index = batch_index * FC_OUTPUT_SIZE +
          output_feature_map_index;

      // initialize to 0
      output_feature_map[current_output_feature_map_index] = 0;

      for (LDATA_T input_feature_map_index = 0;
          input_feature_map_index < FC_INPUT_SIZE;
          input_feature_map_index++) {

        // output_feature_map[batch_index][output_feature_map_index] +=
        //      input_feature_map[batch_index][input_feature_map_index] *
        //      kernel[output_feature_map_index][input_feature_map_index]

        // input_feature_map[batch_index][input_feature_map_index]
        LDATA_T current_input_feature_map_index =
            batch_index * FC_INPUT_SIZE + input_feature_map_index;

        // kernel[output_feature_map_index][input_feature_map_index]
        LDATA_T current_kernel_index =
            output_feature_map_index * FC_INPUT_SIZE + input_feature_map_index;

        // do multiplication, add to previous value
        output_feature_map[current_output_feature_map_index] +=
            input_feature_map[current_input_feature_map_index] *
            kernel[current_kernel_index];
      }
      // add bias: bias[current_output_feature_map_index]
      output_feature_map[current_output_feature_map_index] +=
          bias[output_feature_map_index];
    }
  }
}

void argmax(FDATA_T* input, IDATA_T* result) {
    // input: a probability distribution (BATCH_SIZE, SM_OUTPUT_SIZE)
    // result: the index of each output (BATCH_SIZE, )
    for (LDATA_T batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
        LDATA_T max_index = 0;
        FDATA_T max_val = input[batch_index * SM_CLASS_SIZE];
        for (LDATA_T find_max_index = 0; find_max_index < SM_CLASS_SIZE; find_max_index++) {
            // input[batch_index][find_max_index]
            LDATA_T input_index = batch_index * SM_CLASS_SIZE + find_max_index;
            if (input[input_index] > max_val) {
                max_index = find_max_index;
                max_val = input[input_index];
            }
        }
        result[batch_index] = max_index;
    }
}

////////////////////           Utility Functions          ////////////////////

void copy_word_embedding(FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE],
                         FDATA_T word_embedding_DRAM[WORD_NUM * WORD_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < WORD_NUM * WORD_NUM; i++) {
#pragma HLS pipeline
    word_embedding_BRAM[i] = word_embedding_DRAM[i];
  }
}

void copy_rnn_kernel(FDATA_T rnn_kernel_BRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE],
                     FDATA_T rnn_kernel_DRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < RNN_INPUT_SIZE * RNN_STATE_SIZE; i++) {
#pragma HLS pipeline
    rnn_kernel_BRAM[i] = rnn_kernel_DRAM[i];
  }
}

void copy_rnn_recurrent_kernel(
    FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_recurrent_kernel_DRAM[RNN_STATE_SIZE * RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < RNN_STATE_SIZE * RNN_STATE_SIZE; i++) {
#pragma HLS pipeline
    rnn_recurrent_kernel_BRAM[i] = rnn_recurrent_kernel_DRAM[i];
  }
}

void copy_rnn_bias(FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE],
                   FDATA_T rnn_bias_DRAM[RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < RNN_STATE_SIZE; i++) {
#pragma HLS pipeline
    rnn_bias_BRAM[i] = rnn_bias_DRAM[i];
  }
}

void copy_fc_kernel(FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
                    FDATA_T fc_kernel_DRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < FC_OUTPUT_SIZE * FC_INPUT_SIZE; i++) {
#pragma HLS pipeline
    fc_kernel_BRAM[i] = fc_kernel_DRAM[i];
  }
}

void copy_fc_bias(FDATA_T fc_bias_BRAM[FC_OUTPUT_SIZE],
                  FDATA_T fc_bias_DRAM[FC_OUTPUT_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < FC_OUTPUT_SIZE; i++) {
#pragma HLS pipeline
    fc_bias_BRAM[i] = fc_bias_DRAM[i];
  }
}

void copy_rnn_init_state(
    FDATA_T rnn_state_BRAM[BATCH_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_init_state_DRAM[BATCH_SIZE * RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < BATCH_SIZE * RNN_INPUT_SIZE; i++) {
#pragma HLS pipeline
    rnn_state_BRAM[i] = rnn_init_state_DRAM[i];
  }
}

void copy_rnn_init_idx(IDATA_T rnn_idx_BRAM[BATCH_SIZE],
                       IDATA_T rnn_idx_DRAM[BATCH_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS pipeline
    rnn_idx_DRAM[i] = rnn_idx_DRAM[i];
  }
}

void copy_word_vector(FDATA_T rnn_input_state_BRAM[RNN_STATE_SIZE],
                      FDATA_T word_embedding_BRAM[RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < RNN_STATE_SIZE; i++)
#pragma HLS pipeline
    rnn_input_state_BRAM[i] = word_embedding_BRAM[i];
}

void init_rnn_state(FDATA_T state[BATCH_SIZE * RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < BATCH_SIZE * RNN_STATE_SIZE; i++)
#pragma HLS pipeline
    state[i] = 0;
}

void init_fc_state(FDATA_T state[BATCH_SIZE * FC_OUTPUT_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < BATCH_SIZE * FC_OUTPUT_SIZE; i++)
#pragma HLS pipeline
    state[i] = 0;
}

void result_to_DRAM(IDATA_T result_idx_BRAM[BATCH_SIZE],
    IDATA_T result_idx_DRAM[BATCH_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS pipeline
    result_idx_DRAM[i] = result_idx_BRAM[i];
  }
}
