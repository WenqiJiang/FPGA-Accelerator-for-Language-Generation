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
#pragma SDS data copy(fc_kernel[0: FC_INPUT_SIZE * FC_OUTPUT_SIZE])
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
    FDATA_T fc_kernel[FC_INPUT_SIZE * FC_OUTPUT_SIZE],
    FDATA_T fc_bias[FC_OUTPUT_SIZE],
    FDATA_T rnn_init_state[BATCH_SIZE * RNN_STATE_SIZE],
    IDATA_T rnn_init_idx[BATCH_SIZE],
    IDATA_T result_idx_all[COMPUTE_TIME * BATCH_SIZE]) {

  // declare arrays
  FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE];
  FDATA_T rnn_kernel_BRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE];
  FDATA_T fc_kernel_BRAM[FC_INPUT_SIZE * FC_OUTPUT_SIZE];
  FDATA_T fc_bias_BRAM[FC_OUTPUT_SIZE];

// this value shoule be equal to BATCH_SIZE
#pragma HLS array_partition variable=word_embedding_BRAM block factor=32
// this value should be equal to FC_BATCH_SIZE
#pragma HLS array_partition variable=fc_kernel_BRAM block factor=32
// this value should be equal to FC_BATCH_SIZE
#pragma HLS array_partition variable=fc_bias_BRAM block factor=32

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
        /* result_idx = */result_idx_one_step1);
    result_to_DRAM(result_idx_one_step1, result_idx_all + result_idx_all_idx);

    result_idx_all_idx = (2 * compute_time + 1) * BATCH_SIZE;
    wrapper_rnn_fc(
        word_embedding_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM,
        rnn_bias_BRAM, fc_kernel_BRAM, fc_bias_BRAM,
        /* input_word_idx = */result_idx_one_step1, rnn_input_state_BRAM,
        /* rnn_last_state = */rnn_state1_BRAM,
        /* rnn_output_state = */rnn_state0_BRAM,
        /* result_idx = */result_idx_one_step0);
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
    FDATA_T fc_kernel[FC_INPUT_SIZE * FC_OUTPUT_SIZE],
    FDATA_T fc_bias[FC_OUTPUT_SIZE],
    IDATA_T input_word_idx[BATCH_SIZE],
    FDATA_T rnn_input_state_cache[BATCH_SIZE * RNN_INPUT_SIZE],
    FDATA_T rnn_last_state[BATCH_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_output_state[BATCH_SIZE * RNN_STATE_SIZE],
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
#pragma HLS unroll complete
    LDATA_T rnn_input_state_cache_idx = i * RNN_INPUT_SIZE;
    copy_word_vector(rnn_input_state_cache + rnn_input_state_cache_idx,
                     &word_embedding[input_word_idx[i] * RNN_INPUT_SIZE]);
  }

  rnn(rnn_last_state, rnn_input_state_cache,
      rnn_bias, rnn_kernel, rnn_recurrent_kernel, rnn_output_state);

  // the output state feed to fc layer
  fc(/* input_feature_map = */rnn_output_state, fc_kernel, fc_bias,
     /* output_feature_map = */result_idx);
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

  for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
    // placeholder: loop naming
    // compute each sample in a batch

    for (LDATA_T output_state_index = 0; output_state_index < RNN_STATE_SIZE;
         output_state_index++) {
      // placeholder: loop naming
      // compute output_state[batch_idx][output_state_index]

      // each output_state state has STATE_SIZE elements, compute each of them
      // * each computation is a vector vector multiplication
      // * vector 1: last_state concatenate input_state
      // * vector 2: a row of weights

      // output_state[batch_idx][output_state_index]
      LDATA_T current_output_state_index =
          batch_idx * RNN_STATE_SIZE + output_state_index;

      // initialize to 0
      output_state[current_output_state_index] = 0;

      // do multiplication: weights by last state
      for (LDATA_T last_state_index = 0; last_state_index < RNN_STATE_SIZE;
           last_state_index++) {
        // placeholder: loop naming

        // output_state[batch_idx][output_state_index] +=
        //                 last_state[batch_idx][last_state_index] *
        //                recurrent_kernel[output_state_index][last_state_index]

        // last_state[batch_idx][last_state_index]
        LDATA_T current_last_state_index =
            batch_idx * RNN_STATE_SIZE + last_state_index;

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

        // output_state[batch_idx][output_state_index] +=
        //                input_state[batch_idx][input_state_index] *
        //                kernel[output_state_index][input_state_index]

        // input_state[batch_idx][input_state_index]
        LDATA_T current_input_state_index =
            batch_idx * RNN_INPUT_SIZE + input_state_index;

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
        FDATA_T kernel[FC_INPUT_SIZE * FC_OUTPUT_SIZE],
        FDATA_T bias[FC_OUTPUT_SIZE],
        IDATA_T maximum_output_idx[BATCH_SIZE]) {

  // please do INITIALIZATION before input output_feature_map
  // ------- DIMENSION SETTING  ----------

  //  input_feature_map: BATCH_SIZE * FC_INPUT_SIZE (None * 128)
  //  bias: FC_OUTPUT_SIZE (6144)
  //  kernel:  FC_INPUT_SIZE * FC_OUTPUT_SIZE
  // maximum_output_idx: an array of idx (BATCH_SIZE, )

  // cache the result of 1 TILE of the entire batch [BATCH_SIZE][FC_TILE_SIZE]
  FDATA_T output_feature_map_cache[FC_TILE_SIZE * BATCH_SIZE];
#pragma HLS array_partition variable=output_feature_map_cache cyclic factor=32

  FDATA_T maximum_output[BATCH_SIZE];

  init_fc_state(output_feature_map_cache);
  for (LDATA_T tile_iter = 0; tile_iter < FC_OUTPUT_SIZE / FC_TILE_SIZE;
       tile_iter++) {
    // cannot read and write output_feature_map_cache in fc_compute_tile
// #pragma HLS dataflow
    LDATA_T kernel_start_idx = tile_iter * FC_TILE_SIZE;
    fc_compute_tile(input_feature_map, kernel, bias, kernel_start_idx,
                    output_feature_map_cache);
    fc_tile_argmax(output_feature_map_cache, maximum_output,
                   maximum_output_idx, kernel_start_idx);
  }
}

void fc_compute_tile(
    FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
    FDATA_T kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    FDATA_T bias[FC_OUTPUT_SIZE],
    LDATA_T start_feature_map_idx,
    FDATA_T output_feature_map_cache[FC_TILE_SIZE * BATCH_SIZE]) {

  // one row of input feature map / kernel, stored in registers
  FDATA_T input_feature_map_reg[BATCH_SIZE];
  FDATA_T kernel_tile_reg[FC_TILE_SIZE];

#pragma HLS array_partition variable=input_feature_map_reg complete
#pragma HLS array_partition variable=kernel_tile_reg complete

  fc_init_cache(output_feature_map_cache);

  for (LDATA_T input_feature_map_idx = 0;
       input_feature_map_idx < FC_INPUT_SIZE; input_feature_map_idx++) {
    // cannot read and write output_feature_map_cache in fc_mac
// #pragma HLS dataflow
    fc_copy_input_FM_row(input_feature_map_reg, input_feature_map,
                         start_feature_map_idx + input_feature_map_idx);
    fc_copy_kernel_row(kernel_tile_reg, kernel, input_feature_map_idx);
    fc_mac(input_feature_map_reg, kernel_tile_reg, output_feature_map_cache);
  }
  fc_add_bias(output_feature_map_cache, bias, start_feature_map_idx);
}

// copy one row of input feature map
void fc_copy_input_FM_row(
    FDATA_T input_feature_map_reg[BATCH_SIZE],
    FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
    LDATA_T input_feature_map_idx) {

  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS unroll complete
    input_feature_map_reg[i] =
        input_feature_map[input_feature_map_idx * FC_INPUT_SIZE + i];
  }
}

// copy one row of kernel
void fc_copy_kernel_row(
    FDATA_T kernel_tile_reg[FC_TILE_SIZE],
    FDATA_T kernel_tile[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    LDATA_T kernel_idx) {

  for (LDATA_T i = 0; i < FC_TILE_SIZE; i++) {
#pragma HLS unroll complete
    kernel_tile_reg[i] = kernel_tile[kernel_idx * FC_INPUT_SIZE + i];
  }
}

void fc_mac(FDATA_T input_feature_map_reg[BATCH_SIZE],
            FDATA_T kernel_tile_reg[FC_TILE_SIZE],
            FDATA_T output_feature_map_cache[FC_TILE_SIZE * BATCH_SIZE]) {

  for (LDATA_T tile_idx = 0; tile_idx < FC_TILE_SIZE; tile_idx++) {
// this unroll factor depends on output_feature_map_cache unroll factor
#pragma HLS unroll factor=2
    for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
#pragma HLS unroll complete
      output_feature_map_cache[tile_idx * BATCH_SIZE + batch_idx] +=
          input_feature_map_reg[batch_idx] * kernel_tile_reg[tile_idx];
    }
  }
}

// init cache to 0s
void fc_init_cache(FDATA_T state[FC_TILE_SIZE * BATCH_SIZE]) {
  for (LDATA_T i = 0; i < FC_TILE_SIZE; i++) {
// this unroll factor depends on output_feature_map_cache unroll factor
// #pragma HLS unroll factor=2
    for (LDATA_T b = 0; b < BATCH_SIZE; b++) {
#pragma HLS unroll complete
      state[i * FC_TILE_SIZE + b] = 0;
    }
  }
}

// given a tile of output FM, compare with the history to find the maximum
// value and index
void fc_tile_argmax(FDATA_T output_feature_map_cache[FC_TILE_SIZE * BATCH_SIZE],
                    FDATA_T global_maximum_output[BATCH_SIZE],
                    IDATA_T global_maximum_output_idx[BATCH_SIZE],
                    LDATA_T start_idx) {
 // start_idx: the tile start from which kernel index, from 0 to 6143

  FDATA_T local_maximum_output[BATCH_SIZE];
  IDATA_T local_maximum_output_idx[BATCH_SIZE];
#pragma HLS array_partition local_maximum_output complete
#pragma HLS array_partition local_maximum_output_idx complete

  // init
  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS unroll complete
    local_maximum_output[i] = 0;
    local_maximum_output_idx[i] = 0;
  }

  // find local maximum value
  for (LDATA_T tile_idx = 0; tile_idx < FC_TILE_SIZE; tile_idx++) {

    for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
#pragma HLS unroll complete
      if (output_feature_map_cache[tile_idx * BATCH_SIZE + batch_idx] >
          local_maximum_output[batch_idx]) {
        local_maximum_output[batch_idx] =
            output_feature_map_cache[tile_idx * BATCH_SIZE + batch_idx];
        local_maximum_output_idx[batch_idx] = start_idx + tile_idx;
      }
    }
  }

  // update global maximum
  for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
#pragma HLS pipeline
    if (global_maximum_output[batch_idx] < local_maximum_output[batch_idx]) {
      global_maximum_output[batch_idx] = local_maximum_output[batch_idx];
      global_maximum_output_idx[batch_idx] =
          local_maximum_output_idx[batch_idx];
    }
  }
}

void fc_add_bias(FDATA_T output_feature_map[FC_TILE_SIZE * BATCH_SIZE],
                 FDATA_T bias[FC_OUTPUT_SIZE], LDATA_T bias_start_idx) {

  for (LDATA_T tile_idx = 0; tile_idx < FC_TILE_SIZE; tile_idx++) {

    FDATA_T bias_reg = bias[tile_idx + bias_start_idx];
    for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
#pragma HLS unroll complete
      output_feature_map[tile_idx * BATCH_SIZE + batch_idx] += bias_reg;
    }
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

void copy_fc_kernel(FDATA_T fc_kernel_BRAM[FC_INPUT_SIZE * FC_OUTPUT_SIZE],
                    FDATA_T fc_kernel_DRAM[FC_INPUT_SIZE * FC_OUTPUT_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < FC_INPUT_SIZE * FC_OUTPUT_SIZE; i++) {
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

void init_fc_state(FDATA_T state[FC_TILE_SIZE * BATCH_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < FC_TILE_SIZE * BATCH_SIZE; i++)
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
