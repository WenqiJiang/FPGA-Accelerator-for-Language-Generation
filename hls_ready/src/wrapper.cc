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
#pragma SDS data copy(rnn_kernel[0: RNN_STATE_SIZE * RNN_INPUT_SIZE])
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
    FDATA_T rnn_kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_bias[RNN_STATE_SIZE],
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    FDATA_T fc_bias[FC_OUTPUT_SIZE],
    FDATA_T rnn_init_state[BATCH_SIZE * RNN_STATE_SIZE],
    IDATA_T rnn_init_idx[BATCH_SIZE],
    IDATA_T result_idx_all[COMPUTE_TIME * BATCH_SIZE]) {

  // declare arrays
  FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE];
  FDATA_T rnn_kernel_BRAM[RNN_STATE_SIZE * RNN_INPUT_SIZE];
  FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE];
  FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
  FDATA_T fc_bias_BRAM[FC_OUTPUT_SIZE];


// this value equal to WORD_SIZE / RNN_TILE_NUM
#pragma HLS array_partition variable=word_embedding_BRAM cyclic factor=8

// This two partition factor depends on load kernel clock cycle requirement
#pragma HLS array_partition variable=rnn_kernel_BRAM cyclic factor=8
#pragma HLS array_partition variable=rnn_recurrent_kernel_BRAM cyclic factor=8

// This partition factor depends on load kernel clock cycle requirement
#pragma HLS array_partition variable=fc_kernel_BRAM cyclic factor=8

// This two factor depends on init speed requirement
#pragma HLS array_partition variable=rnn_bias_BRAM cyclic factor=8
#pragma HLS array_partition variable=fc_bias_BRAM cyclic factor=8

  FDATA_T rnn_input_state_BRAM[BATCH_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_state0_BRAM[BATCH_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_state1_BRAM[BATCH_SIZE * RNN_STATE_SIZE];
  IDATA_T result_idx_one_step0[BATCH_SIZE];
  IDATA_T result_idx_one_step1[BATCH_SIZE];

// This three partition factor depends on prefix sum clock cycle requirement
#pragma HLS array_partition variable=rnn_input_state_BRAM cyclic factor=64
#pragma HLS array_partition variable=rnn_state0_BRAM cyclic factor=64
#pragma HLS array_partition variable=rnn_state1_BRAM cyclic factor=64

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

    wrapper_rnn_fc(
        word_embedding_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM,
        rnn_bias_BRAM, fc_kernel_BRAM, fc_bias_BRAM,
        /* input_word_idx = */result_idx_one_step0, rnn_input_state_BRAM,
        /* rnn_last_state = */rnn_state0_BRAM,
        /* rnn_output_state = */rnn_state1_BRAM,
        /* result_idx = */result_idx_one_step1);
    LDATA_T result_idx_all_idx = 2 * compute_time * BATCH_SIZE;
    result_to_DRAM(result_idx_one_step1, result_idx_all + result_idx_all_idx);

    wrapper_rnn_fc(
        word_embedding_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM,
        rnn_bias_BRAM, fc_kernel_BRAM, fc_bias_BRAM,
        /* input_word_idx = */result_idx_one_step1, rnn_input_state_BRAM,
        /* rnn_last_state = */rnn_state1_BRAM,
        /* rnn_output_state = */rnn_state0_BRAM,
        /* result_idx = */result_idx_one_step0);
    result_idx_all_idx = (2 * compute_time + 1) * BATCH_SIZE;
    result_to_DRAM(result_idx_one_step0, result_idx_all + result_idx_all_idx);
  }
}

////////////////////           Layer Wrapper                ////////////////////

// finish 1 batch, e.g. 64, of computation, return the result indexes
void wrapper_rnn_fc(
    FDATA_T word_embedding[WORD_NUM * WORD_SIZE],
    FDATA_T rnn_kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_bias[RNN_STATE_SIZE],
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
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

	FDATA_T fc_output_feature_map[FC_OUTPUT_SIZE * BATCH_SIZE];

  rnn_copy_batch_word_vector(rnn_input_state_cache, word_embedding,
                             input_word_idx);
  rnn(rnn_last_state, rnn_input_state_cache,
      rnn_bias, rnn_kernel, rnn_recurrent_kernel, rnn_output_state);

  // the output state feed to fc layer
  fc(/* input_feature_map = */rnn_output_state, fc_kernel, fc_bias,
     /* output_feature_map = */fc_output_feature_map);
	argmax(fc_output_feature_map, result_idx);
}

////////////////////           Layer Functions              ////////////////////

void rnn_copy_batch_word_vector(
    FDATA_T rnn_input_state_BRAM[BATCH_SIZE * RNN_INPUT_SIZE],
    FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE],
    IDATA_T input_word_idx[BATCH_SIZE]) {

  for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {

    LDATA_T word_idx = input_word_idx[batch_idx];
    for (LDATA_T i = 0; i < RNN_INPUT_SIZE; i++) {
#pragma HLS unroll complete
      rnn_input_state_BRAM[batch_idx * RNN_INPUT_SIZE + i] =
          word_embedding_BRAM[word_idx * WORD_SIZE + i];
    }
  }
}

// compute 1 time step, output state has been INITIALIZED to bias
void rnn(FDATA_T last_state[BATCH_SIZE * RNN_STATE_SIZE],
         FDATA_T input_state[BATCH_SIZE * RNN_INPUT_SIZE],
         FDATA_T bias[RNN_STATE_SIZE],
         FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
         FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
         FDATA_T output_state[BATCH_SIZE * RNN_STATE_SIZE]) {
  //   input_state: BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
  //   last_state: BATCH_SIZE * RNN_STATE_SIZE (None * 128)
  //   bias: RNN_STATE_SIZE (128)
  //   kernel: transposed -> RNN_STATE_SIZE * RNN_INPUT_SIZE (128 * 100)
  //   recurrent_kernel: transposed -> RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128)
  //   output_state: BATCH_SIZE * RNN_STATE_SIZE (None, 128)
	FDATA_T output_state_reg[BATCH_SIZE];
#pragma HLS ARRAY_PARTITION variable=output_state_reg complete

  FDATA_T kernel_reg[RNN_INPUT_SIZE];
  FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE];
#pragma HLS ARRAY_PARTITION variable=kernel_reg cyclic factor=64
#pragma HLS ARRAY_PARTITION variable=recurrent_kernel_reg cyclic factor=64

for (LDATA_T output_state_index = 0; output_state_index < RNN_STATE_SIZE;
         output_state_index++) {
#pragma HLS DATAFLOW
      // load
      rnn_load_kernel(kernel_reg, kernel + output_state_index * RNN_INPUT_SIZE);
      rnn_load_recurrent_kernel(recurrent_kernel_reg,
          recurrent_kernel + output_state_index * RNN_STATE_SIZE);

      // compute
      rnn_compute(input_state, last_state, kernel_reg,
								  recurrent_kernel_reg, output_state_reg);

			// save + add_bias + tanh
			rnn_save_output_state(output_state_reg, bias[output_state_index],
                            output_state_index,
                            output_state + batch_iter * RNN_STATE_SIZE);
    }
  }
}

// load one row of kernel from BRAM to register
void rnn_load_kernel(FDATA_T kernel_reg[RNN_INPUT_SIZE],
                     FDATA_T kernel_part[RNN_INPUT_SIZE]) {
  // load the (output_state_index)'th column of kernel
  // used this column do matrix multiplication
  // the kernel start from a certain index (decided when function call)
  // kernel --- load to ---> kernel_reg

  for (LDATA_T input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
       input_state_index++) {
#pragma HLS UNROLL complete
// #pragma HLS PIPELINE

    kernel_reg[input_state_index] = kernel_part[input_state_index];
  }
}

// load one row of kernel from BRAM to register
void rnn_load_recurrent_kernel(FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE],
                               FDATA_T recurrent_kernel_part[RNN_STATE_SIZE]) {
  // load the (output_state_index)'th column of recurrent_kernel
  // used this column do matrix multiplication
  // recurrent_kernel --- load to ---> recurrent_kernel_reg

  for (LDATA_T last_state_index = 0; last_state_index < RNN_STATE_SIZE;
       last_state_index++) {
#pragma HLS UNROLL complete
// #pragma HLS PIPELINE

    recurrent_kernel_reg[last_state_index] =
        recurrent_kernel_part[last_state_index];
  }
}

// compute a batch of input state, with a single row of kernel
void rnn_compute(
		FDATA_T input_state_reg[BATCH_SIZE * RNN_INPUT_SIZE],
		FDATA_T last_state_reg[BATCH_SIZE * RNN_STATE_SIZE],
		FDATA_T kernel_reg[RNN_INPUT_SIZE],
		FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE],
		FDATA_T output_state_reg_part[BATCH_SIZE]) {
//#pragma HLS inline region
  // take a batch of input_state and last_state,
  //  rnn_compute the output state, and store into the output_state_reg
  // note that we don't add bias here, the bias addition will be done in
  //  function "rnn_save_output_state"
  // input: input_state_reg, last_state_reg, kernel_reg,
  //          recurrent_kernel_reg, output_state_index
  // output: output_state_reg

#define COMPUTE_UNROLL 4
  FDATA_T local_reg[COMPUTE_UNROLL][RNN_STATE_SIZE + RNN_INPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_reg cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=local_reg cyclic factor=2 dim=1

for (LDATA_T tile_iter = 0; tile_iter < BATCH_SIZE / COMPUTE_UNROLL;
     tile_iter++) {

    for (LDATA_T batch_iter = 0; batch_iter < COMPUTE_UNROLL; batch_iter++) {
/////// HACKING, factor should be consistent with COMPUTE_UNROLL //////
/////// can not use macro as factor here due to the HLS syntax   //////
#pragma HLS UNROLL complete

      for (LDATA_T input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
           input_state_index++) {
//#pragma HLS RESOURCE variable=local_reg core=FMul_fulldsp
#pragma HLS UNROLL complete

        local_reg[batch_iter][input_state_index] =
            kernel_reg[input_state_index] *
            input_state_reg[(tile_iter * COMPUTE_UNROLL + batch_iter)
                * RNN_INPUT_SIZE + input_state_index];
      }

      for (LDATA_T last_state_index = 0; last_state_index < RNN_STATE_SIZE;
           last_state_index++) {
//#pragma HLS RESOURCE variable=local_reg core=FMul_fulldsp
#pragma HLS UNROLL complete

        local_reg[batch_iter][RNN_INPUT_SIZE + last_state_index] =
            recurrent_kernel_reg[last_state_index] *
            last_state_reg[tile_iter * COMPUTE_UNROLL + batch_iter]
            [last_state_index];
      }

      ////// HACKING, suppose RNN_STATE_SIZE + RNN_INPUT_SIZE = 228 /////

      // prefix sum
      for (LDATA_T i = 0; i < 114; i++) {
#pragma HLS UNROLL complete
        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][114 + i];
      }

      for (LDATA_T i = 0; i < 57; i++) {
#pragma HLS UNROLL complete
        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][57 + i];
      }

      // 57 = 28 * 2 + 1 -> need 29 reg for next iteration
      // the 57'th number will be copy to 29'th reg
      for (LDATA_T i = 0; i < 28; i++) {
#pragma HLS UNROLL complete
        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][28 + i];
      }
      local_reg[batch_iter][28] = local_reg[batch_iter][56];

      // 29 = 14 * 2 + 1 -> need 15 reg for next iteration
      // the 29'th number will be copy to 15'th reg
      for (LDATA_T i = 0; i < 14; i++) {
#pragma HLS UNROLL complete
        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][14 + i];
      }
      local_reg[batch_iter][14] = local_reg[batch_iter][28];

      // 15 = 7 * 2 + 1 -> need 8 reg for next iteration
      // the 15'th number will be copy to 8'th reg
      for (LDATA_T i = 0; i < 7; i++) {
#pragma HLS UNROLL complete
        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][7 + i];
      }
      local_reg[batch_iter][7] = local_reg[batch_iter][14];

      // from 8, regular prefix sum
      for (LDATA_T i = 0; i < 4; i++) {
#pragma HLS UNROLL complete
        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][4 + i];
      }

      // from 8, regular prefix sum
      for (LDATA_T i = 0; i < 2; i++) {
#pragma HLS UNROLL complete
        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][2 + i];
      }

      // from 8, regular prefix sum
      for (LDATA_T i = 0; i < 1; i++) {
#pragma HLS UNROLL complete
        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][1 + i];
      }

      output_state_reg_part[tile_iter * COMPUTE_UNROLL + batch_iter] =
          local_reg[batch_iter][0];
    }
  }
}

void rnn_save_output_state(FDATA_T output_state_reg[BATCH_SIZE],
                           FDATA_T bias, LDATA_T col,
                           FDATA_T output_state[BATCH_SIZE * RNN_STATE_SIZE]) {

  // the output state in register is not the final result,
  // add bias to finish computing and store them into BRAM
  // the output state starts from a certain index (decided when function call)
  // output state memory layout [BATCH_SIZE][RNN_STATE_SIZE]
  // output_state_reg + bias --- load to ---> output_state

  for (LDATA_T batch_iter = 0; batch_iter < BATCH_SIZE; batch_iter++) {
#pragma HLS UNROLL complete
// #pragma HLS PIPELINE

    FDATA_T tmp = bias + output_state_reg[batch_iter];
    LDATA_T output_state_index = batch_iter * RNN_STATE_SIZE + col;

    output_state[output_state_index] =
// if in Vivado HLS, use this:
//        hls::tanh<FXD_W_LENGTH, FXD_I_LENGTH>(tmp);
// if in SDSoC, use this:
        FDATA_T(tanh(TOFLOAT(tmp)));
// if neither works, use this (but result wouldn't be correct)
//        tmp;
  }
}

// compute a batch of state
void fc(FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
        FDATA_T kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
        FDATA_T bias[FC_OUTPUT_SIZE],
				FDATA_T output_feature_map[FC_OUTPUT_SIZE * BATCH_SIZE]) {
  //  input_feature_map: BATCH_SIZE * FC_INPUT_SIZE (None * 128)
  //  bias: FC_OUTPUT_SIZE (6144)
  //  kernel:  FC_OUTPUT_SIZE * FC_INPUT_SIZE
  // maximum_output_idx: an array of idx (BATCH_SIZE, )

	FDATA_T output_feature_map_reg[BATCH_SIZE];
#pragma HLS array_partition variable=output_feature_map_reg complete

  for (LDATA_T output_feature_map_index = 0;
       output_feature_map_index < FC_OUTPUT_SIZE;
       output_feature_map_index++) {
#pragma HLS DATAFLOW

    // load
    LDATA_T kernel_offset = output_feature_map_index * FC_INPUT_SIZE;
    fc_load_kernel(kernel_reg, kernel + kernel_offset);

    // compute
    fc_compute(input_feature_map, kernel_reg, output_feature_map);

    // save
    LDATA_T output_feature_map_offset = output_feature_map_index * BATCH_SIZE;
    fc_save_output_feature_map(
        output_feature_map_reg, fc_bias[output_feature_map_index],
        output_feature_map + output_feature_map_offset);
  }
}

// load one row of kernel from BRAM to register
void fc_load_kernel(FDATA_T kernel_reg[FC_INPUT_SIZE],
                    FDATA_T kernel_BRAM_part[FC_INPUT_SIZE]) {
  // kernel_DRAM: FC_OUTPUT_SIZE * FC_INPUT_SIZE
  // kernel_reg: FC_INPUT_SIZE
  // output_feature_map_index: which column to read LDATA_To reg

  for (LDATA_T input_feature_map_index = 0;
       input_feature_map_index < FC_INPUT_SIZE;
       input_feature_map_index++) {
#pragma HLS unroll complete

    kernel_reg[input_feature_map_index] =
        kernel_BRAM_part[input_feature_map_index];
  }
}

// compute a batch of output feature map given a single row of feature map
void fc_compute(
		FDATA_T input_feature_map_reg[BATCH_SIZE * FC_INPUT_SIZE],
		FDATA_T kernel_reg[FC_INPUT_SIZE],
		FDATA_T output_feature_map[BATCH_SIZE]) {

  // initialization
  FDATA_T local_reg[FC_TILE_SIZE][FC_INPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_reg dim=2 cyclic factor=32
#pragma HLS ARRAY_PARTITION variable=local_reg dim=1 cyclic factor=2
  for (LDATA_T iter = 0; iter < BATCH_SIZE / FC_TILE_SIZE; iter++) {

    LDATA_T start_batch = iter * FC_TILE_SIZE;

    for (LDATA_T batch_idx = 0; batch_idx < FC_TILE_SIZE; batch_idx++) {
#pragma HLS UNROLL complete
      // compute
      for (LDATA_T i = 0; i < FC_INPUT_SIZE; i++) {
#pragma HLS UNROLL complete
//#pragma HLS RESOURCE variable=local_reg core=FMul_fulldsp
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = kernel_reg[i] *
						input_feature_map_reg[(start_batch + batch_idx)*FC_INPUT_SIZE + i];
      }

      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 2; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] +
            local_reg[batch_idx][i + FC_INPUT_SIZE / 2];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 4; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] +
            local_reg[batch_idx][i + FC_INPUT_SIZE / 4];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 8; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] +
            local_reg[batch_idx][i + FC_INPUT_SIZE / 8];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 16; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] +
            local_reg[batch_idx][i + FC_INPUT_SIZE / 16];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 32; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] +
            local_reg[batch_idx][i + FC_INPUT_SIZE / 32];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 64; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] +
            local_reg[batch_idx][i + FC_INPUT_SIZE / 64];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 128; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] +
            local_reg[batch_idx][i + FC_INPUT_SIZE / 128];
      }
      output_feature_map_reg[start_batch + batch_idx] = local_reg[batch_idx][0];
    }
  }
}

void fc_save_output_feature_map(
    FDATA_T output_feature_map_reg[BATCH_SIZE], FDATA_T bias_reg_single,
    FDATA_T output_feature_map_part[BATCH_SIZE]) {
  // save  outputs a time
  // output_feature_map_reg: BATCH_SIZE x FC_OUTPUT_SIZE
  // output_feature_map_DRAM -> transposed: FC_OUTPUT_SIZE x BATCH_SIZE
  // start_batch_index: which batch to save to BRAM

  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS PIPELINE
    output_feature_map_part[i] =
        bias_reg_single + output_feature_map_reg[i];
  }
}

void argmax(FDATA_T fc_output_feature_map[FC_OUTPUT_SIZE * BATCH_SIZE],
						 IDATA_T result_idx) {

  for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {

    FDATA_T max_val = fc_output_feature_map[0];
    IDATA_T max_idx = 0;

    for (IDATA_T idx = 0; idx < FC_OUTPUT_SIZE; idx++) {
      if (fc_output_feature_map[batch_idx * BATCH_SIZE + idx] > max_val) {
        max_val = fc_output_feature_map[batch_idx * BATCH_SIZE + idx];
        max_idx = idx;
      }
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

void copy_rnn_kernel(FDATA_T rnn_kernel_BRAM[RNN_STATE_SIZE * RNN_INPUT_SIZE],
                     FDATA_T rnn_kernel_DRAM[RNN_STATE_SIZE * RNN_INPUT_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < RNN_STATE_SIZE * RNN_INPUT_SIZE; i++) {
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

void result_to_DRAM(IDATA_T result_idx_BRAM[BATCH_SIZE],
    IDATA_T result_idx_DRAM[BATCH_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS pipeline
    result_idx_DRAM[i] = result_idx_BRAM[i];
  }
}
