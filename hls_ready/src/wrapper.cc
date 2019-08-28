#include "wrapper.h"

#include "constants.h"
#include "types.h"

////////////////////         TOP-LEVEL FUNCTION             ////////////////////

// weights
#pragma SDS data zero_copy(word_embedding[0: WORD_NUM * WORD_SIZE])
#pragma SDS data copy(rnn_kernel[0: RNN_STATE_SIZE * RNN_INPUT_SIZE])
#pragma SDS data copy( \
    rnn_recurrent_kernel[0: RNN_STATE_SIZE * RNN_STATE_SIZE])
#pragma SDS data copy(rnn_bias[0: RNN_STATE_SIZE])
#pragma SDS data zero_copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
#pragma SDS data copy(fc_bias[0: FC_OUTPUT_SIZE])

// input states and indexes
#pragma SDS data copy(rnn_init_state[0: BATCH_SIZE * RNN_STATE_SIZE])
#pragma SDS data copy(rnn_init_idx[0: BATCH_SIZE])

// result indexes
#pragma SDS data copy(result_idx_all[0: COMPUTE_TIME * BATCH_SIZE])

// data access pattern
#pragma SDS data access_pattern( \
  rnn_kernel: SEQUENTIAL, \
  rnn_recurrent_kernel: SEQUENTIAL, \
  rnn_bias: SEQUENTIAL, \
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
  // FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE];
  FDATA_T rnn_kernel_BRAM[RNN_STATE_SIZE * RNN_INPUT_SIZE];
  FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE];
  FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE];
  FDATA_T fc_bias_BRAM[FC_OUTPUT_SIZE];


// this value equal to WORD_SIZE / RNN_TILE_NUM
// #pragma HLS array_partition variable=word_embedding_BRAM cyclic factor=8

// This two partition factor depends on load kernel clock cycle requirement
// #pragma HLS array_partition variable=rnn_kernel_BRAM cyclic factor=8
// #pragma HLS array_partition variable=rnn_recurrent_kernel_BRAM cyclic factor=8

// This partition factor depends on load kernel clock cycle requirement
 #pragma HLS array_partition variable=fc_kernel_BRAM cyclic factor=2

// This two factor depends on init speed requirement
// #pragma HLS array_partition variable=rnn_bias_BRAM cyclic factor=8
// #pragma HLS array_partition variable=fc_bias_BRAM cyclic factor=8

  FDATA_T rnn_input_state_BRAM[BATCH_SIZE * RNN_INPUT_SIZE];
  FDATA_T rnn_state0_BRAM[BATCH_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_state1_BRAM[BATCH_SIZE * RNN_STATE_SIZE];
  IDATA_T result_idx_one_step0[BATCH_SIZE];
  IDATA_T result_idx_one_step1[BATCH_SIZE];

// This three partition factor depends on prefix sum clock cycle requirement
#pragma HLS array_partition variable=rnn_input_state_BRAM cyclic factor=50
#pragma HLS array_partition variable=rnn_state0_BRAM cyclic factor=128
#pragma HLS array_partition variable=rnn_state1_BRAM cyclic factor=128

// This are partitioned for argmax
#pragma HLS array_partition variable=result_idx_one_step0 complete
#pragma HLS array_partition variable=result_idx_one_step1 complete

  // copy all inputs from DRAM to BRAM
  // copy_word_embedding(word_embedding_BRAM, word_embedding);
  copy_rnn_kernel(rnn_kernel_BRAM, rnn_kernel);
  copy_rnn_recurrent_kernel(rnn_recurrent_kernel_BRAM, rnn_recurrent_kernel);
  copy_rnn_bias(rnn_bias_BRAM, rnn_bias);
  copy_fc_kernel(fc_kernel_BRAM, fc_kernel);
  copy_fc_bias(fc_bias_BRAM, fc_bias);
  copy_rnn_init_state(rnn_state0_BRAM, rnn_init_state);
  copy_rnn_init_idx(result_idx_one_step0, rnn_init_idx);

  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME / 2;
       compute_time++) {
    // Use ping-pong buffer

    wrapper_rnn_fc(
        word_embedding, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM,
        rnn_bias_BRAM, fc_kernel_BRAM,
        fc_kernel + FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE, fc_bias_BRAM,
        /* input_word_idx = */result_idx_one_step0, rnn_input_state_BRAM,
        /* rnn_last_state = */rnn_state0_BRAM,
        /* rnn_output_state = */rnn_state1_BRAM,
        /* result_idx = */result_idx_one_step1);
    LDATA_T result_idx_all_idx = 2 * compute_time * BATCH_SIZE;
    result_to_DRAM(result_idx_one_step1, result_idx_all + result_idx_all_idx);

    wrapper_rnn_fc(
        word_embedding, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM,
        rnn_bias_BRAM, fc_kernel_BRAM,
        fc_kernel + FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE, fc_bias_BRAM,
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
    FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE],
    FDATA_T fc_kernel_DRAM[FC_OUTPUT_SIZE_DRAM * FC_INPUT_SIZE],
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
  FDATA_T max_output_feature_map[BATCH_SIZE];
#pragma HLS array_partition variable=fc_output_feature_map cyclic factor=32
#pragma HLS array_partition variable=max_output_feature_map complete

  // init
  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS unroll complete
    max_output_feature_map[i] = 0;
    result_idx[i] = 0;
  }

  rnn_copy_batch_word_vector(rnn_input_state_cache, word_embedding,
                             input_word_idx);
  rnn(rnn_last_state, rnn_input_state_cache,
      rnn_bias, rnn_kernel, rnn_recurrent_kernel, rnn_output_state);

  // the output state feed to fc layer
  fc_BRAM_part(rnn_output_state, fc_kernel_BRAM, fc_bias,
               max_output_feature_map, result_idx);
  fc_DRAM_part(rnn_output_state, fc_kernel_DRAM, fc_bias + FC_OUTPUT_SIZE_BRAM,
               max_output_feature_map, result_idx);
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
// depends on COMPUTE_UNROLL
#pragma HLS array_partition variable=output_state_reg factor=4

  FDATA_T kernel_reg[RNN_INPUT_SIZE];
  FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE];
#pragma HLS array_partition variable=kernel_reg cyclic factor=50
#pragma HLS array_partition variable=recurrent_kernel_reg cyclic factor=64

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
                            output_state_index, output_state);
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
#pragma HLS unroll factor=2
#pragma HLS pipeline

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
#pragma HLS UNROLL factor=2
#pragma HLS PIPELINE

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
#pragma HLS array_partition variable=local_reg cyclic factor=32 dim=2
#pragma HLS array_partition variable=local_reg cyclic factor=4 dim=1

for (LDATA_T tile_iter = 0; tile_iter < BATCH_SIZE / COMPUTE_UNROLL;
     tile_iter++) {
//#pragma HLS UNROLL factor=2

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
            last_state_reg[(tile_iter * COMPUTE_UNROLL + batch_iter)
            * RNN_STATE_SIZE + last_state_index];
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
// #pragma HLS UNROLL factor=4
#pragma HLS PIPELINE

    FDATA_T tmp = bias + output_state_reg[batch_iter];
    LDATA_T output_state_index = batch_iter * RNN_STATE_SIZE + col;

    output_state[output_state_index] =
// if in Vivado HLS, use this:
//        hls::tanh<FIXED_W_LENGTH, FIXED_I_LENGTH>(tmp);
// if in SDSoC, use this:
        FDATA_T(tanh(TOFLOAT(tmp)));
// if neither works, use this (but result wouldn't be correct)
//        tmp;
  }
}

// compute a batch of state using weights stored in BRAM
void fc_BRAM_part(FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
                  FDATA_T kernel[FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE],
                  FDATA_T bias[FC_OUTPUT_SIZE_BRAM],
                  FDATA_T max_output_feature_map[BATCH_SIZE],
                  IDATA_T max_idx[BATCH_SIZE]) {
  //  input_feature_map: BATCH_SIZE * FC_INPUT_SIZE (None * 128)
  //  bias: FC_OUTPUT_SIZE (6144)
  //  kernel:  FC_OUTPUT_SIZE * FC_INPUT_SIZE
  // maximum_output_idx: an array of idx (BATCH_SIZE, )

	FDATA_T output_feature_map_reg[BATCH_SIZE];
// depends on COMPUTE_UNROLL
#pragma HLS array_partition variable=output_feature_map_reg complete

  FDATA_T kernel_reg[FC_INPUT_SIZE];
#pragma HLS array_partition variable=kernel_reg cyclic factor=64 dim=1

  for (IDATA_T output_feature_map_index = 0;
       output_feature_map_index < FC_OUTPUT_SIZE_BRAM;
       output_feature_map_index++) {
#pragma HLS DATAFLOW

    // load
    LDATA_T kernel_offset = output_feature_map_index * FC_INPUT_SIZE;
    fc_load_kernel(kernel_reg, kernel + kernel_offset);

    // compute
    fc_compute(input_feature_map, kernel_reg, output_feature_map_reg);

    // save
    partial_argmax(output_feature_map_reg, bias[output_feature_map_index],
                   output_feature_map_index,
                   max_output_feature_map, max_idx);
  }
}

// compute a batch of state using weights stored in DRAM
void fc_DRAM_part(FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
                  FDATA_T kernel[FC_OUTPUT_SIZE_DRAM * FC_INPUT_SIZE],
                  FDATA_T bias[FC_OUTPUT_SIZE_DRAM],
                  FDATA_T max_output_feature_map[BATCH_SIZE],
                  IDATA_T max_idx[BATCH_SIZE]) {
  //  input_feature_map: BATCH_SIZE * FC_INPUT_SIZE (None * 128)
  //  bias: FC_OUTPUT_SIZE (6144)
  //  kernel:  FC_OUTPUT_SIZE * FC_INPUT_SIZE
  // maximum_output_idx: an array of idx (BATCH_SIZE, )

	FDATA_T output_feature_map_reg[BATCH_SIZE];
// depends on COMPUTE_UNROLL
#pragma HLS array_partition variable=output_feature_map_reg complete

  FDATA_T kernel_reg[FC_INPUT_SIZE];
#pragma HLS array_partition variable=kernel_reg cyclic factor=64 dim=1

  for (IDATA_T output_feature_map_index = 0;
       output_feature_map_index < FC_OUTPUT_SIZE_DRAM;
       output_feature_map_index++) {
#pragma HLS DATAFLOW

    // load
    LDATA_T kernel_offset = output_feature_map_index * FC_INPUT_SIZE;
    fc_load_kernel(kernel_reg, kernel + kernel_offset);

    // compute
    fc_compute(input_feature_map, kernel_reg, output_feature_map_reg);

    // save
    partial_argmax(output_feature_map_reg, bias[output_feature_map_index],
                   output_feature_map_index + FC_OUTPUT_SIZE_BRAM,
                   max_output_feature_map, max_idx);
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
#pragma HLS unroll factor=4
#pragma HLS pipeline

    kernel_reg[input_feature_map_index] =
        kernel_BRAM_part[input_feature_map_index];
  }
}

// compute a batch of output feature map given a single row of feature map
void fc_compute(
		FDATA_T input_feature_map_reg[BATCH_SIZE * FC_INPUT_SIZE],
		FDATA_T kernel_reg[FC_INPUT_SIZE],
		FDATA_T output_feature_map_reg[BATCH_SIZE]) {

#define FC_COMPUTE_UNROLL 8
  // initialization
  FDATA_T local_reg[FC_COMPUTE_UNROLL][FC_INPUT_SIZE];
#pragma HLS array_partition variable=local_reg cyclic factor=32 dim=2
#pragma HLS array_partition variable=local_reg cyclic factor=8 dim=1

  for (LDATA_T iter = 0; iter < BATCH_SIZE / FC_COMPUTE_UNROLL; iter++) {

    LDATA_T start_batch = iter * FC_COMPUTE_UNROLL;

    for (LDATA_T batch_idx = 0; batch_idx < FC_COMPUTE_UNROLL; batch_idx++) {
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

// given current output feature map, compare with the max value, record max idx
void partial_argmax(FDATA_T output_feature_map[BATCH_SIZE],
                    FDATA_T bias, IDATA_T output_feature_map_index,
                    FDATA_T max_output_feature_map[BATCH_SIZE],
                    IDATA_T max_idx[BATCH_SIZE]) {

  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS unroll complete
    if (output_feature_map[i] + bias > max_output_feature_map[i]) {
      max_idx[i] = output_feature_map_index;
      max_output_feature_map[i] = output_feature_map[i] + bias;
    }
  }
}

////////////////////           Utility Functions          ////////////////////

void copy_word_embedding(FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE],
                         FDATA_T word_embedding_DRAM[WORD_NUM * WORD_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < WORD_NUM * WORD_SIZE; i++) {
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

void copy_fc_kernel(
    FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE],
    FDATA_T fc_kernel_DRAM[FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE; i++) {
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
    rnn_idx_BRAM[i] = rnn_idx_DRAM[i];
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
