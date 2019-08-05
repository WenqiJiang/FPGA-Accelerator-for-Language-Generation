#pragma once

#include "constants.h"
#include "types.h"

#define RNN_TILE_SIZE 16
#define RNN_TILE_NUM RNN_STATE_SIZE / RNN_TILE_SIZE  //4
#define FC_TILE_SIZE 4

////////////////////         TOP-LEVEL FUNCTION             ////////////////////

void wrapper_text_generation(
    FDATA_T word_embedding[WORD_NUM * WORD_SIZE],
    FDATA_T rnn_kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_bias[RNN_STATE_SIZE],
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    FDATA_T fc_bias[FC_OUTPUT_SIZE],
    FDATA_T rnn_init_state[BATCH_SIZE * RNN_STATE_SIZE],
    IDATA_T rnn_init_idx[BATCH_SIZE],
    IDATA_T result_idx_all[COMPUTE_TIME * BATCH_SIZE]);

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
    IDATA_T result_idx[BATCH_SIZE]);

////////////////////           Layer Functions              ////////////////////

void rnn_copy_batch_word_vector(
    FDATA_T rnn_input_state_BRAM[BATCH_SIZE * RNN_INPUT_SIZE],
    FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE],
    IDATA_T input_word_idx[BATCH_SIZE]);

// compute 1 time step, output state has been INITIALIZED to bias
void rnn(FDATA_T last_state[BATCH_SIZE * RNN_STATE_SIZE],
         FDATA_T input_state[BATCH_SIZE * RNN_INPUT_SIZE],
         FDATA_T bias[RNN_STATE_SIZE],
         FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
         FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
         FDATA_T output_state[BATCH_SIZE * RNN_STATE_SIZE]);

// load one row of kernel from BRAM to register
void rnn_load_kernel(FDATA_T kernel_reg[RNN_INPUT_SIZE],
                     FDATA_T kernel_part[RNN_INPUT_SIZE]);

// load one row of kernel from BRAM to register
void rnn_load_recurrent_kernel(FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE],
                               FDATA_T recurrent_kernel_part[RNN_STATE_SIZE]);

// compute a batch of input state, with a single row of kernel
void rnn_compute(
		FDATA_T input_state_reg[BATCH_SIZE * RNN_INPUT_SIZE],
		FDATA_T last_state_reg[BATCH_SIZE * RNN_STATE_SIZE],
		FDATA_T kernel_reg[RNN_INPUT_SIZE],
		FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE],
		FDATA_T output_state_reg_part[BATCH_SIZE]);

void rnn_save_output_state(FDATA_T output_state_reg[BATCH_SIZE],
                           FDATA_T bias, LDATA_T col,
                           FDATA_T output_state[BATCH_SIZE * RNN_STATE_SIZE]);

// compute a batch of state using weights stored in BRAM
void fc_BRAM_part(FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
                  FDATA_T kernel[FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE],
                  FDATA_T bias[FC_OUTPUT_SIZE_BRAM],
                  FDATA_T max_output_feature_map[BATCH_SIZE]
                  IDATA_T max_idx[BATCH_SIZE]);

// compute a batch of state using weights stored in DRAM
void fc_DRAM_part(FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
                  FDATA_T kernel[FC_OUTPUT_SIZE_DRAM * FC_INPUT_SIZE],
                  FDATA_T bias[FC_OUTPUT_SIZE_DRAM],
                  FDATA_T max_output_feature_map[BATCH_SIZE]
                  IDATA_T max_idx[BATCH_SIZE]);

// load one row of kernel from BRAM to register
void fc_load_kernel(FDATA_T kernel_reg[FC_INPUT_SIZE],
                    FDATA_T kernel_BRAM_part[FC_INPUT_SIZE]);

// compute a batch of output feature map given a single row of feature map
void fc_compute(
		FDATA_T input_feature_map_reg[BATCH_SIZE * FC_INPUT_SIZE],
		FDATA_T kernel_reg[FC_INPUT_SIZE],
		FDATA_T output_feature_map[BATCH_SIZE]);

void partial_argmax(FDATA_T output_feature_map[BATCH_SIZE],
                    FDATA_T bias, IDATA_T output_feature_map_index,
                    FDATA_T max_output_feature_map[BATCH_SIZE],
                    IDATA_T max_idx[BATCH_SIZE]);

////////////////////            Utility Functions           ////////////////////

// copy word embedding layer from DRAM to BRAM
void copy_word_embedding(FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE],
                         FDATA_T word_embedding_DRAM[WORD_NUM * WORD_SIZE]);

// copy weights from DRAM to BRAM
void copy_rnn_kernel(FDATA_T rnn_kernel_BRAM[RNN_STATE_SIZE * RNN_INPUT_SIZE],
                     FDATA_T rnn_kernel_DRAM[RNN_STATE_SIZE * RNN_INPUT_SIZE]);

// copy weights from DRAM to BRAM
void copy_rnn_recurrent_kernel(
    FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_recurrent_kernel_DRAM[RNN_STATE_SIZE * RNN_STATE_SIZE]);

// copy weights from DRAM to BRAM
void copy_rnn_bias(FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE],
                   FDATA_T rnn_bias_DRAM[RNN_STATE_SIZE]);

// copy weights from DRAM to BRAM
void copy_fc_kernel(
    FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE],
    FDATA_T fc_kernel_DRAM[FC_OUTPUT_SIZE_BRAM * FC_INPUT_SIZE]);

// copy weights from DRAM to BRAM
void copy_fc_bias(FDATA_T fc_bias_BRAM[FC_OUTPUT_SIZE],
                  FDATA_T fc_bias_DRAM[FC_OUTPUT_SIZE]);

// copy the initial states, which is generated after iterate 50 time steps,
// from DRAM to BRAM
void copy_rnn_init_state(
    FDATA_T rnn_state_BRAM[BATCH_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_init_state_DRAM[BATCH_SIZE * RNN_STATE_SIZE]);

// copy the initial "next word", which is the 51'th real word,
// from DRAM to BRAM
void copy_rnn_init_idx(IDATA_T rnn_idx_BRAM[BATCH_SIZE],
                       IDATA_T rnn_idx_DRAM[BATCH_SIZE]);

// copy the result index of a single time step from BRAM to DRAM
void result_to_DRAM(IDATA_T result_idx_BRAM[BATCH_SIZE],
                    IDATA_T result_idx_DRAM[BATCH_SIZE]);
