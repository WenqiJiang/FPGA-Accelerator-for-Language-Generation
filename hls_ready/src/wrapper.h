#pragma once

#include "constants.h"
#include "types.h"

#define FC_TILE_SIZE 64

////////////////////         TOP-LEVEL FUNCTION             ////////////////////

void wrapper_text_generation(
    FDATA_T word_embedding[WORD_NUM * WORD_SIZE],
    FDATA_T rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE],
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
    FDATA_T rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_bias[RNN_STATE_SIZE],
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    FDATA_T fc_bias[FC_OUTPUT_SIZE],
    IDATA_T input_word_idx[BATCH_SIZE],
    FDATA_T rnn_input_state_cache[BATCH_SIZE * RNN_INPUT_SIZE],
    FDATA_T rnn_last_state[BATCH_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_output_state[BATCH_SIZE * RNN_STATE_SIZE],
    IDATA_T result_idx[BATCH_SIZE]);

////////////////////           Layer Functions              ////////////////////

void rnn(FDATA_T last_state[BATCH_SIZE * RNN_STATE_SIZE],
         FDATA_T input_state[BATCH_SIZE * RNN_INPUT_SIZE],
         FDATA_T bias[RNN_STATE_SIZE],
         FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
         FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
         FDATA_T output_state[BATCH_SIZE * RNN_STATE_SIZE]);

void fc(FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
        FDATA_T bias[FC_OUTPUT_SIZE],
        FDATA_T kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
        IDATA_T maximum_output_idx[BATCH_SIZE]);

void fc_compute_tile(
    FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
    FDATA_T kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    FDATA_T bias[FC_OUTPUT_SIZE],
    LDATA_T start_feature_map_idx,
    FDATA_T output_feature_map_cache[FC_TILE_SIZE * BATCH_SIZE]);

void fc_copy_input_FM_row(
    FDATA_T input_feature_map_reg[BATCH_SIZE],
    FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
    LDATA_T input_feature_map_idx);

void fc_copy_kernel_row(
    FDATA_T kernel_tile_reg[FC_TILE_SIZE],
    FDATA_T kernel_tile[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    LDATA_T kernel_idx);

void fc_mac(FDATA_T input_feature_map_reg[BATCH_SIZE],
            FDATA_T kernel_tile_reg[FC_TILE_SIZE],
            FDATA_T output_feature_map_cache[FC_TILE_SIZE * BATCH_SIZE]);

void fc_init_cache(FDATA_T state[FC_TILE_SIZE * BATCH_SIZE]);

void fc_tile_argmax(FDATA_T output_feature_map_cache[FC_TILE_SIZE * BATCH_SIZE],
                    FDATA_T global_maximum_output[BATCH_SIZE],
                    IDATA_T global_maximum_output_idx[BATCH_SIZE],
                    LDATA_T start_idx);

void fc_add_bias(FDATA_T output_feature_map[FC_TILE_SIZE * BATCH_SIZE],
                 FDATA_T bias[FC_OUTPUT_SIZE], LDATA_T bias_start_idx);

////////////////////            Utility Functions           ////////////////////

// copy word embedding layer from DRAM to BRAM
void copy_word_embedding(FDATA_T word_embedding_BRAM[WORD_NUM * WORD_SIZE],
                         FDATA_T word_embedding_DRAM[WORD_NUM * WORD_SIZE]);

// copy weights from DRAM to BRAM
void copy_rnn_kernel(FDATA_T rnn_kernel_BRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE],
                     FDATA_T rnn_kernel_DRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE]);

// copy weights from DRAM to BRAM
void copy_rnn_recurrent_kernel(
    FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T rnn_recurrent_kernel_DRAM[RNN_STATE_SIZE * RNN_STATE_SIZE]);

// copy weights from DRAM to BRAM
void copy_rnn_bias(FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE],
                   FDATA_T rnn_bias_DRAM[RNN_STATE_SIZE]);

// copy weights from DRAM to BRAM
void copy_fc_kernel(FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
                    FDATA_T fc_kernel_DRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE]);

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

// copy a single row of word vector from word embedding layer to
// the rnn input state
void copy_word_vector(FDATA_T rnn_input_state_BRAM[RNN_STATE_SIZE],
                      FDATA_T word_embedding_BRAM[RNN_STATE_SIZE]);

// set state values to 0s
void init_rnn_state(FDATA_T state[BATCH_SIZE * RNN_STATE_SIZE]);

// set state values to 0s
void init_fc_state(FDATA_T state[BATCH_SIZE * FC_TILE_SIZE]);

// copy the result index of a single time step from BRAM to DRAM
void result_to_DRAM(IDATA_T result_idx_BRAM[BATCH_SIZE],
                    IDATA_T result_idx_DRAM[BATCH_SIZE]);
