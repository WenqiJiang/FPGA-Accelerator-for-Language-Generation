#pragma once

#include "constants.h"
#include "types.h"
// a single time step that generate one single word
// 1 rnn layer followed by 1 fc layer
// return the index of the generated word
void wrapper_rnn_fc(
    FDATA_T word_embeddings[WORD_NUM * WORD_SIZE],
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
    IDATA_T result_idx[BATCH_SIZE]);
