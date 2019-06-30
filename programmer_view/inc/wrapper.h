#pragma once

// a single time step that generate one single word
// 1 rnn layer followed by 1 fc layer
// return the index of the generated word
LDATA_T wrapper_rnn_fc(
    FDATA_T word_embeddings[WORD_NUM * WORD_SIZE],
    FDATA_T rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE], 
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE], 
    FDATA_T rnn_bias[RNN_STATE_SIZE], 
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
    FDATA_T fc_bias[FC_OUTPUT_SIZE], 
    LDATA_T input_word_idx,
    FDATA_T last_state[BATCH_SIZE * RNN_STATE_SIZE], 
    FDATA_T output_state[BATCH_SIZE * RNN_STATE_SIZE]);
