#pragma once

char const* EMBEDDINGS_FILE        =
    "../../model/embedding_1_embeddings.txt";
char const* SIMPLE_RNN_BIAS_FILE   = "../../model/simple_rnn_1_bias.txt";
char const* SIMPLE_RNN_KERNEL_FILE
    = "../../model/simple_rnn_1_kernel.txt";
char const* SIMPLE_RNN_RECURRENT_KERNEL_FILE =
    "../../model/simple_rnn_1_recurrent_kernel.txt";
char const* DENSE_BIAS_FILE        = "../../model/dense_1_bias.txt";
char const* DENSE_KERNEL_FILE      = "../../model/dense_1_kernel.txt";

char const* INIT_STATES_FILE       = "../../datasets/init_state_16192.txt";

char const* INIT_WORD_IDX_FILE     = "../../datasets/actual_result.txt";
// if you are using fixed point(16,7), use this file as correct result
char const* CORRECT_RESULT_FILE    =
    "../correct_results/generation_16192_1000_fixed.txt";
// if you are using floating point 32, use this
// char const* CORRECT_RESULT_FILE    =
    // "../correct_results/generation_16192_1000_float.txt";
