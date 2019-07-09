#pragma once

#define SMALL_DICT // use 4096 as dictionary size, else 6144
#ifdef SMALL_DICT
char const* EMBEDDINGS_FILE        =
    "../../../model/embedding_4096_embeddings.txt";
char const* SIMPLE_RNN_BIAS_FILE   = "../../../model/simple_rnn_4096_bias.txt";
char const* SIMPLE_RNN_KERNEL_FILE
    = "../../../model/simple_rnn_4096_kernel.txt";
char const* SIMPLE_RNN_RECURRENT_KERNEL_FILE =
    "../../../model/simple_rnn_4096_recurrent_kernel.txt";
char const* DENSE_BIAS_FILE        = "../../../model/dense_4096_bias.txt";
char const* DENSE_KERNEL_FILE      = "../../../model/dense_4096_kernel.txt";

char const* INIT_STATES_FILE       = "../../../datasets/init_state_4096.txt";
#else
char const* EMBEDDINGS_FILE        =
    "../../../model/embedding_6144_embeddings.txt";
char const* SIMPLE_RNN_BIAS_FILE   = "../../../model/simple_rnn_6144_bias.txt";
char const* SIMPLE_RNN_KERNEL_FILE = "../../../model/simple_rnn_6144_kernel.txt";
char const* SIMPLE_RNN_RECURRENT_KERNEL_FILE =
    "../../../model/simple_rnn_6144_recurrent_kernel.txt";
char const* DENSE_BIAS_FILE        = "../../../model/dense_6144_bias.txt";
char const* DENSE_KERNEL_FILE      = "../../../model/dense_6144_kernel.txt";

char const* INIT_STATES_FILE       = "../../../datasets/init_state_6144.txt";
#endif

char const* INIT_WORD_IDX_FILE     = "../../../datasets/actual_result.txt";
