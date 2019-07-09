#pragma once

#define SMALL_DICT // use 4096 as dictionary size, else 6144
#ifdef SMALL_DICT
char const * EMBEDDINGS_FILE        =
    "../../../model/embedding_4096_embeddings.txt";
char const * SIMPLE_RNN_BIAS_FILE
    = "../../../model/simple_rnn_4096_bias.txt";
char const * SIMPLE_RNN_KERNEL_FILE
    = "../../../model/simple_rnn_4096_kernel.txt";
char const * SIMPLE_RNN_RECURRENT_KERNEL_FILE =
    "../../../model/simple_rnn_4096_recurrent_kernel.txt";
char const * DENSE_BIAS_FILE        = "../../../model/dense_4096_bias.txt";
char const * DENSE_KERNEL_FILE      = "../../../model/dense_4096_kernel.txt";
#else
char const * EMBEDDINGS_FILE        =
    "../../../model/embedding_6144_embeddings.txt";
char const * SIMPLE_RNN_BIAS_FILE   = "../../../model/simple_rnn_6144_bias.txt";
char const * SIMPLE_RNN_KERNEL_FILE
    = "../../../model/simple_rnn_6144_kernel.txt";
char const * SIMPLE_RNN_RECURRENT_KERNEL_FILE =
    "../../../model/simple_rnn_6144_recurrent_kernel.txt";
char const * DENSE_BIAS_FILE        = "../../../model/dense_6144_bias.txt";
char const * DENSE_KERNEL_FILE      = "../../../model/dense_6144_kernel.txt";
#endif

char const * ORG_SEQ_FILE           = "../../../datasets/org_seq.txt";
char const * RNN_RESULT_FILE        = "../../../datasets/rnn_result.txt";
char const * ACTUAL_RESULT_FILE     = "../../../datasets/actual_result.txt";
