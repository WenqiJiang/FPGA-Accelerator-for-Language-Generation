#pragma once

#define TRUE 1
#define FALSE 0

// The arguments below are NOT independent: they are defined dependently for
// computational efficiency if you need to change them, change them TOGETHER!

// Embedding layer : 6144 * 100
#define SMALL_DICT // 4096 words
#ifdef SMALL_DICT
#define WORD_NUM            4096
#else
#define WORD_NUM            6144
#endif
#define WORD_SIZE           100

// == RNN layers ===============================================================
#define RNN_STATE_SIZE      128 // state length
#define RNN_INPUT_SIZE      WORD_SIZE // input of RNN layer, e.g. word vector

// RNN layer dimension:
//        input_state: BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
//        last_state: BATCH_SIZE * RNN_STATE_SIZE (None * 128)
//        bias: RNN_STATE_SIZE (128)
//        kernel: RNN_INPUT_SIZE * RNN_STATE_SIZE (100 * 128)
//        recurrent_kernel: RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128)
//        output_state: BATCH_SIZE * RNN_STATE_SIZE (None, 128)
//
//   computation:
//        for each sample in batch:
//        output_state = input_state mul kernel +
//                       last_state mul recurrent_kernel +
//                       bias

// == Fully-Connected layers ===================================================
#define FC_INPUT_SIZE   RNN_STATE_SIZE   // same as STATE_SIZE
#define FC_OUTPUT_SIZE  WORD_NUM // say, vocabulary number

// FC layer dimension:
//      input_feature_map: BATCH_SIZE * FC_INPUT_SIZE (None * 128)
//      bias: FC_OUTPUT_SIZE (6144)
//      kernel: FC_INPUT_SIZE * FC_OUTPUT_SIZE (128 * 6144)
//      output_feature_map: BATCH_SIZE * FC_OUTPUT_SIZE (None * 6144)
//
// computation:
//      for each sample in batch:
//          output_feature_map = input_feature_map mul kernel +
//                               bias

// == Softmax layers ===========================================================
#define SM_CLASS_SIZE   WORD_NUM // how many classes

// Softmax layer dimension:
//    input_feature_map: SM_BATCH_SIZE * SM_CLASS_SIZE
//    output_feature_map: SM_BATCH_SIZE * SM_CLASS_SIZE

#define BATCH_SIZE 64
#define COMPUTE_TIME 1000
