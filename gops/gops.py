NN_STATE_SIZE = 128
RNN_INPUT_SIZE = 100
BATCH_SIZE = 64
TOTAL_STEP = 5000

def matmul_ops(A, B, C):
    """
    Mat1: A x B
    Mat2: B x C
    """
    # Matrix multiplication: (A, B) * (B, C)
    # Addition: (A * (B - 1) * C)
    # Multiplications: A * (B * C)
    return (A * (B - 1) * C) + A * (B * C)

def activation_ops(A, B):
    """
    Mat: A x B
    """
    return A * B

def argmax_ops(A, B):
    """
    Mat: A x B, where A is batch size
    """
    return A * (B - 1)

if __name__ == "__main__":

    for word_num in [4096, 6144, 16192]:

        ops_per_step = 0

        # rnn
        ops_per_step += matmul_ops(BATCH_SIZE, RNN_STATE_SIZE + RNN_INPUT_SIZE, RNN_STATE_SIZE)
        ops_per_step += activation_ops(BATCH_SIZE, RNN_STATE_SIZE)

        # fc
        ops_per_step += matmul_ops(BATCH_SIZE, RNN_STATE_SIZE, word_num)
        ops_per_step += argmax_ops(BATCH_SIZE, word_num)

        print("model size: {}".format(word_num))
        print("total operations: {} ops, which is {} Gops".format(
            ops_per_step, ops_per_step / 1e9))
        print("5000 steps: {} ops".format(ops_per_step * TOTAL_STEP))


