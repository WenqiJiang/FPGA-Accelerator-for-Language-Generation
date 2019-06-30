# embedding layer
python splitH5.py --iname="./train-embeddings-rnn-50-6144.h5" \
				--dataset="/model_weights/embedding_3/embedding_3/embeddings:0" \
				--oname="./embedding_3_embeddings.h5" \
				--new_dataset="/embedding_3/embeddings:0"
				# h5dump -d "/embedding_3/embeddings:0" embedding_3_embeddings.h5

# RNN layer
python splitH5.py --iname="./train-embeddings-rnn-50-6144.h5" \
				--dataset="/model_weights/simple_rnn_3/simple_rnn_3/bias:0" \
				--oname="./simple_rnn_3_bias.h5" \
				--new_dataset="/simple_rnn_3/bias:0"
				# h5dump -d "/simple_rnn_3/bias:0" simple_rnn_3_bias.h5

python splitH5.py --iname="./train-embeddings-rnn-50-6144.h5" \
				--dataset="/model_weights/simple_rnn_3/simple_rnn_3/kernel:0" \
				--oname="./simple_rnn_3_kernel.h5" \
				--new_dataset="/simple_rnn_3/kernel:0"
				# h5dump -d "/simple_rnn_3/kernel:0" simple_rnn_3_kernel.h5

python splitH5.py --iname="./train-embeddings-rnn-50-6144.h5" \
				--dataset="/model_weights/simple_rnn_3/simple_rnn_3/recurrent_kernel:0" \
				--oname="./simple_rnn_3_recurrent_kernel.h5" \
				--new_dataset="/simple_rnn_3/recurrent_kernel:0"
				# h5dump -d "/simple_rnn_3/recurrent_kernel:0" simple_rnn_3_recurrent_kernel.h5

# dense layer

python splitH5.py --iname="./train-embeddings-rnn-50-6144.h5" \
				--dataset="/model_weights/dense_3/dense_3/bias:0" \
				--oname="./dense_3_bias.h5" \
				--new_dataset="/dense_3/bias:0"
				# h5dump -d "/dense_3/bias:0" dense_3_bias.h5

python splitH5.py --iname="./train-embeddings-rnn-50-6144.h5" \
				--dataset="/model_weights/dense_3/dense_3/kernel:0" \
				--oname="./dense_3_kernel.h5" \
				--new_dataset="/dense_3/kernel:0"
				# h5dump -d "/dense_3/kernel:0" dense_3_kernel.h5
