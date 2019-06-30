# embedding layer
python h5_to_txt.py --iname="./embedding_3_embeddings.h5" \
				--dataset="/embedding_3/embeddings:0" \
				--dim_num=2 --length1=6144 --length2=100 \
				> embedding_3_embeddings.txt
				# h5dump -d "/embedding_3/embeddings:0" embedding_3_embeddings.h5

# RNN layer
python h5_to_txt.py --iname="./simple_rnn_3_bias.h5" --dataset="/simple_rnn_3/bias:0" --dim_num=1 --length1=128 > simple_rnn_3_bias.txt
				# h5dump -d "/simple_rnn_3/bias:0" simple_rnn_3_bias.h5
python h5_to_txt.py --iname="./simple_rnn_3_kernel.h5" --dataset="/simple_rnn_3/kernel:0" --dim_num=2 --length1=100 --length2=128 > simple_rnn_3_kernel.txt
				# h5dump -d "/simple_rnn_3/kernel:0" simple_rnn_3_kernel.h5
python h5_to_txt.py --iname="./simple_rnn_3_recurrent_kernel.h5" --dataset="/simple_rnn_3/recurrent_kernel:0" --dim_num=2 --length1=128 --length2=128 > simple_rnn_3_recurrent_kernel.txt
				# h5dump -d "/simple_rnn_3/recurrent_kernel:0" simple_rnn_3_recurrent_kernel.h5

# dense layer

python h5_to_txt.py --iname="./dense_3_bias.h5" \
				--dataset="/dense_3/bias:0" \
				--dim_num=1 --length1=6144 > dense_3_bias.txt
				# h5dump -d "/dense_3/bias:0" dense_3_bias.h5

python h5_to_txt.py --iname="./dense_3_kernel.h5" \
				--dataset="/dense_3/kernel:0" \
				--dim_num=2 --length1=128 --length2=6144 \
					> dense_3_kernel.txt
				# h5dump -d "/dense_3/kernel:0" dense_3_kernel.h5
