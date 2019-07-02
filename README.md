# FPGA-Accelerator-for-Langauge-Generation

## Directories

programmer_view: the floating point language generation implementation

fixed_point_simulation: the fixed point version of the implementation above, the fixed point length and the place of decimal point is finished in the directory "verification below"

verification: verfity the model correcteness, i.e. given 50 words, predict the next. This directory has both the floating point and fixed point version, and the later one is used to decide the fixed point length.

hls_ready: the FPGA implementation using Vivado HLS

keras_rnn_lm: the model training files using Keras.

h5: the h5 model trained by Keras, we convert it to txt file to load into the main function.

models: the txt files of the model weights

datasets: the datasets that the verification used in verification
