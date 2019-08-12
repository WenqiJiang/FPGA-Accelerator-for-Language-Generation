# FPGA-Accelerator-for-Langauge-Generation

## Branches
### These branches are w.r.t. different model sizes discussed in our paper (link here):
small_model_4096  
medium_model_6144  
large_model_16192  
large_model_16192_opt (optimized architecture for large model) 

### These are aborted branches:
prefix_sum  
tile  


## Directories

hls_ready: the FPGA implementation using Vivado HLS

programmer_view: the software version of language generation implementation of both floating point and fixed point version.

verification: verfity the model correcteness, i.e. given 50 words, predict the next. This directory has both the floating point and fixed point version, and the later one is used to decide the fixed point length.

keras_rnn_lm: the model training files using Keras.

h5: the h5 model trained by Keras, we convert it to txt file to load into the main function.

models: the txt files of the model weights

datasets: the datasets that the verification used in verification

idx2sentence: given the generates word indexes, e.g. "14 1 321 21", generate corresponding texts, e.g. "the neural network is"

gops: compute how many operations per step will be executed in our model

## Replicate Experiments

### Environment

SDSoC 2019.1 is required. We use Xilinx ZCU102 as our platform, you may use this device or devices that support SDSoC and have better performance than ZCU102. If you use device that is not as good as ZCU102, e.g. ZCU106, the resources such as DSP and LUT may not be enough.

If you want to run the python scripts in directory ./idx2sentence, python version of 3.x is required.

### Hardware Accelerator

To compile the hardware accelerator. We first need to activate the SDSoC environment, in my case:

source /opt/Xilinx-SDSoC/Vivado/2019.1/settings64.sh

If you install SDSoC in different directory, source that file.

Then, go to the directory "hls_ready/sys". Clean the directory first to make sure there is no weird stuff in the directory, then generate the SDSoC files. 

make ultraclean
make all

After hours of compilation (4 or 5 hours for me), you get some compiled file and a directory "sd_card", that is exactly what we want. Since the accelerator needs to load some file, e.g. model weights and initial states, we need to reorganize stuffs in "sd_card" including change the directory structure and adding new files.

./organize_sdcard.sh

After that 

### Compare speed with CPU

To run the program on CPU, we only need to do few changes.

(1)
