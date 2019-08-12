# FPGA-Accelerator-for-Langauge-Generation

The accelerator for small model achieves 18.91 Gops, which is 30x speed up comparing to CPU. Different model sizes lead to significant performance difference. Details are discussed in our paper.

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

If you are using boards other than ZCU102, you need to change the device name in "Makefile".

After hours of compilation (4 or 5 hours for me), you will get some compiled file and a directory "sd_card" which is exactly what we want. Since the accelerator needs to load some file, e.g. model weights and initial states, we need to reorganize stuffs in "sd_card" including change the directory structure and adding new files.

./organize_sdcard.sh

After that copy the reorganized directory, e.g. "LG_HW_6144" to /mnt of the sd card of the board, e.g. ZCU102.

Copy BOOT.BIN, image.ub in "LG_HW_6144" to /mnt in sd card.

Insert the card back to the board and reboot it. Notice that you need to copy the two files above to /mnt and reboot whenever you are going to run a different accelerator.

After rebooting, cd /LG_HW_6144/run/run, run the accelerator:

./c-rnn.elf

The clock cycle, run time and correctness will show up.


### Compare speed with CPU

#### To run the program on CPU, we only need to do few changes.

(1) hls_ready/src/types.h

around line 13 and line 15:

#define FDATA_T ap_fixed<FIXED_W_LENGTH, FIXED_I_LENGTH>
#define TOFLOAT(a) a.to_float()

change them to:

#define FDATA_T float
#define TOFLOAT(a) a

This is because the floating point version is faster on CPU: the fixed point version operations will call the library which is super slow.

(2) hls_ready/src/config.h

Since we use the floating point, the correct result is different than the fixed point version.

at the bottom of the file, change

char const* CORRECT_RESULT_FILE    =
    "../correct_results/generation_6144_1000_fixed.txt";
// if you are using floating point 32, use this
// char const* CORRECT_RESULT_FILE    =
    // "../correct_results/generation_6144_1000_float.txt";

to

// char const* CORRECT_RESULT_FILE    =
//     "../correct_results/generation_6144_1000_fixed.txt";
// if you are using floating point 32, use this
char const* CORRECT_RESULT_FILE    =
    "../correct_results/generation_6144_1000_float.txt";

(3) hls_ready/sys/Makefile

comment this line (around line 47) to prevent SDSoC generate an hardware accelerator:

SDS_FLAGS += -sds-hw $(TOP_HW_FUNCTION) $(TOP_HW_FUNCTION_SOURCE) -sds-end

(4) hls_ready/src/constants.h

You can choose to do this step or not, depends on how much time would you spend on running the program. To run 5000 steps as default, the CPU may need 10 ~ 40 minutes depends on the model size.

Around line 57, you can change the steps numbers to e.g. 500 to avoid long waiting time.

#define COMPUTE_TIME 5000

#### rest of the steps are same as the HW steps

To compile the hardware accelerator. We first need to activate the SDSoC environment, in my case:

source /opt/Xilinx-SDSoC/Vivado/2019.1/settings64.sh

If you install SDSoC in different directory, source that file.

Then, go to the directory "hls_ready/sys". Clean the directory first to make sure there is no weird stuff in the directory, then generate the SDSoC files.

make ultraclean
make all

If you are using boards other than ZCU102, you need to change the device name in "Makefile".

After hours of compilation (4 or 5 hours for me), you will get some compiled file and a directory "sd_card" which is exactly what we want. Since the accelerator needs to load some file, e.g. model weights and initial states, we need to reorganize stuffs in "sd_card" including change the directory structure and adding new files.

./organize_sdcard.sh

After that copy the reorganized directory, e.g. "LG_HW_6144" to /mnt of the sd card of the board, e.g. ZCU102. Notice that here we may change the directory name to "LG_SW_6144" indicates it's a software version.

Copy BOOT.BIN, image.ub in "LG_SW_6144" to /mnt in sd card.

Insert the card back to the board and reboot it. Notice that you need to copy the two files above to /mnt and reboot whenever you are going to run a different accelerator.

After rebooting, cd /LG_SW_6144/run/run, run the accelerator:

./c-rnn.elf

The clock cycle, run time and correctness will show up.

