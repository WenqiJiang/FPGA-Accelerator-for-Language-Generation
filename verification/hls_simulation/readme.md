In this folder, we have several steps to turn the hls_ready program for software simulation.

1. Remove pragmas
Comment all of them, since they cannot be preprocessed by software.

2. Use malloc in hardware functions
We may define some large array in hardwares, but software may suffer from segment fault if we don't use malloc.

3. Turn 2d arrays into 2d pointers
In hardware, we can specify 2d arrays like array[1024][2048]. But again this is too large for software, we need to use "malloc_2d_array" in "utils.cc", then "free_2d_array". When passing two dimensional arrays as function parameters, use the format of 2d pointers instead of 2d arrays, i.e. float** array instead of float array[1024][2048].
