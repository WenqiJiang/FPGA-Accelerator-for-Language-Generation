This folder is used for verification usage.

Specifically, the folder "floating point" is the language predicting model we trained, i.e. given 50 input words, predict the next word. This is not the application that we want to accelerate, but to verify the correctness of the weights, layer inplementations, etc.

The folder "hls_simulation" is used for simulate the hls_ready program, which has different interface than programmer_view. In this folder, we remove pragmas and use "malloc" for arrays in hardwares.
