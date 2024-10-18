//
//  Shaders.metal
//  MNISTMetalApp
//
//  Created by Sebastian Juarez on 10/18/24.
//

#include <metal_stdlib>
using namespace metal;

// This function performs matrix multiplication (used for neural network layers)
kernel void matMul(
    device float* inA [[ buffer(0) ]],
    device float* inB [[ buffer(1) ]],
    device float* outC [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    // Assuming square matrices for simplicity
    int width = 512; // Matrix width, update this based on the model
    int row = gid / width;
    int col = gid % width;

    float sum = 0.0;
    for (int i = 0; i < width; i++) {
        sum += inA[row * width + i] * inB[i * width + col];
    }

    outC[gid] = sum;
}

// This shader applies the ReLU activation function element-wise
kernel void reluActivation(
    device float* input [[ buffer(0) ]],
    device float* output [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    output[gid] = max(0.0, input[gid]);
}
