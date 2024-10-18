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

kernel void softmax(
    device float* input [[ buffer(0) ]],
    device float* output [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    const int n = 10; // Number of classes, adjust based on your model
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += exp(input[i]);
    }
    for (int i = 0; i < n; i++) {
        output[i] = exp(input[i]) / sum;
    }
}

kernel void cross_entropy_loss(
    device float* predictions [[ buffer(0) ]],
    device int* labels [[ buffer(1) ]],
    device float* loss [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    const int n = 10; // Number of classes
    int true_label = labels[gid];
    loss[gid] = -log(predictions[true_label]);
}

kernel void softmax_grad(
    device float* predictions [[ buffer(0) ]],
    device int* labels [[ buffer(1) ]],
    device float* grad_output [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    const int n = 10; // Number of classes
    int true_label = labels[gid];
    for (int i = 0; i < n; i++) {
        grad_output[i] = predictions[i] - (i == true_label ? 1.0 : 0.0);
    }
}
