//
//  main.swift
//  MNISTMetalApp
//
//  Created by Sebastian Juarez on 10/18/24.
//

import Metal
import MetalKit

import Foundation

print("Starting MNIST dataset download and processing...")

let semaphore = DispatchSemaphore(value: 0)  // Create a semaphore to wait for completion

downloadMNIST { dataset in
    if let dataset = dataset {
        print("MNIST dataset downloaded and processed.")
        
        let trainDataPoints = dataset.trainImages.count
        let validDataPoints = dataset.testImages.count
        
        print("Train dataset:")
        print("Number of datapoints: \(trainDataPoints)")
        print("Root location: ./data/")
        print("Split: Train")
        
        print("Validation dataset:")
        print("Number of datapoints: \(validDataPoints)")
        print("Root location: ./data/")
        print("Split: Test")
    } else {
        print("Failed to load MNIST dataset.")
    }
    
    print("Training step completed.")
    semaphore.signal()  // Signal that the download is completed
}

// Wait for the downloads to finish before exiting
semaphore.wait()
print("Program finished.")

// Get the default metal device
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device")
}

// Create command queue
let commandQueue = device.makeCommandQueue()!

// Define matrix size (e.g., for a fully connected layer with 512 neurons)
let matrixWidth = 512
let inputSize = 28 * 28 // MNIST image size (flattened)
let outputSize = 10 // Number of classes for MNIST

// Allocate buffers for the network layers
guard let inputBuffer = device.makeBuffer(length: inputSize * MemoryLayout<Float>.stride, options: []),
      let weightBuffer1 = device.makeBuffer(length: inputSize * matrixWidth * MemoryLayout<Float>.stride, options: []),
      let biasBuffer1 = device.makeBuffer(length: matrixWidth * MemoryLayout<Float>.stride, options: []),
      let outputBuffer1 = device.makeBuffer(length: matrixWidth * MemoryLayout<Float>.stride, options: []),
      let weightBuffer2 = device.makeBuffer(length: matrixWidth * outputSize * MemoryLayout<Float>.stride, options: []),
      let biasBuffer2 = device.makeBuffer(length: outputSize * MemoryLayout<Float>.stride, options: []),
      let outputBuffer2 = device.makeBuffer(length: outputSize * MemoryLayout<Float>.stride, options: []),
      let targetBuffer = device.makeBuffer(length: outputSize * MemoryLayout<Float>.stride, options: []),
      let lossBuffer = device.makeBuffer(length: outputSize * MemoryLayout<Float>.stride, options: []),
      let gradientBuffer = device.makeBuffer(length: outputSize * MemoryLayout<Float>.stride, options: []) else {
    fatalError("Failed to create Metal buffers")
}


// Create command buffer
let commandBuffer = commandQueue.makeCommandBuffer()!


// Load the Metal shader functions (assuming we have a Metal library)
let library = try device.makeDefaultLibrary(bundle: .main)
let matMulFunction = library.makeFunction(name: "matMul")!
let reluActivationFunction = library.makeFunction(name: "reluActivation")!
let computeLossFunction = library.makeFunction(name: "computeLoss")!
let computeOutputGradientFunction = library.makeFunction(name: "computeOutputGradient")!
let updateWeightsFunction = library.makeFunction(name: "updateWeights")!

// Create compute pipeline states
let matMulPipeline = try device.makeComputePipelineState(function: matMulFunction)
let reluActivationPipeline = try device.makeComputePipelineState(function: reluActivationFunction)
let computeLossPipeline = try device.makeComputePipelineState(function: computeLossFunction)
let computeOutputGradientPipeline = try device.makeComputePipelineState(function: computeOutputGradientFunction)
let updateWeightsPipeline = try device.makeComputePipelineState(function: updateWeightsFunction)


// Configure threadgroup size for 1D grid (only width)
let threadGroupSize = MTLSize(width: 512, height: 1, depth: 1)  // 1D threads per group (match matrix width)
let threadGroupCount = MTLSize(width: (matrixWidth + 511) / 512, height: 1, depth: 1) // Ensure we cover the matrix

// Forward pass: Layer 1 (input to hidden)
let matMulEncoder1 = commandBuffer.makeComputeCommandEncoder()!
matMulEncoder1.setComputePipelineState(matMulPipeline)
matMulEncoder1.setBuffer(inputBuffer, offset: 0, index: 0) // Input
matMulEncoder1.setBuffer(weightBuffer1, offset: 0, index: 1) // Weights
matMulEncoder1.setBuffer(outputBuffer1, offset: 0, index: 2) // Output
matMulEncoder1.dispatchThreads(threadGroupCount, threadsPerThreadgroup: threadGroupSize) // Dispatch with the corrected 1D configuration
matMulEncoder1.endEncoding()

// Activation: ReLU Layer 1
let reluEncoder1 = commandBuffer.makeComputeCommandEncoder()!
reluEncoder1.setComputePipelineState(reluActivationPipeline)
reluEncoder1.setBuffer(outputBuffer1, offset: 0, index: 0) // Input (post-matMul)
reluEncoder1.setBuffer(outputBuffer1, offset: 0, index: 1) // Output (overwritten)
reluEncoder1.dispatchThreads(threadGroupCount, threadsPerThreadgroup: threadGroupSize) // Correct dispatch for ReLU with 1D
reluEncoder1.endEncoding()



// Forward pass: Layer 2 (hidden to output)
let matMulEncoder2 = commandBuffer.makeComputeCommandEncoder()!
matMulEncoder2.setComputePipelineState(matMulPipeline)
matMulEncoder2.setBuffer(outputBuffer1, offset: 0, index: 0)    // Input (from Layer 1)
matMulEncoder2.setBuffer(weightBuffer2, offset: 0, index: 1)    // Weights
matMulEncoder2.setBuffer(outputBuffer2, offset: 0, index: 2)    // Output
matMulEncoder2.dispatchThreads(threadGroupCount, threadsPerThreadgroup: threadGroupSize)
matMulEncoder2.endEncoding()

// Activation: ReLU Layer 2
let reluEncoder2 = commandBuffer.makeComputeCommandEncoder()!
reluEncoder2.setComputePipelineState(reluActivationPipeline)
reluEncoder2.setBuffer(outputBuffer2, offset: 0, index: 0) // Input (post-matMul)
reluEncoder2.setBuffer(outputBuffer2, offset: 0, index: 1) // Output (overwritten)
reluEncoder2.dispatchThreads(threadGroupCount, threadsPerThreadgroup: threadGroupSize)
reluEncoder2.endEncoding()

// Compute Loss
let lossEncoder = commandBuffer.makeComputeCommandEncoder()!
lossEncoder.setComputePipelineState(computeLossPipeline)
lossEncoder.setBuffer(outputBuffer2, offset: 0, index: 0) // Predictions
lossEncoder.setBuffer(targetBuffer, offset: 0, index: 1)  // Targets (ground truth)
lossEncoder.setBuffer(lossBuffer, offset: 0, index: 2)    // Loss output
lossEncoder.dispatchThreads(threadGroupCount, threadsPerThreadgroup: threadGroupSize)
lossEncoder.endEncoding()

// Compute Output Gradient (for backpropagation)
let gradientEncoder = commandBuffer.makeComputeCommandEncoder()!
gradientEncoder.setComputePipelineState(computeOutputGradientPipeline)
gradientEncoder.setBuffer(outputBuffer2, offset: 0, index: 0) // Predictions
gradientEncoder.setBuffer(targetBuffer, offset: 0, index: 1)  // Targets
gradientEncoder.setBuffer(gradientBuffer, offset: 0, index: 2) // Gradient output
gradientEncoder.dispatchThreads(threadGroupCount, threadsPerThreadgroup: threadGroupSize)
gradientEncoder.endEncoding()

// Create a buffer for the learning rate
var learningRate: Float = 0.01
let learningRateBuffer = device.makeBuffer(bytes: &learningRate, length: MemoryLayout<Float>.stride, options: [])

// Update weights (Layer 2)
let updateWeightsEncoder2 = commandBuffer.makeComputeCommandEncoder()!
updateWeightsEncoder2.setComputePipelineState(updateWeightsPipeline)
updateWeightsEncoder2.setBuffer(weightBuffer2, offset: 0, index: 0)  // Weights
updateWeightsEncoder2.setBuffer(gradientBuffer, offset: 0, index: 1) // Gradients
updateWeightsEncoder2.setBuffer(learningRateBuffer, offset: 0, index: 2) // Learning rate buffer
updateWeightsEncoder2.dispatchThreads(threadGroupCount, threadsPerThreadgroup: threadGroupSize)
updateWeightsEncoder2.endEncoding()


// Update Weights (Layer 1)
// Similar process to update layer 1 weights using backpropagation through the hidden layer

// Commit the command buffer and wait for completion
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

print("Training step completed.")

