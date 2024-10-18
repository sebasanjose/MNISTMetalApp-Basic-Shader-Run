//
//  main.swift
//  MNISTMetalApp
//
//  Created by Sebastian Juarez on 10/18/24.
//

import Metal
import MetalKit

// get the default metal device
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device")
}

// Create command queue
let commandQueue = device.makeCommandQueue()!

// Define matrix size (e.g., for a fully connected layer with 512 neurons)
let matrixWidth = 512
let inputSize = 28 * 28 // MNIST image size (flattened)

// Allocate buffers for the network layers
guard let inputBuffer = device.makeBuffer(length: inputSize * MemoryLayout<Float>.stride, options: []),
      let weightBuffer1 = device.makeBuffer(length: inputSize * matrixWidth * MemoryLayout<Float>.stride, options: []),
      let biasBuffer1 = device.makeBuffer(length: matrixWidth * MemoryLayout<Float>.stride, options: []),
      let outputBuffer1 = device.makeBuffer(length: matrixWidth * MemoryLayout<Float>.stride, options: []),
      let weightBuffer2 = device.makeBuffer(length: matrixWidth * matrixWidth * MemoryLayout<Float>.stride, options: []),
      let biasBuffer2 = device.makeBuffer(length: matrixWidth * MemoryLayout<Float>.stride, options: []),
      let outputBuffer2 = device.makeBuffer(length: matrixWidth * MemoryLayout<Float>.stride, options: []) else {
    fatalError("Failed to create Metal buffers")
}

// Initialize inputBuffer with random values
let inputPointer = inputBuffer.contents().bindMemory(to: Float.self, capacity: inputSize)
for i in 0..<inputSize {
    inputPointer[i] = Float.random(in: 0..<1)
}

// Initialize weightBuffer1 with random values
let weightPointer1 = weightBuffer1.contents().bindMemory(to: Float.self, capacity: inputSize * matrixWidth)
for i in 0..<inputSize * matrixWidth {
    weightPointer1[i] = Float.random(in: 0..<1)
}

// Print input values for verification
for i in 0..<inputSize {
    print("Input value \(i): \(inputPointer[i])")
}

// Define the Metal compute pipeline for matrix multiplication
let library = device.makeDefaultLibrary()
guard let matMulFunction = library?.makeFunction(name: "matMul") else {
    fatalError("Failed to create matMul function from library")
}
let matMulPipelineState = try! device.makeComputePipelineState(function: matMulFunction)

// Command buffer to run the Metal commands
guard let commandBuffer = commandQueue.makeCommandBuffer() else {
    fatalError("Failed to create command buffer")
}

// Create a compute encoder
guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
    fatalError("Failed to create compute encoder")
}

// Configure the first layer
computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
computeEncoder.setBuffer(weightBuffer1, offset: 0, index: 1)
computeEncoder.setBuffer(outputBuffer1, offset: 0, index: 2)

// Set the pipeline state
computeEncoder.setComputePipelineState(matMulPipelineState)

// Define grid size and thread group size (assuming square grids)
let gridSize = MTLSize(width: matrixWidth, height: 1, depth: 1)
let threadGroupSize = MTLSize(width: min(matMulPipelineState.maxTotalThreadsPerThreadgroup, matrixWidth), height: 1, depth: 1)

// Dispatch the compute shader
computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

// End encoding and commit the command buffer
computeEncoder.endEncoding()
commandBuffer.commit()

// Wait for completion
commandBuffer.waitUntilCompleted()

// Print results (for debugging purposes)
let outputPointer = outputBuffer1.contents().bindMemory(to: Float.self, capacity: matrixWidth)
for i in 0..<matrixWidth {
    print("Output neuron \(i): \(outputPointer[i])")
}

// This handles the forward pass for the first layer; similar code can be written for additional layers.
