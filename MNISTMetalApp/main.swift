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

let commandQueue = device.makeCommandQueue()


////// Matrix Multiplication Shader
//// Prepare Metal buffers for input data and weights
//let inputSize = 512 * 512 // Example size, update for your network
//let bufferA = device.makeBuffer(length: inputSize * MemoryLayout<Float>.size, options: [])
//let bufferB = device.makeBuffer(length: inputSize * MemoryLayout<Float>.size, options: [])
//let bufferC = device.makeBuffer(length: inputSize * MemoryLayout<Float>.size, options: [])
//
//// Access buffers' contents as Float arrays
//let aPointer = bufferA?.contents().bindMemory(to: Float.self, capacity: inputSize)
//let bPointer = bufferB?.contents().bindMemory(to: Float.self, capacity: inputSize)
//let cPointer = bufferC?.contents().bindMemory(to: Float.self, capacity: inputSize)
//
//// Fill buffers with data (for testing, you can randomize them)
//for i in 0..<inputSize {
//    aPointer?[i] = Float.random(in: 0..<1)
//    bPointer?[i] = Float.random(in: 0..<1)
//}
//
//// Create a Metal library and load the matrix multiplication function
//let library = device.makeDefaultLibrary()
//let matMulFunction = library?.makeFunction(name: "matMul")
//
//// Create a pipeline state object
//let pipelineState = try! device.makeComputePipelineState(function: matMulFunction!)
//
//// Create a command buffer and encoder to run the shader
//let commandBuffer = commandQueue?.makeCommandBuffer()
//let computeEncoder = commandBuffer?.makeComputeCommandEncoder()
//
//computeEncoder?.setComputePipelineState(pipelineState)
//computeEncoder?.setBuffer(bufferA, offset: 0, index: 0)
//computeEncoder?.setBuffer(bufferB, offset: 0, index: 1)
//computeEncoder?.setBuffer(bufferC, offset: 0, index: 2)
//
//// Determine thread group sizes
//let gridSize = MTLSize(width: inputSize, height: 1, depth: 1)
//let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)
//
//computeEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
//computeEncoder?.endEncoding()
//
//// Commit the command buffer and wait for execution
//commandBuffer?.commit()
//commandBuffer?.waitUntilCompleted()
//
//// Read the output data (result of matrix multiplication)
//for i in 0..<inputSize {
//    print(cPointer?[i] ?? 0)
//}

/// RELU Shader
// Assume inputSize is the number of elements in the input tensor
let inputSize = 512 * 512 // Adjust this based on your model

// Create Metal buffers for input and output
let inputBuffer = device.makeBuffer(length: inputSize * MemoryLayout<Float>.size, options: [])
let outputBuffer = device.makeBuffer(length: inputSize * MemoryLayout<Float>.size, options: [])

// Fill the input buffer with random data for testing
let inputPointer = inputBuffer?.contents().bindMemory(to: Float.self, capacity: inputSize)
for i in 0..<inputSize {
    inputPointer?[i] = Float.random(in: -1.0..<1.0)  // Some values will be negative to test ReLU
}

// Create a Metal library and load the ReLU function
let library = device.makeDefaultLibrary()
let reluFunction = library?.makeFunction(name: "reluActivation")
let reluPipelineState = try! device.makeComputePipelineState(function: reluFunction!)

// Create a command buffer and encoder to run the ReLU shader
let reluCommandBuffer = commandQueue?.makeCommandBuffer()
let reluComputeEncoder = reluCommandBuffer?.makeComputeCommandEncoder()

// Set the pipeline state and buffers
reluComputeEncoder?.setComputePipelineState(reluPipelineState)
reluComputeEncoder?.setBuffer(inputBuffer, offset: 0, index: 0)
reluComputeEncoder?.setBuffer(outputBuffer, offset: 0, index: 1)

// Dispatch threads (same as matrix size)
let reluGridSize = MTLSize(width: inputSize, height: 1, depth: 1)
let reluThreadGroupSize = MTLSize(width: 32, height: 1, depth: 1)

// Dispatch threads for ReLU
reluComputeEncoder?.dispatchThreads(reluGridSize, threadsPerThreadgroup: reluThreadGroupSize)
reluComputeEncoder?.endEncoding()

// Commit the command buffer and wait for execution
reluCommandBuffer?.commit()
reluCommandBuffer?.waitUntilCompleted()

// Read the output data
let outputPointer = outputBuffer?.contents().bindMemory(to: Float.self, capacity: inputSize)
for i in 0..<inputSize {
    print("Input: \(inputPointer?[i] ?? 0), Output (ReLU): \(outputPointer?[i] ?? 0)")
}


print("Hello, World!")

