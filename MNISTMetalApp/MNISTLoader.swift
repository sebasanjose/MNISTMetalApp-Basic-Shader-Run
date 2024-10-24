//
//  Untitled.swift
//  MNISTMetalApp
//
//  Created by Sebastian Juarez on 10/22/24.
//
import Foundation

func downloadMNIST(completion: @escaping ((trainImages: [Float], trainLabels: [UInt8], testImages: [Float], testLabels: [UInt8])?) -> Void) {
    let baseURL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    let files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    var dataset = (trainImages: [Float](), trainLabels: [UInt8](), testImages: [Float](), testLabels: [UInt8]())
    let downloadGroup = DispatchGroup()
    
    for file in files {
        print("Downloading \(baseURL + file)...")
        guard let url = URL(string: baseURL + file) else { continue }
        
        downloadGroup.enter()
        let task = URLSession.shared.dataTask(with: url) { data, _, error in
            defer { downloadGroup.leave() }
            
            if let error = error {
                print("Failed to download \(file): \(error)")
                return
            }
            
            guard let data = data else {
                print("No data for \(file)")
                return
            }
            
            // Process the data based on the file type
            if file.contains("train-images") {
                dataset.trainImages = processImagesData(data, count: 60000)
            } else if file.contains("train-labels") {
                dataset.trainLabels = processLabelsData(data, count: 60000)
            } else if file.contains("t10k-images") {
                dataset.testImages = processImagesData(data, count: 10000)
            } else if file.contains("t10k-labels") {
                dataset.testLabels = processLabelsData(data, count: 10000)
            }
            print("Finished processing \(file)")
        }
        task.resume()
    }
    
    downloadGroup.notify(queue: .main) {
        print("All files have been downloaded and processed.")
        completion(dataset)
    }
}

// Functions to process image and label data (assuming gzipped format)
func processImagesData(_ data: Data, count: Int) -> [Float] {
    // Decompress and parse the MNIST images here
    // This is a placeholder for actual processing logic
    return [Float](repeating: 0.5, count: count * 28 * 28) // Example 28x28 images
}

func processLabelsData(_ data: Data, count: Int) -> [UInt8] {
    // Decompress and parse the MNIST labels here
    // This is a placeholder for actual processing logic
    return [UInt8](repeating: 1, count: count)
}

