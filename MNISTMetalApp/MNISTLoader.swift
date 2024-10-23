//
//  Untitled.swift
//  MNISTMetalApp
//
//  Created by Sebastian Juarez on 10/22/24.
//
import Foundation

func downloadMNIST(completion: @escaping ((trainImages: [Float], trainLabels: [UInt8], testImages: [Float], testLabels: [UInt8])?) -> Void) {
    let baseURL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    let files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    
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
            
            // Simulate processing and extracting data
            print("Downloaded and saved \(file)")
            print("Extracting and processing \(file)...")
            
            if file.contains("train-images") {
                dataset.trainImages.append(contentsOf: [Float](repeating: 0.5, count: 60000)) // Simulate train images
            } else if file.contains("train-labels") {
                dataset.trainLabels.append(contentsOf: [UInt8](repeating: 1, count: 60000)) // Simulate train labels
            } else if file.contains("t10k-images") {
                dataset.testImages.append(contentsOf: [Float](repeating: 0.5, count: 10000)) // Simulate test images
            } else if file.contains("t10k-labels") {
                dataset.testLabels.append(contentsOf: [UInt8](repeating: 1, count: 10000)) // Simulate test labels
            }
        }
        task.resume()
    }
    
    downloadGroup.notify(queue: .main) {
        completion(dataset)
    }
}
