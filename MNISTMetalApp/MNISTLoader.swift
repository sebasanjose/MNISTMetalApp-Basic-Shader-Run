import Foundation

class MNISTLoader {

    let mnistBaseURL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    let fileNames = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    let downloadDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    
    func downloadMNISTData(completion: @escaping (Bool) -> Void) {
        let dispatchGroup = DispatchGroup()
        
        for fileName in fileNames {
            let fileURL = mnistBaseURL + fileName
            let destinationURL = downloadDirectory.appendingPathComponent(fileName)
            
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                print("File already exists: \(fileName), skipping download.")
                continue
            }
            
            dispatchGroup.enter()
            print("Downloading \(fileName)...")
            
            let url = URL(string: fileURL)!
            let task = URLSession.shared.downloadTask(with: url) { tempLocalUrl, response, error in
                if let error = error {
                    print("Error downloading \(fileName): \(error)")
                    dispatchGroup.leave()
                    return
                }
                
                guard let tempLocalUrl = tempLocalUrl else {
                    print("Error: File not found at \(fileName)")
                    dispatchGroup.leave()
                    return
                }
                
                do {
                    try FileManager.default.moveItem(at: tempLocalUrl, to: destinationURL)
                    print("Successfully downloaded and saved \(fileName)")
                } catch let moveError {
                    print("Error moving \(fileName): \(moveError)")
                }
                dispatchGroup.leave()
            }
            task.resume()
        }
        
        dispatchGroup.notify(queue: .main) {
            completion(true)
        }
    }
}
