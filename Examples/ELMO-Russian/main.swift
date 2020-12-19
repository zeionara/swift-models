// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import Foundation
import ModelSupport
import TensorFlow
import TextModels
import TrainingLoop
import x10_optimizers_optimizer
import ArgumentParser

let projectRoot = "\(ProcessInfo.processInfo.environment["HOME"]!)/swift-models"
let device = Device.default
let specialTokens = [String]()

func getModelsRootPath() -> URL{
    return URL(fileURLWithPath: "\(projectRoot)/assets/models")
}

func generateVocabulary(corpusName: String) throws {
    let corpusRoot = "\(projectRoot)/assets/\(corpusName)"
    let corpusFilePath = URL(fileURLWithPath: "\(corpusRoot)/sentences.txt")
    let vocabularyFilePath = URL(fileURLWithPath: "\(corpusRoot)/vocabulary.txt")

    let tokenizer = BasicTokenizer(caseSensitive: false)
    let stringSentences = try String(contentsOf: corpusFilePath)

    let tokens = specialTokens + tokenizer.tokenize(stringSentences)
    try Set(tokens).joined(separator: "\n").write(to: vocabularyFilePath, atomically: false, encoding: .utf8)
}

enum ModelError: Error {
    case unsupportedModel(message: String)
}

struct Train: ParsableCommand {

    private enum Model: String, ExpressibleByArgument {
        case elmo
    }

    @Argument(help: "Name of the model architecure in lowercase")
    private var model: Model

    @Option(name: .shortAndLong, default: "baneks", help: "Name of folder containing input data")
    private var inputPath: String

    @Option(name: .shortAndLong, default: "elmo", help: "Name of model for saving")
    private var filename: String

    @Option(name: .shortAndLong, default: 0.05, help: "Learning rate to use during model training")
    private var learningRate: Float

    @Option(name: .shortAndLong, default: 100, help: "Size of token embeddings to use")
    private var embeddingSize: Int

    @Option(name: .shortAndLong, default: 128, help: "Max number of tokens in a batch")
    private var maxSequenceLength: Int

    @Option(name: .shortAndLong, default: 10, help: "Number of train steps")
    private var nEpochs: Int

    mutating func run() throws {
        if model != .elmo {
            throw ModelError.unsupportedModel(message: "Model \(model) is not supported yet!")
        }

        try generateVocabulary(corpusName: inputPath)

        let stringSentences = try String(contentsOf: URL(fileURLWithPath: "\(projectRoot)/assets/\(inputPath)/sentences.txt")).components(separatedBy: "\n")
        let vocabulary = try Vocabulary(fromFile: URL(fileURLWithPath: "\(projectRoot)/assets/\(inputPath)/vocabulary.txt"), bert: false)
        var model = ELMO(vocabulary: vocabulary, tokenizer: BERTTokenizer(vocabulary: vocabulary), embeddingSize: embeddingSize)//, hiddenSize: 50)
        var optimizer = Adam(for: model, learningRate: learningRate)
        let sequences = ["твой анек", "больше лайков", "вчера говорили"] // ["Купил мужик", "она анекдот", "как раз"] // ["a b", "c d", "e f", "a f"]

        let preprocessedText = model.preprocess(sequences: stringSentences, maxSequenceLength: maxSequenceLength)
        let nBatches = preprocessedText.count

        for epochIndex in 1...nEpochs {
            let epochStartTime = DispatchTime.now().uptimeNanoseconds
            for (batchIndex, batch) in preprocessedText.enumerated() {
                let targetWordIndices = Tensor(batch.flattened().unstacked().map{i in i.scalar!}.dropFirst() +
                    [batchIndex == preprocessedText.count - 1 ? 0 : preprocessedText[batchIndex + 1].unstacked().first!.scalar!])

                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    let probs = model(batch)
                    let res = softmaxCrossEntropy(logits: probs, labels: targetWordIndices)
                    return res
                }

                optimizer.update(&model, along: grad)

                if (epochIndex % 100 == 0) && (batchIndex == 0) {
                    print("After \(epochIndex) epochs: ")
                    print("Loss: \(loss)")
                    model.test(sequences)
                }

                print("Completed \(batchIndex + 1)/\(nBatches) batch in \(epochIndex)/\(nEpochs) epoch")
            }
            print("Completed \(epochIndex)th epoch in \((DispatchTime.now().uptimeNanoseconds - epochStartTime) / 1_000_000_000) seconds")

            try model.save(getModelsRootPath(), filename)
        }

        let testModel = try ELMO(getModelsRootPath(), filename)

        testModel.test(sequences)
    }
}

struct OperateLanguageModel: ParsableCommand {
    static var configuration = CommandConfiguration(
            abstract: "A tool for operating language models",
            subcommands: [Train.self], // TrainExternally.self
            defaultSubcommand: Train.self
    )
}

OperateLanguageModel.main()
