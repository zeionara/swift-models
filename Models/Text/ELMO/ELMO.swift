// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import Checkpoints
import Datasets
import Foundation
import ModelSupport
import SwiftProtobuf
import TensorFlow

// public extension LSTMCell {
//     public init(forgetBias: Tensor<Scalar>) {
//         self.forgetBias = forgetBias
//     }
// }

func makeSequencePairs(_ sequences: [String]) -> [[String]] {
    var pairs = [[String]]()
    for i in 0..<sequences.count {
        for j in (i + 1)..<sequences.count {
            pairs.append(
                [sequences[i], sequences[j]]
            )
        }
    }
    return pairs
}

public struct BiLSTM: Layer {
    public typealias Scalar = Float

    public var forwardLSTM: LSTM<Scalar>
    public var backwardLSTM: LSTM<Scalar>

    public init(inputSize: Int, hiddenSize: Int) {
        forwardLSTM = LSTM<Scalar>(LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize))
        backwardLSTM = LSTM<Scalar>(LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let result = forwardLSTM([input])
        // let testResult = forwardLSTM(input.unstacked())
        // print(testResult.count)
        let resultBackward = backwardLSTM([input.reversed(inAxes: [0])])

        // print(resultBackward[0].hidden.shape)

        let concatenatedResult = Tensor<Scalar>(
                concatenating: [
                    result[0].hidden.transposed(permutation: [1, 0]),
                    resultBackward[0].hidden.transposed(permutation: [1, 0])
                ]
        ).transposed(permutation: [1, 0])

        return concatenatedResult
    }

    // public var tensor: Tensor<Scalar> {
    //     forwardLSTM.cell.forgetBias = forwardLSTM.cell.forgetBias
    //     return forwardLSTM.cell.forgetBias
    // }
}

let DENSE_LAYER_KEY = "dense"
let FIRST_BILSTM_CELL_KEY = "first-bilstm"


public struct ELMO: Module {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    public typealias Scalar = Float

    @noDerivative public let vocabulary: Vocabulary
    @noDerivative public let embeddingSize: Int
    @noDerivative public let tokenizer: Tokenizer
    @noDerivative public let hiddenSize: Int
    @noDerivative public let initializerStandardDeviation: Scalar

    public var tokenEmbeddings: Embedding<Scalar>
    public var firstRecurrentCell: BiLSTM
    public var secondRecurrentCell: BiLSTM
    public var dense: Dense<Scalar>

    public init(
            vocabulary: Vocabulary,
            tokenizer: Tokenizer,
            embeddingSize: Int,
            initializerStandardDeviation: Scalar = 0.02 // ,
            // hiddenSize: Int
    ) {
        self.vocabulary = vocabulary
        self.embeddingSize = embeddingSize
        self.tokenizer = tokenizer
        self.initializerStandardDeviation = initializerStandardDeviation
        self.hiddenSize = embeddingSize / 2 // hiddenSize

        // recurrentCells = Sequential {
        firstRecurrentCell = BiLSTM(inputSize: embeddingSize, hiddenSize: hiddenSize)
        secondRecurrentCell = BiLSTM(inputSize: hiddenSize * 2, hiddenSize: hiddenSize)
        // }
        
        dense = Dense<Scalar>(inputSize: hiddenSize * 2, outputSize: vocabulary.count, activation: softmax)

        tokenEmbeddings = Embedding<Scalar>(
                vocabularySize: vocabulary.count,
                embeddingSize: embeddingSize,
                embeddingsInitializer: truncatedNormalInitializer(
                        standardDeviation: Tensor<Scalar>(initializerStandardDeviation)
                )
        )
    }

    // public init(_ path: URL, _ modelName: String) {
    //     self.vocabulary = try Vocabulary(fromFile: path.appendingPathComponent("\(modelName).txt"), bert: false)
    //     self.embeddingSize = embeddingSize
    //     self.tokenizer = BERTTokenizer(vocabulary: vocabulary)
    //     self.initializerStandardDeviation = initializerStandardDeviation
    //     self.hiddenSize = hiddenSize

    //     recurrentCells = Sequential {
    //         BiLSTM(inputSize: embeddingSize, hiddenSize: hiddenSize)
    //         BiLSTM(inputSize: hiddenSize * 2, hiddenSize: hiddenSize)
    //     }
        
    //     dense = Dense<Scalar>(inputSize: hiddenSize * 2, outputSize: vocabulary.count, activation: softmax)

    //     tokenEmbeddings = Embedding<Scalar>(
    //             vocabularySize: vocabulary.count,
    //             embeddingSize: embeddingSize,
    //             embeddingsInitializer: truncatedNormalInitializer(
    //                     standardDeviation: Tensor<Scalar>(initializerStandardDeviation)
    //             )
    //     )
    // }

    // Obtains tokens' indices in the vocabulary
    public func preprocess(sequences: [String]) -> Tensor<Int32> {
        var sequences = sequences.map(tokenizer.tokenize)

        var tokens = [String]()

        for sequence in sequences {
            for token in sequence {
                tokens.append(token)
            }
        }
        let tokenIds = tokens.map {
            Int32(vocabulary.id(forToken: $0)!)
        }

        return Tensor(tokenIds)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar> {
        
        // 1. Take token embeddings
        
        let tokenEmbeddings = self.tokenEmbeddings(input)
        
        // 2. Pass embeddings through the recurrent cells
        
        let firstCellConcatenatedResult = firstRecurrentCell(tokenEmbeddings) + tokenEmbeddings // Tensor(stacking: [tokenEmbeddings, tokenEmbeddings], alongAxis: 1).reshaped(to: [-1, embeddingSize])
        let secondCellConcatenatedResult = secondRecurrentCell(firstCellConcatenatedResult)

        // 4. Convert results to next word probabilities
        // print("probs shape")
        // print(Tensor(
        //             stacking: [
        //                 tokenEmbeddings,
        //                 firstCellConcatenatedResult,
        //                 secondCellConcatenatedResult
        //             ]
        //         ).mean(alongAxes: [0]).reshaped(to: [-1, embeddingSize]).shape)
        let probs = softmax(
            dense(
                Tensor(
                    stacking: [
                        tokenEmbeddings,
                        firstCellConcatenatedResult,
                        secondCellConcatenatedResult
                    ]
                ).mean(alongAxes: [0]).reshaped(to: [-1, embeddingSize])
            )
        )
        return probs
    }

    public func test(_ sequences: [String]) {
        var logEntries = [(Float, String)]()
        for sequencePair in makeSequencePairs(sequences) {
            let nSequences = sequencePair.count
            let embs = firstRecurrentCell(
                tokenEmbeddings(
                    preprocess(
                        sequences: sequencePair
                    )
                )
            ).reshaped(to: [nSequences, -1, hiddenSize * 2]).mean(alongAxes: [1]).reshaped(to: [nSequences, -1])

            let vectors = embs.unstacked()
            // let similarity = cosineDistance(
            //     vectors[0],
            //     vectors[1]
            // )
            let similarity = (vectors[0] * vectors[1]).sum() / (sqrt((vectors[0] * vectors[0]).sum()) * sqrt((vectors[1] * vectors[1]).sum()))
            // let similarity = cosineDistance(vectors[0], vectors[1])

            logEntries.append((similarity.scalar!, "Similarity of phrases '\(sequencePair[0])' and '\(sequencePair[1])' is \(String(format: "%.3f", similarity.scalar!))"))
        }

        for (similarity, logEntry) in logEntries.sorted{$0.0 > $1.0} {
            print(logEntry)
        }
    }

    // public func save(_ path: URL, modelName: String = "elmo") throws {
    //     try FileManager.default.copyItem(at: vocabulary.path!, to: path.appendingPathComponent("\(modelName).vocabulary"))

    //     print(recurrentCells.layer1.tensor.shape)

    //     try CheckpointWriter(
    //         tensors: [
    //             DENSE_LAYER_KEY: dense.weight,
    //             FIRST_BILSTM_CELL_KEY: recurrentCells.layer1.tensor
    //         ]
    //     ).write(
    //         to: path,
    //         name: modelName
    //     )
    // }
}
