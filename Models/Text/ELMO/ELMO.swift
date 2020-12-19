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

extension Array {
    public func applyMask(mask: [Int32], reversed: Bool = false) -> [Element] {
        var result = [Element]()

        for (i, item) in enumerated() {
            if mask[i] == 0 && reversed || mask[i] == 1 && !reversed {
                result.append(item)
            }
        }

        return result
    }

    public func batched(size: Int, shouldRandomize: Bool = true, filter: Optional<(Element) -> Bool> = Optional.none) -> [[Element]] {
        var filteredArray = self
        if shouldRandomize {
            filteredArray.shuffle()
        }
        if let filter_ = filter {
            filteredArray = filteredArray.filter(filter_)
        }
        let nBatches = Int((Float(filteredArray.count) / Float(size)).rounded(.up))
        return (0..<nBatches).map {i in
            let lastIndex = Swift.min((i+1)*size, filteredArray.count)
            return Array(
                filteredArray[i*size..<lastIndex]
            )
        }
    }
}

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

private func checkShapeAndSet(_ tensor: inout Tensor<Float>, to value: Tensor<Float>) {
    assert(
        tensor.shape == value.shape, "shape mismatch while setting: \(tensor.shape) to \(value.shape)"
    )
    tensor = value
}

public struct BiLSTM: Layer {
    public typealias Scalar = Float

    public var forwardLSTM: LSTM<Scalar>
    public var backwardLSTM: LSTM<Scalar>

    public init(inputSize: Int, hiddenSize: Int) {
        forwardLSTM = LSTM<Scalar>(LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize))
        backwardLSTM = LSTM<Scalar>(LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize))
    }

    public init(inputSize: Int, hiddenSize: Int, weight: Tensor<Scalar>) {
        let unstackedWeight = weight.unstacked()

        let forwardLSTMBias = unstackedWeight[0]
        let backwardLSTMBias = unstackedWeight[1]

        let unstackedWeights = Array(unstackedWeight[2..<unstackedWeight.count])
        let weightHeight = unstackedWeights.count / 2

        let forwardLSTMWeight = Tensor<Scalar>(stacking: Array(unstackedWeights[0..<weightHeight]))
        let backwardLSTMWeight = Tensor<Scalar>(stacking: Array(unstackedWeights[weightHeight..<unstackedWeights.count]))

        var forwardLSTMCell = LSTMCell<Scalar>(inputSize: inputSize, hiddenSize: hiddenSize)

        forwardLSTMCell.fusedWeight = forwardLSTMWeight
        forwardLSTMCell.fusedBias = forwardLSTMBias

        forwardLSTM = LSTM<Scalar>(forwardLSTMCell)

        var backwardLSTMCell = LSTMCell<Scalar>(inputSize: inputSize, hiddenSize: hiddenSize)

        backwardLSTMCell.fusedWeight = backwardLSTMWeight
        backwardLSTMCell.fusedBias = backwardLSTMBias

        backwardLSTM = LSTM<Scalar>(backwardLSTMCell)

        // forwardLSTM = LSTM<Scalar>(LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize))
        // backwardLSTM = LSTM<Scalar>(LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize))
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

    public var tensor: Tensor<Scalar> {
        let tensor = Tensor<Scalar>(
            stacking: [forwardLSTM.cell.fusedBias] +
            [backwardLSTM.cell.fusedBias] +
            forwardLSTM.cell.fusedWeight.unstacked() +
            backwardLSTM.cell.fusedWeight.unstacked()
        )
        // print(tensor.shape)
        // forwardLSTM.cell.fusedWeight = forwardLSTM.cell.fusedWeight
        return tensor
    }
}

let EMBEDDINGS_KEY = "embeddings"
let FIRST_BILSTM_CELL_KEY = "first-bilstm"
let SECOND_BILSTM_CELL_KEY = "second-bilstm"
let DENSE_LAYER_KEY = "dense"

public let MODEL_NAME = "elmo"

public struct ELMO: Module {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    public typealias Scalar = Float

    @noDerivative public let vocabulary: Vocabulary
    @noDerivative public let embeddingSize: Int
    @noDerivative public let tokenizer: Tokenizer
    @noDerivative public let hiddenSize: Int
    @noDerivative public let initializerStandardDeviation: Scalar?

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

    public init(_ path: URL, _ modelName: String = MODEL_NAME) throws {
        self.vocabulary = try Vocabulary(fromFile: path.appendingPathComponent("\(modelName).vocabulary"), bert: false)
        self.tokenizer = BERTTokenizer(vocabulary: vocabulary)

        let checkpointOperator = try CheckpointReader(
            checkpointLocation: path.appendingPathComponent(modelName),
            modelName: modelName
        )

        let dense_ = Dense<Scalar>(weight: Tensor<Scalar>(checkpointOperator.loadTensor(named: DENSE_LAYER_KEY)), activation: softmax)
        let hiddenSize_ = dense_.weight.shape[0] / 2

        let tokenEmbeddings_ = Embedding<Scalar>(
            embeddings: Tensor<Scalar>(checkpointOperator.loadTensor(named: EMBEDDINGS_KEY))
        )

        let embeddingSize_ = tokenEmbeddings_.embeddings.shape[1]

        // print("Initializing bilstm cells")
        // recurrentCells = Sequential {
        firstRecurrentCell = BiLSTM(inputSize: embeddingSize_, hiddenSize: hiddenSize_, weight: Tensor<Scalar>(checkpointOperator.loadTensor(named: FIRST_BILSTM_CELL_KEY)))
        secondRecurrentCell = BiLSTM(inputSize: hiddenSize_ * 2, hiddenSize: hiddenSize_, weight: Tensor<Scalar>(checkpointOperator.loadTensor(named: SECOND_BILSTM_CELL_KEY)))
        // }
        // print("Initialized bilstm cells")
        
        self.dense = dense_
        self.hiddenSize = hiddenSize_
        self.tokenEmbeddings = tokenEmbeddings_
        self.embeddingSize = embeddingSize_
        self.initializerStandardDeviation = Optional.none
    }

    // Obtains tokens' indices in the vocabulary
    public func preprocess(sequences: [String], maxSequenceLength: Int? = Optional.none, shouldOmitEOS: Bool = false) -> [ Tensor<Int32> ] {
        var sequences = sequences.filter{$0 != ""}.map(tokenizer.tokenize)

        var tokens = [String]()

        for (sequenceIndex, sequence) in sequences.enumerated() {
            for token in sequence {
                tokens.append(token)
            }
            if (sequence.count > 0) && (sequenceIndex < sequences.count - 1) && !shouldOmitEOS {
                tokens.append("[EOS]")
            }
        }
        let tokenIds = tokens.map {
            Int32(vocabulary.id(forToken: $0) ?? vocabulary.unknownTokenId)
        }

        if let maxSequenceLength_ = maxSequenceLength {
            return tokenIds.batched(size: maxSequenceLength_, shouldRandomize: false).map{ Tensor($0) }
        } else {
            return [ Tensor(tokenIds) ]
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar> {
        
        // 1. Take token embeddings
        // print(input.shape)
        // print("Obtaining embeddings")
        let tokenEmbeddings_ = self.tokenEmbeddings(input)
        // print(tokenEmbeddings_.shape)
        
        // 2. Pass embeddings through the recurrent cells
        
        let firstCellConcatenatedResult = firstRecurrentCell(tokenEmbeddings_) + tokenEmbeddings_ // Tensor(stacking: [tokenEmbeddings, tokenEmbeddings], alongAxis: 1).reshaped(to: [-1, embeddingSize])
        // print("Passed first cell")
        let secondCellConcatenatedResult = secondRecurrentCell(firstCellConcatenatedResult)
        // print("Passed second cell")

        // 4. Convert results to next word probabilities
        // print("probs shape")
        // print(Tensor(
        //             stacking: [
        //                 tokenEmbeddings,
        //                 firstCellConcatenatedResult,
        //                 secondCellConcatenatedResult
        //             ]
        //         ).mean(alongAxes: [0]).reshaped(to: [-1, embeddingSize]).shape)
        let combined = Tensor(
                    stacking: [
                        tokenEmbeddings_,
                        firstCellConcatenatedResult,
                        secondCellConcatenatedResult
                    ]
                )
        // print("Combined")
        let probs = softmax(
            dense(
                combined.mean(alongAxes: [0]).reshaped(to: [-1, embeddingSize])
            )
        )
        return probs
    }

    public func embed(_ texts: [String], shouldOmitEOS: Bool = false) -> Tensor<Float> {
        let nSequences = texts.count

        let embeddings = // Tensor(
            // stacking: texts.map{
                tokenEmbeddings(
                    preprocess(
                        sequences: texts, // [$0]
                        shouldOmitEOS: shouldOmitEOS
                    )[0]
                )
            // }
        // )

        // print("Computing first cell output")
        let firstCellOutput = firstRecurrentCell(embeddings)
        // print("Computed first cell output")
        let secondCellOutput = secondRecurrentCell(firstCellOutput + embeddings)

        // print(firstCellOutput.shape)
        // print(secondCellOutput.shape)

        return Tensor(
            stacking: [
                embeddings,
                firstCellOutput,
                secondCellOutput
            ]
        ).mean(alongAxes: [0]).reshaped(to: [nSequences, -1, hiddenSize * 2]).mean(alongAxes: [1]).reshaped(to: [nSequences, -1])
    }

    public func embed(_ texts: [String?]) -> [Tensor<Float>?] {
        return texts.map{ text in
            if let unwrappedText = text {
                return embed([unwrappedText]).flattened()
            } else {
                return Optional.none
            }
        }
    }

    public func test(_ sequences: [String]) {
        var logEntries = [(Float, String)]()
        for sequencePair in makeSequencePairs(sequences) {
            // print("a")
            let embs = embed(sequencePair, shouldOmitEOS: true)
            // print("b")
            let vectors = embs.unstacked()
            // print("c")
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

    public func save(_ path: URL, _ modelName: String = MODEL_NAME) throws {
        try FileManager.default.copyItem(at: vocabulary.path!, to: path.appendingPathComponent("\(modelName).vocabulary"))

        try CheckpointWriter(
            tensors: [
                EMBEDDINGS_KEY: tokenEmbeddings.embeddings,
                FIRST_BILSTM_CELL_KEY: firstRecurrentCell.tensor,
                SECOND_BILSTM_CELL_KEY: secondRecurrentCell.tensor,
                DENSE_LAYER_KEY: dense.weight
            ]
        ).write(
            to: path,
            name: modelName
        )
    }
}
