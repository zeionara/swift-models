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
                    resultBackward[0].hidden.transposed(permutation: [1, 0]).reversed(inAxes: [0])
                ]
        ).transposed(permutation: [1, 0])

        return concatenatedResult
    }
}

public struct ELMO: Module {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    public typealias Scalar = Float

    @noDerivative public let vocabulary: Vocabulary
    @noDerivative public let embeddingSize: Int
    @noDerivative public let tokenizer: Tokenizer
    @noDerivative public let hiddenSize: Int
    @noDerivative public let initializerStandardDeviation: Scalar

    public var tokenEmbeddings: Embedding<Scalar>
    public var recurrentCells: Sequential<BiLSTM, BiLSTM>
    public var dense: Dense<Scalar>

    public init(
            vocabulary: Vocabulary,
            tokenizer: Tokenizer,
            embeddingSize: Int,
            initializerStandardDeviation: Scalar = 0.02,
            hiddenSize: Int
    ) {
        self.vocabulary = vocabulary
        self.embeddingSize = embeddingSize
        self.tokenizer = tokenizer
        self.initializerStandardDeviation = initializerStandardDeviation
        self.hiddenSize = hiddenSize

        recurrentCells = Sequential {
            BiLSTM(inputSize: embeddingSize, hiddenSize: hiddenSize)
            BiLSTM(inputSize: hiddenSize * 2, hiddenSize: hiddenSize)
        }
        
        dense = Dense<Scalar>(inputSize: hiddenSize * 2, outputSize: vocabulary.count, activation: softmax)

        tokenEmbeddings = Embedding<Scalar>(
                vocabularySize: vocabulary.count,
                embeddingSize: embeddingSize,
                embeddingsInitializer: truncatedNormalInitializer(
                        standardDeviation: Tensor<Scalar>(initializerStandardDeviation)
                )
        )
    }

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
        
        let concatenatedResult = recurrentCells(tokenEmbeddings)

        // 4. Convert results to next word probabilities
        let probs = dense(concatenatedResult)
        return probs
    }
}
