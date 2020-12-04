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

// Adapted from: https://gist.github.com/eaplatanios/eae9c1b4141e961c949d6f2e7d424c6f
// Untested.

import Datasets
import ModelSupport
import TensorFlow

public struct BERTLanguageModel: Module, Regularizable {
    public var bert: BERT
    public var fullyConnected: Dense<Float>
    public var output: Dense<Float>

    public var regularizationValue: TangentVector {
        TangentVector(
                bert: bert.regularizationValue,
                fullyConnected: fullyConnected.regularizationValue,
                output: output.regularizationValue
        )
    }

    public init(bert: BERT) {
        self.bert = bert
        self.fullyConnected = Dense<Float>(inputSize: bert.hiddenSize, outputSize: bert.hiddenSize)
        self.output = Dense<Float>(inputSize: bert.hiddenSize, outputSize: bert.vocabulary.count, activation: softmax)
    }

    /// Returns: logits with shape `[batchSize, classCount]`.
    @differentiable(wrt: self)
    public func callAsFunction(_ input: TextBatch) -> Tensor<Float> {
        output(fullyConnected(bert(input)))
    }
}
