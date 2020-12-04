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

public struct ELMO: Module {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    public typealias Scalar = Float

//    @noDerivative public let variant: Variant
    @noDerivative public let vocabulary: Vocabulary
    @noDerivative public let embeddingSize: Int
    @noDerivative public let tokenizer: Tokenizer
    @noDerivative public let hiddenSize: Int
//    @noDerivative public let caseSensitive: Bool
//    @noDerivative public let hiddenSize: Int
//    @noDerivative public let hiddenLayerCount: Int
//    @noDerivative public let attentionHeadCount: Int
//    @noDerivative public let intermediateSize: Int
//    @noDerivative public let intermediateActivation: Activation<Scalar>
//    @noDerivative public let hiddenDropoutProbability: Scalar
//    @noDerivative public let attentionDropoutProbability: Scalar
//    @noDerivative public let maxSequenceLength: Int
//    @noDerivative public let typeVocabularySize: Int
    @noDerivative public let initializerStandardDeviation: Scalar

    public var tokenEmbeddings: Embedding<Scalar>
    public var forwardLSTM: LSTM<Scalar>
    public var backwardLSTM: LSTM<Scalar>
    public var dense: Dense<Scalar>
//    public var tokenTypeEmbedding: Embedding<Scalar>
//    public var positionEmbedding: Embedding<Scalar>
//    public var embeddingLayerNorm: LayerNorm<Scalar>
//    @noDerivative public var embeddingDropout: Dropout<Scalar>
//    public var embeddingProjection: [Dense<Scalar>]
//    public var encoder: TransformerEncoder

    public init(
            vocabulary: Vocabulary,
            tokenizer: Tokenizer,
            embeddingSize: Int,
            initializerStandardDeviation: Scalar = 0.02,
            hiddenSize: Int
    ) {
//        self.variant = variant
        self.vocabulary = vocabulary
        self.embeddingSize = embeddingSize
        self.tokenizer = tokenizer
//        self.caseSensitive = caseSensitive
//        self.hiddenSize = hiddenSize
//        self.hiddenLayerCount = hiddenLayerCount
//        self.attentionHeadCount = attentionHeadCount
//        self.intermediateSize = intermediateSize
//        self.intermediateActivation = intermediateActivation
//        self.hiddenDropoutProbability = hiddenDropoutProbability
//        self.attentionDropoutProbability = attentionDropoutProbability
//        self.maxSequenceLength = maxSequenceLength
//        self.typeVocabularySize = typeVocabularySize
        self.initializerStandardDeviation = initializerStandardDeviation
        forwardLSTM = LSTM<Scalar>(LSTMCell(inputSize: embeddingSize, hiddenSize: hiddenSize))
        backwardLSTM = LSTM<Scalar>(LSTMCell(inputSize: embeddingSize, hiddenSize: hiddenSize))
        dense = Dense<Scalar>(inputSize: hiddenSize * 2, outputSize: vocabulary.count, activation: softmax)
        self.hiddenSize = hiddenSize

//        if case let .albert(_, hiddenGroupCount) = variant {
//            precondition(
//                    hiddenGroupCount <= hiddenLayerCount,
//                    "The number of hidden groups must be smaller than the number of hidden layers.")
//        }
//
//        let embeddingSize: Int = {
//            switch variant {
//            case .bert, .roberta, .electra: return hiddenSize
//            case let .albert(embeddingSize, _): return embeddingSize
//            }
//        }()

        tokenEmbeddings = Embedding<Scalar>(
                vocabularySize: vocabulary.count,
                embeddingSize: embeddingSize,
                embeddingsInitializer: truncatedNormalInitializer(
                        standardDeviation: Tensor<Scalar>(initializerStandardDeviation)
                )
        )


        // The token type vocabulary will always be small and so we use the one-hot approach here
        // as it is always faster for small vocabularies.
//        tokenTypeEmbedding = Embedding<Scalar>(
//                vocabularySize: typeVocabularySize,
//                embeddingSize: embeddingSize,
//                embeddingsInitializer: truncatedNormalInitializer(
//                        standardDeviation: Tensor<Scalar>(initializerStandardDeviation)))

        // Since the position embeddings table is a learned variable, we create it using a (long)
        // sequence length, `maxSequenceLength`. The actual sequence length might be shorter than
        // this, for faster training of tasks that do not have long sequences. So,
        // `positionEmbedding` effectively contains an embedding table for positions
        // [0, 1, 2, ..., maxPositionEmbeddings - 1], and the current sequence may have positions
        // [0, 1, 2, ..., sequenceLength - 1], so we can just perform a slice.
//        let positionPaddingIndex = { () -> Int in
//            switch variant {
//            case .bert, .albert, .electra: return 0
//            case .roberta: return 2
//            }
//        }()
//        positionEmbedding = Embedding(
//                vocabularySize: positionPaddingIndex + maxSequenceLength,
//                embeddingSize: embeddingSize,
//                embeddingsInitializer: truncatedNormalInitializer(
//                        standardDeviation: Tensor(initializerStandardDeviation)))
//
//        embeddingLayerNorm = LayerNorm<Scalar>(
//                featureCount: hiddenSize,
//                axis: -1)
//        // TODO: Make dropout generic over the probability type.
//        embeddingDropout = Dropout(probability: Double(hiddenDropoutProbability))
//
//        // Add an embedding projection layer if using the ALBERT variant.
//        embeddingProjection = {
//            switch variant {
//            case .bert, .roberta, .electra: return []
//            case let .albert(embeddingSize, _):
//                // TODO: [AD] Change to optional once supported.
//                return [Dense<Scalar>(
//                        inputSize: embeddingSize,
//                        outputSize: hiddenSize,
//                        weightInitializer: truncatedNormalInitializer(
//                                standardDeviation: Tensor(initializerStandardDeviation)))]
//            }
//        }()
//
//        switch variant {
//        case .bert, .roberta, .electra:
//            encoder = TransformerEncoder(
//                    hiddenSize: hiddenSize,
//                    layerCount: hiddenLayerCount,
//                    attentionHeadCount: attentionHeadCount,
//                    attentionQueryActivation: { $0 },
//                    attentionKeyActivation: { $0 },
//                    attentionValueActivation: { $0 },
//                    intermediateSize: intermediateSize,
//                    intermediateActivation: intermediateActivation,
//                    hiddenDropoutProbability: hiddenDropoutProbability,
//                    attentionDropoutProbability: attentionDropoutProbability
//            )
//        case let .albert(_, hiddenGroupCount):
//            encoder = TransformerEncoder(
//                    hiddenSize: hiddenSize,
//                    layerCount: hiddenLayerCount,
//                    attentionHeadCount: attentionHeadCount,
//                    attentionQueryActivation: { $0 },
//                    attentionKeyActivation: { $0 },
//                    attentionValueActivation: { $0 },
//                    intermediateSize: intermediateSize,
//                    intermediateActivation: intermediateActivation,
//                    hiddenDropoutProbability: hiddenDropoutProbability,
//                    attentionDropoutProbability: attentionDropoutProbability
//            )
//        }
    }

/// Preprocesses an array of text sequences and prepares them for processing with BERT.
/// Preprocessing mainly consists of tokenization.
///
/// - Parameters:
///   - sequences: Text sequences (not tokenized).
///   - maxSequenceLength: Maximum sequence length supported by the text perception module.
///     This is mainly used for padding the preprocessed sequences. If not provided, it
///     defaults to this model's maximum supported sequence length.
///   - tokenizer: Tokenizer to use while preprocessing.
///
/// - Returns: Text batch that can be processed by BERT.
    public func preprocess(sequences: [String]) -> Tensor<Int32> {
//        let maxSequenceLength = maxSequenceLength ?? self.maxSequenceLength
//        let nMaskedTokens = nMaskedTokens ?? 0
        var sequences = sequences.map(tokenizer.tokenize)

//        func sampleTokenIndex(sampledIndices: inout [Int]) -> [Int] {
//            var maskedTokenIndex: Int = -1
//            while ((maskedTokenIndex < 0 || tokenIds[maskedTokenIndex] < 4) || sampledIndices.contains(maskedTokenIndex)) {
//                maskedTokenIndex = (0..<tokenIds.count).randomElement()!
//            }
//            sampledIndices.append(maskedTokenIndex)
//            return sampledIndices
//        }

        // Truncate the sequences based on the maximum allowed sequence length, while accounting
        // for the '[CLS]' token and for `sequences.count` '[SEP]' tokens. The following is a
        // simple heuristic which will truncate the longer sequence one token at a time. This makes
        // more sense than truncating an equal percent of tokens from each sequence, since if one
        // sequence is very short then each token that is truncated likely contains more
        // information than respective tokens in longer sequences.
//        var totalLength = sequences.map {
//            $0.count
//        }.reduce(0, +)
//        let totalLengthLimit = { () -> Int in
//            switch variant {
//            case .bert, .albert, .electra: return maxSequenceLength - 1 - sequences.count
//            case .roberta: return maxSequenceLength - 1 - 2 * sequences.count
//            }
//        }()
//        while totalLength >= totalLengthLimit {
//            let maxIndex = sequences.enumerated().max(by: { $0.1.count < $1.1.count })!.0
//            sequences[maxIndex] = [String](sequences[maxIndex].dropLast())
//            totalLength = sequences.map {
//                $0.count
//            }.reduce(0, +)
//        }

        // The convention in BERT is:
        //   (a) For sequence pairs:
        //       tokens:       [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        //       tokenTypeIds: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        //   (b) For single sequences:
        //       tokens:       [CLS] the dog is hairy . [SEP]
        //       tokenTypeIds: 0     0   0   0  0     0 0
        // where "tokenTypeIds" are used to indicate whether this is the first sequence or the
        // second sequence. The embedding vectors for `tokenTypeId = 0` and `tokenTypeId = 1` were
        // learned during pre-training and are added to the WordPiece embedding vector (and
        // position vector). This is not *strictly* necessary since the [SEP] token unambiguously
        // separates the sequences. However, it makes it easier for the model to learn the concept
        // of sequences.
        //
        // For classification tasks, the first vector (corresponding to `[CLS]`) is used as the
        // "sentence embedding". Note that this only makes sense because the entire model is
        // fine-tuned under this assumption.
        var tokens = [String]()
//        var tokenTypeIds = [Int32(0)]
        for sequence in sequences {
            for token in sequence {
                tokens.append(token)
//                tokenTypeIds.append(Int32(sequenceId))
            }
//            tokens.append("[SEP]")
//            tokenTypeIds.append(Int32(sequenceId))
//            if case .roberta = variant, sequenceId < sequences.count - 1 {
//                tokens.append("[SEP]")
//                tokenTypeIds.append(Int32(sequenceId))
//            }
        }
        let tokenIds = tokens.map {
            Int32(vocabulary.id(forToken: $0)!)
        }

//        var maskedTokenIndices = [Int]()
//        (0..<nMaskedTokens).map { _ -> [Int] in
//            sampleTokenIndex(sampledIndices: &maskedTokenIndices)
//        }

        // The mask is set to `true` for real tokens and `false` for special tokens. This is so
        // that only real tokens are attended to.
//        let mask = tokenIds.enumerated().map { (i, tokenId) -> Int32 in
//            tokenId < 4 || maskedTokenIndices.contains(i) ? 0 : 1
//        }
//
//        let languageModelMask = (0..<tokenIds.count).map { i -> Int32 in
//            maskedTokenIndices.contains(i) ? 0 : 1
//        }

        return Tensor(tokenIds)
//        return TextBatch(
//                tokenIds: Tensor(tokenIds).expandingShape(at: 0),
//                tokenTypeIds: Tensor(tokenTypeIds).expandingShape(at: 0),
//                mask: Tensor(mask).expandingShape(at: 0),
//                languageModelMask: Tensor(languageModelMask).expandingShape(at: 0)
//        )
    }

    @differentiable(wrt: self)
    public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar> {
//        let sequenceLength = input.shape[0]
//        let variant = withoutDerivative(at: self.variant)

        // Compute the input embeddings and apply layer normalization and dropout on them.
        let tokenEmbeddings = self.tokenEmbeddings(input)
        let result = forwardLSTM([tokenEmbeddings])
        let resultBackward = backwardLSTM([tokenEmbeddings.reversed(inAxes: [0])])
//        print(result[0].hidden.transposed(permutation: [1, 0]).shape)
//        print(resultBackward[0].hidden.shape)
        let concatenatedResult = Tensor<Scalar>(
                concatenating: [
                    result[0].hidden.transposed(permutation: [1, 0]),
                    resultBackward[0].hidden.transposed(permutation: [1, 0])
                ]
        ).transposed(permutation: [1, 0])
        let probs = dense(concatenatedResult)
//        print(probs.shape)

//        for token in tokenEmbeddings.unstacked() {
//            print(
//                    forwardLSTM(
//RNNCellInput(input: token, state: forwardLSTM.zeroState(for: token))
//                    )
//            )
//        }
////        let tokenTypeEmbeddings = tokenTypeEmbedding(input.tokenTypeIds)
//        let positionPaddingIndex: Int
//        switch variant {
//        case .bert, .albert, .electra: positionPaddingIndex = 0
//        case .roberta: positionPaddingIndex = 2
//        }
//        let positionEmbeddings = positionEmbedding.embeddings.slice(
//                lowerBounds: [positionPaddingIndex, 0],
//                upperBounds: [positionPaddingIndex + sequenceLength, -1]
//        ).expandingShape(at: 0)
//        var embeddings = tokenEmbeddings + positionEmbeddings
//
//        // Add token type embeddings if needed, based on which BERT variant is being used.
//        switch variant {
//        case .bert, .albert, .electra: embeddings = embeddings + tokenTypeEmbeddings
//        case .roberta: break
//        }
//
//        embeddings = embeddingLayerNorm(embeddings)
//        embeddings = embeddingDropout(embeddings)
//
//        if case .albert = variant {
//            embeddings = embeddingProjection[0](embeddings)
//        }
//
//        // Create an attention mask for the inputs with shape
//        // `[batchSize, sequenceLength, sequenceLength]`.
//        let attentionMask = createAttentionMask(forTextBatch: input)
//
//        // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a
//        // 3-D tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free
//        // on TPUs, and so we want to minimize them to help the optimizer.
//        var transformerInput = embeddings.reshapedToMatrix()
//        let batchSize = embeddings.shape[0]
//
//        // Run the stacked transformer.
//        switch variant {
//        case .bert, .roberta, .electra:
////            transformerInput = encoder(
////                    TransformerInput(
////                            sequence: transformerInput,
////                            attentionMask: attentionMask,
////                            batchSize: batchSize
////                    )
////            )
//            for layerIndex in 0..<(withoutDerivative(at: encoder.encoderLayers) {
//                $0.count
//            }) {
//                transformerInput = encoder.encoderLayers[layerIndex](
//                        TransformerInput(
//                                sequence: transformerInput,
//                                attentionMask: attentionMask,
//                                batchSize: batchSize
//                        )
//                )
//            }
//        case let .albert(_, hiddenGroupCount):
//            let groupsPerLayer = Float(hiddenGroupCount) / Float(hiddenLayerCount)
//            for layerIndex in 0..<hiddenLayerCount {
//                let groupIndex = Int(Float(layerIndex) * groupsPerLayer)
//                transformerInput = encoder.encoderLayers[groupIndex](TransformerInput(
//                        sequence: transformerInput,
//                        attentionMask: attentionMask,
//                        batchSize: batchSize))
//            }
//        }
//
//        // Reshape back to the original tensor shape.
//        return transformerInput.reshapedFromMatrix(originalShape: embeddings.shape)
        return probs
    }
}
