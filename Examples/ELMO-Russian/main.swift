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

let projectRoot = "\(ProcessInfo.processInfo.environment["HOME"]!)/swift-models"
let device = Device.default
let specialTokens = [String]() // ["[CLS]", "[SEP]", "[UNK]"]

let nEpochs = 10

// extension Array {
//     public func applyMask(mask: [Int32], reversed: Bool = false) -> [Element] {
//         var result = [Element]()

//         for (i, item) in enumerated() {
//             if mask[i] == 0 && reversed || mask[i] == 1 && !reversed {
//                 result.append(item)
//             }
//         }

//         return result
//     }

//     public func batched(size: Int, shouldRandomize: Bool = true, filter: Optional<(Element) -> Bool> = Optional.none) -> [[Element]] {
//         var filteredArray = self
//         if shouldRandomize {
//             filteredArray.shuffle()
//         }
//         if let filter_ = filter {
//             filteredArray = filteredArray.filter(filter_)
//         }
//         let nBatches = Int((Float(filteredArray.count) / Float(size)).rounded(.up))
//         return (0..<nBatches).map {i in
//             let lastIndex = Swift.min((i+1)*size, filteredArray.count)
//             return Array(
//                 filteredArray[i*size..<lastIndex]
//             )
//         }
//     }
// }

func getModelsRootPath() -> URL{
    return URL(fileURLWithPath: "/home/zeio/swift-models/assets/models")
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

try generateVocabulary(corpusName: "baneks")
//
//let nMaskedTokensPerSequence = 5
let stringSentences = try String(contentsOf: URL(fileURLWithPath: "\(projectRoot)/assets/baneks/sentences.txt")).components(separatedBy: "\n")
let vocabulary = try Vocabulary(fromFile: URL(fileURLWithPath: "\(projectRoot)/assets/baneks/vocabulary.txt"), bert: false)
var model = ELMO(vocabulary: vocabulary, tokenizer: BERTTokenizer(vocabulary: vocabulary), embeddingSize: 100)//, hiddenSize: 50)
var optimizer = Adam(for: model, learningRate: 0.05)
let sequences = ["твой анек", "больше лайков", "вчера говорили"] // ["Купил мужик", "она анекдот", "как раз"] // ["a b", "c d", "e f", "a f"]

let preprocessedText = model.preprocess(sequences: stringSentences, maxSequenceLength: 128)
let nBatches = preprocessedText.count

for epochIndex in 0...nEpochs {
    let epochStartTime = DispatchTime.now().uptimeNanoseconds
    for (batchIndex, batch) in preprocessedText.enumerated() {
        // print("a \(batch.count)")
        // let preprocessedText = model.preprocess(sequences: batch)
        // print("b \(preprocessedText.shape)")
        // let probs = model(batch)
        // print("c")
        // print("Current batch: \(batch)")
        // if batchIndex < preprocessedText.count - 1 {
            // print("Next batch: \(preprocessedText[batchIndex + 1])")
        // }
        let targetWordIndices = Tensor(batch.flattened().unstacked().map{i in i.scalar!}.dropFirst() +
            [batchIndex == preprocessedText.count - 1 ? 0 : preprocessedText[batchIndex + 1].unstacked().first!.scalar!])
        // print("Target word indices: \(targetWordIndices)")
        // print("d")
        // let labels = model.vocabulary.oneHotEncodings(forIds: targetWordIndices.map{Int($0)})
        // print("e")

        let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let probs = model(batch)
            let res = softmaxCrossEntropy(logits: probs, labels: targetWordIndices)
            return res
        }

        // print("f")

        optimizer.update(&model, along: grad)

        // print("g")

        if (epochIndex % 100 == 0) && (batchIndex == 0) {
            print("After \(epochIndex) epochs: ")
            print("Loss: \(loss)")
            model.test(sequences)
        }

        print("Completed \(batchIndex + 1)/\(nBatches) batch in \(epochIndex + 1)/\(nEpochs) epoch")
    }
    print("Completed \(epochIndex)th epoch in \((DispatchTime.now().uptimeNanoseconds - epochStartTime) / 1_000_000_000) seconds")

    try model.save(getModelsRootPath())
}


//print(labels.shape)
//
//var languageModel = BERTLanguageModel(bert: model)
//let preds = languageModel(preprocessedText)

//let mask = preprocessedText.languageModelMask!.flattened().unstacked().map{ i in i.scalar!}
//print(preprocessedText.languageModelMask!.flattened().unstacked().map{ i in i.scalar!})
//let probs = Tensor(preds.reshaped(to: [-1, model.vocabulary.count]).unstacked().applyMask(mask: mask, reversed: true))
//print(preprocessedText.tokenIds.flattened().unstacked().map{ i in i.scalar!}.applyMask(mask: mask, reversed: true))
//let labels = model.vocabulary.oneHotEncodings(forIds: preprocessedText.tokenIds.flattened().unstacked().map{ i in Int(i.scalar!)}.applyMask(mask: mask, reversed: true))

// print(kullbackLeiblerDivergence(predicted: probs, expected: softmax(labels * 10, alongAxis: 1)))

//var optimizer = x10_optimizers_optimizer.GeneralOptimizer(
//        for: languageModel,
//        TensorVisitorPlan(languageModel.differentiableVectorView),
//        defaultOptimizer: makeSGD(
//                learningRate: 0.02
//        )
//)



//var optimizer = x10_optimizers_optimizer.GeneralOptimizer(
//        for: languageModel,
//        TensorVisitorPlan(languageModel.differentiableVectorView),
//        defaultOptimizer: makeSGD(
//                learningRate: 0.02
//        )
//)









// print(embs.shape)

// try model.save(getModelsRootPath())

let testModel = try ELMO(getModelsRootPath())

testModel.test(sequences)

///
///
///


//var trainingLoop: TrainingLoop = TrainingLoop(
//        training: cola.trainingEpochs,
//        validation: cola.validationBatches,
//        optimizer: optimizer,
//        lossFunction: sigmoidCrossEntropyReshaped,
//        metrics: [.matthewsCorrelationCoefficient],
//        callbacks: [
//            clipGradByGlobalNorm,
//            LearningRateScheduler(
//                    scheduledParameterGetter: scheduledParameterGetter,
//                    biasCorrectionBeta1: beta1,
//                    biasCorrectionBeta2: beta2).schedule
//        ])



///
///
///

//let (nSequences, nTokensPerSequence, nTokensInVocabulary) = (languageModel(preprocessedText).shape[0], languageModel(preprocessedText).shape[1], languageModel(preprocessedText).shape[2])
//
//for sequenceId in 0..<nSequences {
//    let sequenceTokensAttentionMask = preprocessedText.mask[sequenceId]
//    print(sequenceTokensAttentionMask)
//}


//var bertPretrained: BERT.PreTrainedModel
//if CommandLine.arguments.count >= 2 {
//    if CommandLine.arguments[1].lowercased() == "albert" {
//        bertPretrained = BERT.PreTrainedModel.albertBase
//    } else if CommandLine.arguments[1].lowercased() == "roberta" {
//        bertPretrained = BERT.PreTrainedModel.robertaBase
//    } else if CommandLine.arguments[1].lowercased() == "electra" {
//        bertPretrained = BERT.PreTrainedModel.electraBase
//    } else {
//        bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
//    }
//} else {
//    bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
//}
//
//let bert = try bertPretrained.load()
//var bertClassifier = BERTClassifier(bert: bert, classCount: 1)
//bertClassifier.move(to: device)
//
//// Regarding the batch size, note that the way batching is performed currently is that we bucket
//// input sequences based on their length (e.g., first bucket contains sequences of length 1 to 10,
//// second 11 to 20, etc.). We then keep processing examples in the input data pipeline until a
//// bucket contains enough sequences to form a batch. The batch size specified in the task
//// constructor specifies the *total number of tokens in the batch* and not the total number of
//// sequences. So, if the batch size is set to 1024, the first bucket (i.e., lengths 1 to 10)
//// will need 1024 / 10 = 102 examples to form a batch (every sentence in the bucket is padded
//// to the max length of the bucket). This kind of bucketing is common practice with NLP models and
//// it is done to improve memory usage and computational efficiency when dealing with sequences of
//// varied lengths. Note that this is not used in the original BERT implementation released by
//// Google and so the batch size setting here is expected to differ from that one.
//let maxSequenceLength = 128
//let batchSize = 1024
//let epochCount = 3
//let stepsPerEpoch = 1068  // function of training set size and batching configuration
//let peakLearningRate: Float = 2e-5
//
//let workspaceURL = URL(
//        fileURLWithPath: "bert_models", isDirectory: true,
//        relativeTo: URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true))
//
//var cola = try CoLA(
//        taskDirectoryURL: workspaceURL,
//        maxSequenceLength: maxSequenceLength,
//        batchSize: batchSize,
//        entropy: SystemRandomNumberGenerator(),
//        on: device
//) { example in
//    // In this closure, both the input and output text batches must be eager
//    // since the text is not padded and x10 requires stable shapes.
//    let classifier = bertClassifier
//    let textBatch = classifier.bert.preprocess(
//            sequences: [example.sentence],
//            maxSequenceLength: maxSequenceLength)
//    return LabeledData(data: textBatch, label: Tensor<Int32>(example.isAcceptable! ? 1 : 0))
//}
//
//print("Dataset acquired.")
//
//let beta1: Float = 0.9
//let beta2: Float = 0.999
//let useBiasCorrection = true
//
//var optimizer = x10_optimizers_optimizer.GeneralOptimizer(
//        for: bertClassifier,
//        TensorVisitorPlan(bertClassifier.differentiableVectorView),
//        defaultOptimizer: makeSGD(
//                learningRate: peakLearningRate
//        )
//)
//
///// Computes sigmoidCrossEntropy loss from `logits` and `labels`.
/////
///// This defines the loss function used in TrainingLoop; it's a wrapper of the
///// standard sigmoidCrossEntropy; it reshapes logits to required shape before
///// calling the standard sigmoidCrossEntropy.
//@differentiable
//public func sigmoidCrossEntropyReshaped<Scalar>(logits: Tensor<Scalar>, labels: Tensor<Int32>)
//                -> Tensor<
//        Scalar
//        > where Scalar: TensorFlowFloatingPoint {
//    return sigmoidCrossEntropy(
//            logits: logits.squeezingShape(at: -1),
//            labels: Tensor<Scalar>(labels),
//            reduction: _mean)
//}
//
///// Clips the gradients by global norm.
/////
///// This's defined as a callback registered into TrainingLoop.
//func clipGradByGlobalNorm<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
//    if event == .updateStart {
//        var gradients = loop.lastStepGradient!
//        gradients.clipByGlobalNorm(clipNorm: 1)
//        loop.lastStepGradient = gradients
//    }
//}
//
///// A function that returns a LinearlyDecayedParameter but with first 10 steps linearly warmed up;
///// for remaining steps it decays at slope of -(peakLearningRate / `totalStepCount`).
//let scheduledParameterGetter = { (_ totalStepCount: Float) -> LinearlyDecayedParameter<LinearlyWarmedUpParameter<FixedParameter<Float>>> in
//    LinearlyDecayedParameter(
//            baseParameter: LinearlyWarmedUpParameter(
//                    baseParameter: FixedParameter<Float>(peakLearningRate),
//                    warmUpStepCount: 10,
//                    warmUpOffset: 0),
//            slope: -(peakLearningRate / totalStepCount), // The LR decays linearly to zero.
//            startStep: 10
//    )
//}
//
//var trainingLoop: TrainingLoop = TrainingLoop(
//        training: cola.trainingEpochs,
//        validation: cola.validationBatches,
//        optimizer: optimizer,
//        lossFunction: sigmoidCrossEntropyReshaped,
//        metrics: [.matthewsCorrelationCoefficient],
//        callbacks: [
//            clipGradByGlobalNorm,
//            LearningRateScheduler(
//                    scheduledParameterGetter: scheduledParameterGetter,
//                    biasCorrectionBeta1: beta1,
//                    biasCorrectionBeta2: beta2).schedule
//        ])
//
//print("Training \(bertPretrained.name) for the CoLA task!")
//try! trainingLoop.fit(&bertClassifier, epochs: epochCount, on: device)
