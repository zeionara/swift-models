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

import ModelSupport
import TensorFlow
import Logging
import Foundation

// Workaround https://bugs.swift.org/browse/TF-1122 that prevents us from registering a
// loss function inside our TrainingLoop struct
public final class LossFunctionWrapper<Output: Differentiable, Target> {
  public typealias F = @differentiable(Output, @noDerivative Target) -> Tensor<Float>
  public typealias S = @differentiable(Output, @noDerivative Output) -> Tensor<Float>
  public var f: F
  public var s: S
  init(_ f: @escaping F, _ s: @escaping S) { self.f = f; self.s = s }
}

public func computeSoftmaxCrossEntropy<LogitsScalar, ProbabilitiesScalar, ResultScalar>(_ logits: Tensor<LogitsScalar>, _ probabilities: Tensor<ProbabilitiesScalar>) -> Tensor<ResultScalar>
where LogitsScalar: TensorFlowFloatingPoint, ProbabilitiesScalar: TensorFlowFloatingPoint, ResultScalar: TensorFlowFloatingPoint {
  return Tensor<ResultScalar>(softmaxCrossEntropy(logits: Tensor<Float>(logits), probabilities: Tensor<Float>(probabilities))) 
}

public func measureExecitionTime<ResultType>(prefix: String = "Executed", nDecimalPlaces: Int = 3, executeClosure: () throws -> ResultType, log: (String, Double) -> Void) throws -> ResultType {
  let executionStartTimestamp = DispatchTime.now().uptimeNanoseconds
  let result = try executeClosure()
  let elapsedTime = Double(DispatchTime.now().uptimeNanoseconds - executionStartTimestamp) / 1e+9
  log("\(prefix) in \(String(format: "%.\(nDecimalPlaces)f", elapsedTime)) seconds", elapsedTime)
  return result
}

var logger = Logger(label: "training-loop")

/// Types whose elements represent a training loop.
///
/// - Note: This protocol is mainly there to give us an easy type for a generic `TrainingLoop`
///   and unless you need to rewrite your own training loop entirely, you should use `TrainingLoop`.
public protocol TrainingLoopProtocol {
  // Associatedtypes
  /// The type of the sequence of epochs for the training data.
  associatedtype Training
  where
    Training: Sequence, Training.Element: Collection
    // Training.Element.Element == LabeledData<Opt.Model.Input, Target>

  /// The type of the collection of batches for the validation data.
  associatedtype Validation
  where
    Validation: Collection
    // Validation.Element == LabeledData<Opt.Model.Input, Target>

  /// The type of the teacher Model
  // associatedtype TeachingModel

  associatedtype TeacherScalar: TensorFlowFloatingPoint
  associatedtype StudentScalar: TensorFlowFloatingPoint

  /// The type of the target of our model.
  associatedtype Target

  /// The type of the optimizer used.
  associatedtype Opt: Optimizer where Opt.Model: Module

  // Typealiases
  /// The type of the model.
  typealias Model = Opt.Model

  /// The type of the input of the model.
  typealias Input = Opt.Model.Input

  /// The type of the output of the model.
  typealias Output = Opt.Model.Output

  /// The type of a batch.
  typealias Batch = LabeledData<Input, Target>

  // In a wrapper for now because of TF-1122.
  /// The type of the loss function.
  typealias LossFunction = LossFunctionWrapper<Output, Target>

  // Data
  /// The training epochs.
  var training: Training { get }

  /// The validation batches.
  var validation: Validation { get }

  // Optimizer and loss function
  /// The optimizer.
  var optimizer: Opt { get set }

  /// The loss function.
  var lossFunction: LossFunction { get set }

  /// The metrics on which training is measured.
  var metrics: [TrainingMetrics] { get set }

  // Callbacks
  /// The callbacks used to customize the training loop.
  var callbacks: [TrainingLoopCallback<Self>] { get set }

  // Temporary data

  // MARK: - Step-level data

  /// The last input fed to the model.
  var lastStepInput: Input? { get set }

  /// The last target.
  var lastStepTarget: Target? { get set }

  /// The last predictions of the model.
  var lastStepOutput: Output? { get set }

  /// The last gradients computed.
  var lastStepGradient: Model.TangentVector? { get set }

  /// The last loss.
  var lastStepLoss: Tensor<Float>? { get set }

  /// The number of batches in the current collection of batches.
  var batchCount: Int? { get set }

  /// The index of the current batch.
  var batchIndex: Int? { get set }

  // MARK: - Epoch-level data

  /// The number of epochs we are currently fitting for.
  var epochCount: Int? { get set }

  /// The index of the current epoch.
  var epochIndex: Int? { get set }

  // MARK: - Others

  /// The log for last statistics
  var lastStatsLog: [(name: String, value: Float)]? { get set }
}

/// The events that occur during a call to `fit` in the `TrainingLoop`
///
/// - Note: The method is called `fit` and not `train` because it trains the model and validates it.
///   Each epoch is composed of a *training* phase and a *validation* phase.
public enum TrainingLoopEvent {
  /// The start of a fit.
  case fitStart

  /// The end of a fit.
  case fitEnd

  /// The start of one epoch (training + validation).
  case epochStart

  /// The start of one epoch (training + validation).
  case epochEnd

  /// The start of a training phase.
  case trainingStart

  /// The end of a training phase.
  case trainingEnd

  /// The start of a validation phase.
  case validationStart

  /// The end of a validation phase.
  case validationEnd

  /// The start of a training or inference step on a batch.
  case batchStart

  /// The end of a training or inference step on a batch.
  case batchEnd

  /// At the start of the optimizer update, just after the differentiable step.
  case updateStart

  /// Just after the model prediction at inference, before computing the loss.
  case inferencePredictionEnd
}

/// Callbacks that can inject custom behavior in a training loop.
public typealias TrainingLoopCallback<L: TrainingLoopProtocol> = (
  _ loop: inout L, _ event: TrainingLoopEvent
) throws -> Void

/// A generic training loop.
///
/// - Parameter `Training`: the type of the sequence of epochs for training data.
/// - Parameter `Validation`: the type of the collection of batches for validation.
/// - Parameter `Target`: the type of the target.
/// - Parameter `Opt`: the type of the optimizer used.
public struct TrainingLoop<
  Training: Sequence, Validation: Collection, Target, Opt: Optimizer, TeacherScalar, StudentScalar
>: TrainingLoopProtocol
where
  Training.Element: Collection, // Training.Element.Element == LabeledData<Opt.Model.Input, Target>,
  Opt.Model: Module, // Validation.Element == LabeledData<Opt.Model.Input, Target>, 
  TeacherScalar: TensorFlowFloatingPoint, StudentScalar: TensorFlowFloatingPoint
{
  // Typealiases
  /// The type of the model.
  public typealias Model = Opt.Model

  /// The type of the input of the model.
  public typealias Input = Opt.Model.Input

  /// The type of the output of the model.
  public typealias Output = Opt.Model.Output

  /// The type of a batch.
  public typealias Batch = LabeledData<Input, Target>

  // In a wrapper for now because of TF-1122.
  /// The type of the loss function.
  public typealias LossFunction = LossFunctionWrapper<Output, Target>

  // Data
  /// The training epochs.
  public let training: Training

  /// The validation batches.
  public let validation: Validation

  // Optimizer and loss function
  /// The optimizer.
  public var optimizer: Opt

  /// The loss function
  public var lossFunction: LossFunction

  /// The metrics
  public var metrics: [TrainingMetrics]

  /// Callbacks

  /// The callbacks used to customize the training loop.
  public var callbacks: [TrainingLoopCallback<Self>]

  // MARK: - Default callback objects

  /// The callback that records the training statistics.
  public var statisticsRecorder: StatisticsRecorder? = nil

  /// The callback that prints the training progress.
  public var progressPrinter: ProgressPrinter? = nil

  /// Temporary data

  // MARK: - Step-level data

  /// The last input fed to the model.
  public var lastStepInput: Input? = nil

  /// The last target.
  public var lastStepTarget: Target? = nil

  /// The last predictions of the model.
  public var lastStepOutput: Output? = nil

  /// The last gradients computed.
  public var lastStepGradient: Model.TangentVector? = nil

  /// The last loss.
  public var lastStepLoss: Tensor<Float>? = nil

  /// The number of batches in the current collection of batches.
  public var batchCount: Int? = nil

  /// The index of the current batch.
  public var batchIndex: Int? = nil

  // MARK: - Epoch-level data

  /// The number of epochs we are currently fitting for.
  public var epochCount: Int? = nil

  /// The index of the current epoch.
  public var epochIndex: Int? = nil

  // MARK: - Others

  /// The log for last statistics
  public var lastStatsLog: [(name: String, value: Float)]? = nil

  /// Creates an instance from `training` and `validation` data, a `model`, an `optimizer` and a
  /// `lossFunction`.
  ///
  /// Parameter callbacks: Callbacks that the `TrainingLoop` will use in every call to fit.
  public init(
    training: Training, validation: Validation, optimizer: Opt,
    lossFunction: @escaping LossFunction.F,
    metrics: [TrainingMetrics] = [],
    callbacks: [TrainingLoopCallback<Self>] = [],
    includeDefaultCallbacks: Bool = true
  ) {
    logger.logLevel = .info
    self.training = training
    self.validation = validation
    self.optimizer = optimizer
    self.lossFunction = LossFunction(lossFunction) { output, target in
      // print("Computing cv...")
      computeSoftmaxCrossEntropy((output as! Tensor<StudentScalar>), (target as! Tensor<StudentScalar>))
      // print("Computed cv...")
      // return result
      // softmaxCrossEntropy(logits: (output as! Tensor<Float>), probabilities: (target as! Tensor<Float>))
    }
    self.metrics = metrics

    if includeDefaultCallbacks {
      let statisticsRecorder = StatisticsRecorder(metrics: [.loss] + metrics)
      let progressPrinter = ProgressPrinter()
      self.statisticsRecorder = statisticsRecorder
      self.progressPrinter = progressPrinter
      self.callbacks = [
        statisticsRecorder.record,
        progressPrinter.printProgress,
      ] + callbacks
    } else {
      self.callbacks = callbacks
    }
  }
}

extension TrainingLoop {
  /// The default differentiable step.
  public mutating func differentiableStep(model: Model, teacherLogits: Output? = Optional.none) throws {
    guard let data = lastStepInput else { return }
    guard let target = lastStepTarget else { return }
    // let targetDevice = (target as! Tensor<Int32>).device
    // let fixedTarget = Tensor(
    //   (target as! Tensor<Int32>).unstacked().map{target_label in
    //     Tensor<Float>(
    //       (0..<10).map{$0 == target_label.scalar! ? 1.0 : 0.0},
    //       on: targetDevice
    //     )
    //   }
    // )
    (lastStepLoss, lastStepGradient) = valueWithGradient(at: model) {
      (model: Model) -> Tensor<Float> in
      let predictions = model(data)
      lastStepOutput = predictions
      // print((predictions as! Tensor<Float>).shape)
      // print((target as! Tensor<Int32>).shape)
      // print(fixedTarget)
      // print(softmax(predictions as! Tensor<Float>))
      // return lossFunction.s(predictions, (fixedTarget as! Output))
      if let logits = teacherLogits {
        // print("Computing cv...")
        return lossFunction.s(predictions, logits)
        // print("Computed cv...")
      } else {
        return lossFunction.f(predictions, target) // lossFunction.s(predictions, (fixedTarget as! Output))        
      }
    }
  }

  /// The step used for inference.
  public mutating func inferenceStep(model: Model, teacherLogits: Output? = Optional.none) throws {
    guard let data = lastStepInput else { return }
    lastStepOutput = model(data)
    guard let target = lastStepTarget else { return }
    try handleEvent(.inferencePredictionEnd)
    if let logits = teacherLogits {
      lastStepLoss = lossFunction.s(lastStepOutput!, logits)
    } else {
      lastStepLoss = lossFunction.f(lastStepOutput!, target)
    }
  }

  /// The step used for training.
  public mutating func trainingStep(
    model: inout Model, differentiableStep: (Model, inout Self, Output?) throws -> Void, teacherLogits: Output? = Optional.none
  ) throws {
    try differentiableStep(model, &self, teacherLogits)
    try handleEvent(.updateStart)
    optimizer.update(&model, along: lastStepGradient!)
  }
}

/// Control flow of the training loop.
///
/// - Note: Each of the "end" event is called after its corresponding "cancel" action for cleanup.
public enum TrainingLoopAction: Error {
  /// Abort actions in the current training/inference step and goes to the next batch.
  case cancelBatch

  /// Abort actions in the current training phase and goes to the validation phase.
  case cancelTraining

  /// Abort actions in the current validation phase and goes to the next epoch.
  case cancelValidation

  /// Abort actions in the current epoch and goes to the next epoch.
  case cancelEpoch

  /// Abort actions in the current fit and ends fitting.
  case cancelFit
}

extension TrainingLoop {
  /// Call `event` on all callbacks.
  mutating private func handleEvent(_ event: TrainingLoopEvent) throws {
    for callback in callbacks {
      try callback(&self, event)
    }
  }
}

extension TrainingLoop {
  /// Performs `step` on each of `batches`.
  mutating private func multipleSteps<Batches: Collection, TeachingModel>(
    on batches: Batches, step: (inout Self, Output?) throws -> Void, teacher: TeachingModel? = Optional.none
  ) throws where TeachingModel: Module { // Batches.Element == Batch, TeachingModel.Input == Opt.Model.Input
    batchCount = batches.count
    for (i, batch) in batches.enumerated() {
      batchIndex = i
      let fixedBatch = LabeledData<Tensor<StudentScalar>, Tensor<Int32>>(
        data: Tensor<StudentScalar>(
          (batch as! LabeledData<Tensor<Float>, Tensor<Int32>>).data
        ),
        label: (batch as! LabeledData<Tensor<Float>, Tensor<Int32>>).label
      )
      (lastStepInput, lastStepTarget) = ((fixedBatch as! Batch).data, (fixedBatch as! Batch).label)
      var teacherLogits: Output? = Optional.none
      if let unwrappedTeacher = teacher {
        teacherLogits = (
          Tensor<StudentScalar>(
            softmax(
              unwrappedTeacher(
                Tensor<TeacherScalar>(lastStepInput! as! Tensor<StudentScalar>) as! TeachingModel.Input
              ) as! Tensor<TeacherScalar>
            )
          ) as! Output
        )
      }
      // print(batch.label)
      do {
        try handleEvent(.batchStart)
        try step(&self, teacherLogits)
      } catch TrainingLoopAction.cancelBatch {}
      try handleEvent(.batchEnd)
      LazyTensorBarrier()
    }
  }
}

extension TrainingLoop {
  /// Fit the model for `epochs` using `callbacks` to customize the default training loop.
  ///
  /// - Parameters:
  ///   - inferenceStep: The step used during the validation phase of each epoch. The default value
  ///     uses the `inferenceStep` method of `TrainingLoop`.
  ///   - trainingStep: The step used during the training phase of each epoch. The default value
  ///     uses the `trainingStep` method of `TrainingLoop`.
  public mutating func fit<TeachingModel>(
    _ model: inout Model, epochs: Int, callbacks: [TrainingLoopCallback<Self>] = [],
    on device: Device = Device.default,
    differentiableStep: (Model, inout Self, Output?) throws -> Void = {
      try $1.differentiableStep(model: $0, teacherLogits: $2)
    },
    teacher: TeachingModel? = Optional.none
  ) throws -> [Double] where TeachingModel: Module { // TeachingModel.Input == Opt.Model.Input
    let callbacksCount = self.callbacks.count
    self.callbacks += callbacks
    defer { self.callbacks = Array(self.callbacks.prefix(callbacksCount)) }
    epochCount = epochs

    model.move(to: device)
    optimizer = Opt(copying: optimizer, to: device)

    var validationTimes = [Double]()

    do {
      try handleEvent(.fitStart)
      LazyTensorBarrier()

      for (i, batches) in training.prefix(epochs).enumerated() {
        epochIndex = i
        do {
          try handleEvent(.epochStart)

          // Training phase
          try measureExecitionTime(prefix: "Training phase has been finished in") {
            do {
              Context.local.learningPhase = .training
              try handleEvent(.trainingStart)
              try multipleSteps(
                on: batches,
                step: {
                  try $0.trainingStep(model: &model, differentiableStep: differentiableStep, teacherLogits: $1)
                },
                teacher: teacher
              )
            } catch TrainingLoopAction.cancelTraining {}
            try handleEvent(.trainingEnd)
          } log: { message, _ in
            logger.notice("\(message)")
          }

          // Validation phase
          try measureExecitionTime(prefix: "Validation phase has been finished in") {
            do {
              Context.local.learningPhase = .inference
              try handleEvent(.validationStart)
              let teacher_: TeachingModel? = Optional.none
              try multipleSteps(on: validation, step: { try $0.inferenceStep(model: model, teacherLogits: $1) }, teacher: teacher_)
            } catch TrainingLoopAction.cancelValidation {}
            try handleEvent(.validationEnd)
          } log: { message, elapsedTime in
            logger.notice("\(message)")
            validationTimes.append(elapsedTime)
          }
        } catch TrainingLoopAction.cancelEpoch {}
        try handleEvent(.epochEnd)
      }
    } catch TrainingLoopAction.cancelFit {}
    try handleEvent(.fitEnd)
    return validationTimes
  }
}
