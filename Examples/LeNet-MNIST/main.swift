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

import Datasets
import TensorFlow
import TrainingLoop
import Logging
import ArgumentParser

public extension Array where Element == Double {
  public func average() -> Double {
    return self.reduce(0, +) / Double(self.count)
  }
}

// let epochCount = 1
// let batchSize = 256

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
// #if os(macOS)
//   let device = Device.defaultTFEager
// #else
//   let device = Device.defaultXLA
// #endif

typealias TeacherScalar = Double
typealias StudentScalar = Float

typealias TeacherModel = Sequential<
  Conv2D<TeacherScalar>,
  Sequential<AvgPool2D<TeacherScalar>,
  Sequential<Conv2D<TeacherScalar>,
  Sequential<AvgPool2D<TeacherScalar>,
  Sequential<Flatten<TeacherScalar>,
  Sequential<Dense<TeacherScalar>,
  Sequential<Dense<TeacherScalar>,
  Dense<TeacherScalar>
>>>>>>>
typealias StudentModel = Sequential<
  Conv2D<StudentScalar>,
  Sequential<AvgPool2D<StudentScalar>,
  Sequential<Conv2D<StudentScalar>,
  Sequential<AvgPool2D<StudentScalar>,
  Sequential<Flatten<StudentScalar>,
  Sequential<Dense<StudentScalar>,
  Dense<StudentScalar>
>>>>>>

public func computeSoftmaxCrossEntropyUsingLabels<LogitsScalar, LabelsScalar, ResultScalar>(_ logits: Tensor<LogitsScalar>, _ labels: Tensor<LabelsScalar>) -> Tensor<ResultScalar>
where LogitsScalar: TensorFlowFloatingPoint, LabelsScalar: TensorFlowScalar, LabelsScalar: BinaryInteger, ResultScalar: TensorFlowFloatingPoint {
  return Tensor<ResultScalar>(softmaxCrossEntropy(logits: Tensor<Float>(logits), labels: Tensor<Int32>(labels))) 
}

struct LeNetOptimizer: ParsableCommand {

  private enum ModelDevice: String, ExpressibleByArgument {
      case eager
      case xla
  }

  @Option(name: .shortAndLong, default: 3, help: "Precision with which floating-point values will be printed")
  var accuracy: Int

  @Option(name: .shortAndLong, default: 1, help: "Number of epochs for models training")
  var nEpochs: Int

  @Option(name: .shortAndLong, default: 256, help: "How many images to precess at once")
  var batchSize: Int

  @Option(default: .eager, help: "Device for training and evaluating a teacher model")
  private var teacherDevice: ModelDevice

  @Option(default: .eager, help: "Device for training and evaluating a student model")
  private var studentDevice: ModelDevice

  @Flag(name: .shortAndLong, help: "Include results of trainining student model without a teacher")
  private var disabledTeacherMode = false

  private func trainTeacher(nEpochs: Int = 1, batchSize: Int = 256, device: Device = .default) -> (TeacherModel, [Double]) {
    // let device: Device = .default

    let dataset = MNIST(batchSize: batchSize, on: device)
    
    // The LeNet-5 model, equivalent to `LeNet` in `ImageClassificationModels`.
    var teacher = Sequential {
      Conv2D<TeacherScalar>(filterShape: (5, 5, 1, 6), padding: .same, activation: gelu)
      AvgPool2D<TeacherScalar>(poolSize: (2, 2), strides: (2, 2))
      Conv2D<TeacherScalar>(filterShape: (5, 5, 6, 16), activation: gelu)
      AvgPool2D<TeacherScalar>(poolSize: (2, 2), strides: (2, 2))
      Flatten<TeacherScalar>()
      Dense<TeacherScalar>(inputSize: 400, outputSize: 120, activation: gelu)
      Dense<TeacherScalar>(inputSize: 120, outputSize: 84, activation: gelu)
      Dense<TeacherScalar>(inputSize: 84, outputSize: 10)
    }

    var teacherOptimizer = SGD(for: teacher, learningRate: 0.1)

    var trainingLoop = TrainingLoop<
      MNIST<SystemRandomNumberGenerator>.Training, MNIST<SystemRandomNumberGenerator>.Validation, Tensor<Int32>, SGD<TeacherModel>, TeacherScalar, TeacherScalar
    >(
      training: dataset.training,
      validation: dataset.validation,
      optimizer: teacherOptimizer,
      lossFunction: computeSoftmaxCrossEntropyUsingLabels,
      metrics: [.accuracy],
      callbacks: [try! CSVLogger().log],
      accuracy: accuracy
    )

    trainingLoop.statisticsRecorder!.setReportTrigger(.endOfBatch)


    let teacher_: TeacherModel? = Optional.none
    let validationTimes = try! trainingLoop.fit(&teacher, epochs: nEpochs, on: device, teacher: teacher_)

    return (teacher, validationTimes)
  }

  private func trainStudent(nEpochs: Int = 1, batchSize: Int = 256, teacher: TeacherModel?, teacherDevice: Device, device: Device = .default) -> (StudentModel, [Double]) {
    // let device: Device = .defaultXLA

    let dataset = MNIST(batchSize: batchSize, on: device)
    
    // The LeNet-5 model, equivalent to `LeNet` in `ImageClassificationModels`.
    var model = Sequential {
      Conv2D<StudentScalar>(copying: Conv2D<StudentScalar>(filterShape: (5, 5, 1, 3), padding: .same, activation: relu), to: device)
      AvgPool2D<StudentScalar>(copying: AvgPool2D<StudentScalar>(poolSize: (2, 2), strides: (2, 2)), to: device)
      Conv2D<StudentScalar>(copying: Conv2D<StudentScalar>(filterShape: (5, 5, 3, 8), activation: relu), to: device)
      AvgPool2D<StudentScalar>(copying: AvgPool2D<StudentScalar>(poolSize: (2, 2), strides: (2, 2)), to: device)
      Flatten<StudentScalar>(copying: Flatten<StudentScalar>(), to: device)
      Dense<StudentScalar>(copying: Dense<StudentScalar>(inputSize: 200, outputSize: 50, activation: relu), to: device)
      // Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
      Dense<StudentScalar>(copying: Dense<StudentScalar>(inputSize: 50, outputSize: 10), to: device)
    }

    // var model = Sequential {
    //   Conv2D<StudentScalar>(filterShape: (5, 5, 1, 3), padding: .same, activation: relu)
    //   AvgPool2D<StudentScalar>(poolSize: (2, 2), strides: (2, 2))
    //   Conv2D<StudentScalar>(filterShape: (5, 5, 3, 8), activation: relu)
    //   AvgPool2D<StudentScalar>(poolSize: (2, 2), strides: (2, 2))
    //   Flatten<StudentScalar>()
    //   Dense<StudentScalar>(inputSize: 200, outputSize: 50, activation: relu)
    //   // Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    //   Dense<StudentScalar>(inputSize: 50, outputSize: 10)
    // }

    var optimizer = SGD(copying: SGD(for: model, learningRate: 0.1), to: device)

    print("Training")

    var trainingLoop = TrainingLoop<
      MNIST<SystemRandomNumberGenerator>.Training, MNIST<SystemRandomNumberGenerator>.Validation, Tensor<Int32>, SGD<StudentModel>, TeacherScalar, StudentScalar
    >(
      training: dataset.training,
      validation: dataset.validation,
      optimizer: optimizer,
      lossFunction: softmaxCrossEntropy,
      metrics: [.accuracy],
      callbacks: [try! CSVLogger().log],
      accuracy: accuracy
    )

    trainingLoop.statisticsRecorder!.setReportTrigger(.endOfBatch)

    let validationTimes = try! trainingLoop.fit(&model, epochs: nEpochs, on: device, teacher: teacher, teacherDevice: teacherDevice, studentDevice: device)

    return (model, validationTimes)

  }

  mutating func run() throws {
    // var teacherModel: TeacherModel? = Optional.none
    // var teacherModelDevice: Device = .default
    // var averageTeacherValidationTime: Double? = Optional.none
    
    var logger = Logger(label: "root")

    logger.logLevel = .info

    let teacherDevice_ = teacherDevice == .eager ? Device.default : Device.defaultXLA
    // let studentDevice_ = Device.default // studentDevice == .eager ? Device.default : Device.defaultXLA 

    let (teacher, teacherValidationTimes) = try measureExecitionTime(prefix: "Trained teacher") {
      trainTeacher(nEpochs: nEpochs, batchSize: batchSize, device: teacherDevice_)
    } log: { message, _ in
      logger.notice("\(message)")
    }

    logger.notice("Average teacher validation time: \(String(format: "%.\(accuracy)f", teacherValidationTimes.average())) seconds")

    let (student, studentValidationTimes) = try measureExecitionTime(prefix: "Trained student") {
      trainStudent(nEpochs: nEpochs, batchSize: batchSize, teacher: teacher, teacherDevice: teacherDevice_, device: studentDevice == .eager ? Device.default : Device.defaultXLA)
    } log: { message, _ in
      logger.notice("\(message)")
    }

    logger.notice("Average student validation time: \(String(format: "%.\(accuracy)f", studentValidationTimes.average())) seconds (\(String(format: "%.\(accuracy)f", teacherValidationTimes.average() / studentValidationTimes.average())) times faster than teacher)")

    if disabledTeacherMode {
      let (student, studentValidationTimes) = try measureExecitionTime(prefix: "Trained student") {
        trainStudent(nEpochs: nEpochs, batchSize: batchSize, teacher: Optional.none, teacherDevice: teacherDevice_, device: studentDevice == .eager ? Device.default : Device.defaultXLA)
      } log: { message, _ in
        logger.notice("\(message)")
      }

      logger.notice("Average student validation time: \(String(format: "%.\(accuracy)f", studentValidationTimes.average())) seconds")
    }  
  }
}

struct NNOptimizer: ParsableCommand {
    static var configuration = CommandConfiguration(
            abstract: "A tool for demonstrating approaches for neural networks optimization",
            subcommands: [LeNetOptimizer.self], // TrainExternally.self
            defaultSubcommand: LeNetOptimizer.self
    )
}

NNOptimizer.main()
