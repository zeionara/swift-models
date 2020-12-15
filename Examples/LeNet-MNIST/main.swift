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

public extension Array where Element == Double {
  public func average() -> Double {
    return self.reduce(0, +) / Double(self.count)
  }
}

// let epochCount = 1
// let batchSize = 256

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

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

func trainTeacher(nEpochs: Int = 1, batchSize: Int = 256) -> (TeacherModel, [Double]) {
  
  let dataset = MNIST(batchSize: batchSize, on: device)
  
  // The LeNet-5 model, equivalent to `LeNet` in `ImageClassificationModels`.
  var teacher = Sequential {
    Conv2D<TeacherScalar>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    AvgPool2D<TeacherScalar>(poolSize: (2, 2), strides: (2, 2))
    Conv2D<TeacherScalar>(filterShape: (5, 5, 6, 16), activation: relu)
    AvgPool2D<TeacherScalar>(poolSize: (2, 2), strides: (2, 2))
    Flatten<TeacherScalar>()
    Dense<TeacherScalar>(inputSize: 400, outputSize: 120, activation: relu)
    Dense<TeacherScalar>(inputSize: 120, outputSize: 84, activation: relu)
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
    callbacks: [try! CSVLogger().log]
  )

  trainingLoop.statisticsRecorder!.setReportTrigger(.endOfBatch)


  let teacher_: TeacherModel? = Optional.none
  let validationTimes = try! trainingLoop.fit(&teacher, epochs: nEpochs, on: device, teacher: teacher_)

  return (teacher, validationTimes)

}

func trainStudent(nEpochs: Int = 1, batchSize: Int = 256, teacher: TeacherModel) -> (StudentModel, [Double]) {
  
  let dataset = MNIST(batchSize: batchSize, on: device)
  
  // The LeNet-5 model, equivalent to `LeNet` in `ImageClassificationModels`.
  var model = Sequential {
    Conv2D<StudentScalar>(filterShape: (5, 5, 1, 3), padding: .same, activation: relu)
    AvgPool2D<StudentScalar>(poolSize: (2, 2), strides: (2, 2))
    Conv2D<StudentScalar>(filterShape: (5, 5, 3, 8), activation: relu)
    AvgPool2D<StudentScalar>(poolSize: (2, 2), strides: (2, 2))
    Flatten<StudentScalar>()
    Dense<StudentScalar>(inputSize: 200, outputSize: 50, activation: relu)
    // Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    Dense<StudentScalar>(inputSize: 50, outputSize: 10)
  }

  var optimizer = SGD(for: model, learningRate: 0.1)

  var trainingLoop = TrainingLoop<
    MNIST<SystemRandomNumberGenerator>.Training, MNIST<SystemRandomNumberGenerator>.Validation, Tensor<Int32>, SGD<StudentModel>, TeacherScalar, StudentScalar
  >(
    training: dataset.training,
    validation: dataset.validation,
    optimizer: optimizer,
    lossFunction: softmaxCrossEntropy,
    metrics: [.accuracy],
    callbacks: [try! CSVLogger().log]
  )

  trainingLoop.statisticsRecorder!.setReportTrigger(.endOfBatch)

  let validationTimes = try! trainingLoop.fit(&model, epochs: nEpochs, on: device, teacher: teacher)

  return (model, validationTimes)

}

var logger = Logger(label: "root")
logger.logLevel = .info

let (teacher, teacherValidationTimes) = try measureExecitionTime(prefix: "Trained teacher in") {
  trainTeacher(nEpochs: 1, batchSize: 256)
} log: { message, _ in
  logger.notice("\(message)")
}

logger.notice("Average teacher validation time: \(String(format: "%.\(3)f", teacherValidationTimes.average())) seconds")

let (student, studentValidationTimes) = try measureExecitionTime(prefix: "Trained student in") {
  trainStudent(nEpochs: 1, batchSize: 256, teacher: teacher)
} log: { message, _ in
  logger.notice("\(message)")
}

let validationTimeDifference = teacherValidationTimes.average() / studentValidationTimes.average()

logger.notice("Average student validation time: \(String(format: "%.\(3)f", studentValidationTimes.average())) seconds (\(String(format: "%.\(3)f", validationTimeDifference)) times faster than teacher)")
