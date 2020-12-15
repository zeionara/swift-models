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

// let epochCount = 1
// let batchSize = 256

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

typealias TeacherModel = Sequential<Conv2D<Float>, Sequential<AvgPool2D<Float>, Sequential<Conv2D<Float>, Sequential<AvgPool2D<Float>, Sequential<Flatten<Float>, Sequential<Dense<Float>, Sequential<Dense<Float>, Dense<Float>>>>>>>>
typealias StudentModel = Sequential<Conv2D<Float>, Sequential<AvgPool2D<Float>, Sequential<Conv2D<Float>, Sequential<AvgPool2D<Float>, Sequential<Flatten<Float>, Sequential<Dense<Float>, Dense<Float>>>>>>>

func trainTeacher(nEpochs: Int = 1, batchSize: Int = 256) -> TeacherModel {
  
  let dataset = MNIST(batchSize: batchSize, on: device)
  
  // The LeNet-5 model, equivalent to `LeNet` in `ImageClassificationModels`.
  var teacher = Sequential {
    Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Flatten<Float>()
    Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    Dense<Float>(inputSize: 84, outputSize: 10)
  }

  var teacherOptimizer = SGD(for: teacher, learningRate: 0.1)

  var trainingLoop = TrainingLoop(
    training: dataset.training,
    validation: dataset.validation,
    optimizer: teacherOptimizer,
    lossFunction: softmaxCrossEntropy,
    metrics: [.accuracy],
    callbacks: [try! CSVLogger().log]
  )

  trainingLoop.statisticsRecorder!.setReportTrigger(.endOfBatch)


  let teacher_: TeacherModel? = Optional.none
  try! trainingLoop.fit(&teacher, epochs: nEpochs, on: device, teacher: teacher_)

  return teacher

}

func trainStudent(nEpochs: Int = 1, batchSize: Int = 256, teacher: TeacherModel) -> StudentModel {
  
  let dataset = MNIST(batchSize: batchSize, on: device)
  
  // The LeNet-5 model, equivalent to `LeNet` in `ImageClassificationModels`.
  var model = Sequential {
    Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Flatten<Float>()
    Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    // Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    Dense<Float>(inputSize: 120, outputSize: 10)
  }

  var optimizer = SGD(for: model, learningRate: 0.1)

  var trainingLoop = TrainingLoop(
    training: dataset.training,
    validation: dataset.validation,
    optimizer: optimizer,
    lossFunction: softmaxCrossEntropy,
    metrics: [.accuracy],
    callbacks: [try! CSVLogger().log]
  )

  trainingLoop.statisticsRecorder!.setReportTrigger(.endOfBatch)

  try! trainingLoop.fit(&model, epochs: nEpochs, on: device, teacher: teacher)

  return model

}

let teacher = trainTeacher(nEpochs: 1, batchSize: 256)
let student = trainStudent(nEpochs: 1, batchSize: 256, teacher: teacher)
