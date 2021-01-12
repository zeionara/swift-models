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
import ImageClassificationModels
import TensorFlow
import TrainingLoop
import Logging
import ArgumentParser

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

let dataset = CIFAR10(batchSize: 10, on: device)
var model = ResNet(classCount: 10, depth: .resNet56, downsamplingInFirstStage: false)
var optimizer = SGD(for: model, learningRate: 0.001)

var trainingLoop = TrainingLoop<CIFAR10<SystemRandomNumberGenerator>.Training, CIFAR10<SystemRandomNumberGenerator>.Validation, Tensor<Int32>, SGD<ResNet>, Float, Float>(
  training: dataset.training,
  validation: dataset.validation,
  optimizer: optimizer,
  lossFunction: softmaxCrossEntropy,
  metrics: [.accuracy])

let teacher: ResNet? = Optional.none
try! trainingLoop.fit(&model, epochs: 10, on: device, teacher: teacher)
