# LeNet-5 with MNIST

This example demonstrates how to train the [LeNet-5 network]( http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) against the [MNIST digit classification dataset](http://yann.lecun.com/exdb/mnist/).

The LeNet network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the MNIST dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.


## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

This tool demonstrates:
- Using output from a more powerful model for training a smaller one without significant loss in results quality;
- Throwing out layers from the original model for speeding up inference time;
- Reducing filter sizes for decreasing memory footprint;
- Replacing advanced activation functions with simpler ones;
- Using more economical scalar datatypes (e.g. `BFloat16` or recently introduced `TensorFloat` instead of the widespread `Float`).

To observe the effect of model optimization, run (if you have a sufficiently powerful gpu, you can omit setting env variable `CUDA_VISIBLE_DEVICES`):

```sh
cd swift-models
swift build --product LeNet-MNIST
CUDA_VISIBLE_DEVICES=-1 .build/debug/LeNet-MNIST -a 4 -n 10
```

Key points from the command output (both models were trained on CPU):

```sh
accuracy: 0.9847
Trained teacher in 750.952 seconds
Average teacher validation time: 5.9152 seconds

accuracy: 0.9772
Trained student in 316.126 seconds
Average student validation time: 1.0662 seconds (5.5481 times faster than teacher)
```

Key points from the command output (both models were trained on GPU without and with XLA respectively):

```sh
accuracy: 0.9835
Trained teacher in 235.443 seconds
Average teacher validation time: 0.9836 seconds

accuracy: 0.9776
Trained student in 111.172 seconds
Average student validation time: 0.3544 seconds (2.7757 times faster than teacher)
```

Comparison of the student model trained with and without a teacher (in this case training student without a teacher looks more attractive):

```sh
accuracy: 0.9831
Trained teacher in 449.977 seconds
Average teacher validation time: 1.9012 seconds

accuracy: 0.9787
Trained student in 191.255 seconds
Average student validation time: 0.4047 seconds (4.6977 times faster than teacher)

accuracy: 0.9801
Trained student in 84.828 seconds
Average student validation time: 0.4379 seconds
```
