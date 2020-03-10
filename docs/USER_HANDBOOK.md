Before you try any step described in this document, please make sure you have installed Bolt correctly. You can refer to [INSTALL.md](INSTALL.md) for more details.



# Basic Usage

### Model Conversion



1. **Caffe model to Bolt model**

  <1> Push the `caffe2bolt` executable file to the phone; 

  <2> Push the caffe model to the phone;

  <3> Use `caffe2bolt` to transform model of caffe format to model of bolt format

  		Parameters:    caffe_model_path    caffe_model_name    precision

  ​        Note: Your should make sure the .prototxt file and the .caffemodel file have the same model name

Example: Transform mobilenet_v1 of caffe format into bolt format

```shell
<1> adb push /home/bolt/install_llvm/tools/caffe2bolt /data/local/bolt/tools/caffe2bolt
<2> adb push /home/bolt/models/caffe/mobilenet_v1/ /data/local/bolt_model/caffe/mobilenet_v1
<3> adb shell "./data/local/bolt/tools/caffe2bolt ./data/local/bolt_model/caffe/mobilenet_v1/ mobilenet_v1 FP16"
```

  After running, you can see the mobilenet_v1_f16.bolt file in the same directory with the original caffe model.

  The suffix "_f16" indicates that the bolt model is saved in FP16 representations, and will be run with FP16 operations (ARMv8.2) by default.

  If you want to deploy the model as FP32, please set the last parameter to "FP32" for caffe2bolt. You will then get mobilenet_v1_f32.bolt.

  This precision setting also applies to onnx2bolt and tflite2bolt.



2. **Onnx model to Bolt model**	

  <1> Push the `onnx2bolt` executable file to the phone;

  <2> Push the onnx model to the phone;

  <3> Use `onnx2bolt` to transform model of onnx format to model of bolt format

  ​		Parameters:    onnx_model_path    onnx_model_name    remove_op_number    precision    inputN    inputC    inputH    inputW

Example: Transform ghostnet of onnx format into bolt format

```shell
<1> adb push /home/bolt/tools/onnx2bolt /data/local/bolt/tools/onnx2bolt
<2> adb push /home/bolt/models/onnx/ghostnet/ /data/local/bolt_model/caffe/ghostnet
<3> adb shell "./data/local/bolt/tools/onnx2bolt ./data/local/bolt_model/onnx/ghostnet/ ghostnet 3 FP16 1 3 224 224"
```

  After running, you can see the ghostnet_f16.bolt file in /data/local/bolt_model/onnx/ghostnet/ on the phone.

  Since onnx models may not specify the input dimensions, onnx2bolt accepts 4 more parameters. If they are not provided, the .bolt model will specify 1x3x224x224 by default, which is the typical input size for ImageNet networks.



3. **TensorFlow model to Bolt model**	

  The process flow is : TensorFlow model to Caffe model, and then to Bolt model.

  <1> Tensorflow model to Caffe model

  Refer to the [tensorflow2caffe README.md](../model-tools/tools/tensorflow2caffe/README.md) for more details on transforming TensorFlow model to Caffe model.

  <2> Caffe model to Bolt model

  Refer to the former steps in "Caffe model to Bolt model" section in this chapter.



4. **PyTorch model to Bolt model**

  PyTorch should have native support for onnx format. For your own convenience, you can try that first.

  The process flow is: PyTorch model to Caffe model, and then to Bolt model

  <1> PyTorch model to Caffe model

  Refer to the [pytorch2caffe README.md](../model-tools/tools/pytorch2caffe/README.md) for more details on transforming Pytorch model to Caffe model.

  <2> Caffe model to Bolt model

  Refer to the former steps in "Caffe model to Bolt model" section in this chapter.



### Model Inference

We provide several demo programs, and here we will explain the usage of two typical programs: image classification and tinybert.



1. **Classification**

   <1> Push classification to the phone;
   
   <2> Push the testing image data to the phone;
   
   <3> Run classification and get the result.
   
   Parameters:    bolt_model    image_directory    image_format    scale_value    TopK    correct_label    thread_affinity

Example: Run mobilenet_v1 for image classification 

```shell
<1> adb push /home/bolt/install_llvm/bin/classification /data/local/bolt/bin/classification
<2> adb push /home/bolt/data/ILSVRC/n02085620/ /data/local/bolt_data/cv/ILSVRC/n02085620
<3> adb shell "./data/local/bolt/bin/classification /data/local/bolt_model/caffe/mobilenet_v1/mobilenet_v1_f16.bolt /data/local/bolt_data/cv/ILSVRC/n02085620 BGR 0.017 5 151 CPU_AFFINITY_HIGH_PERFORMANCE"
```

  After running, you should be able to see the TopK labels for each image calculated according to the model, the Top1 and TopK accuracy, and the execution time.

  Here we explain a little more for some of the parameters.

  - image_format:    The image format requested by the model. For example, caffe models usually require BGR format. You can refer to [image_processing.cpp](../image/src/image_processing.cpp) for more details.
  - scale_value:     The scale value requested in the input preprocessing. This value is also used in [image_processing.cpp](../image/src/image_processing.cpp). If your network required normalized inputs, the typical scale value is 0.017.
  - TopK:            The number of predictions that you are interested in for each image. Typical choice is 5.
  - correct_label:   The correct label number for the whole image directory.
  - thread_affinity: When it is set to be CPU_AFFINITY_HIGH_PERFORMANCE, Bolt will look for a high-frequency core and bind to it. When it is set to be CPU_AFFINITY_LOW_POWER, Bolt will look for a low-frequency core. If the parameter is missing, the default value is "CPU_AFFINITY_HIGH_PERFORMANCE".



2. **Tinybert**

   <1> Push tinybert to the phone;

   <2> Push the testing sequence data to the phone;

   <3> Run tinybert and get the result.

   Parameters:    bolt_model    sequence_directory    thread_affinity

Example:

```shell
<1> adb push /home/bolt/bin/tinybert /data/local/bolt/bin/tinybert
<2> adb mkdir /data/local/bolt_data/nlp/tinybert/data
<3> adb mkdir /data/local/bolt_data/nlp/tinybert/data/input
<4> adb mkdir /data/local/bolt_data/nlp/tinybert/data/result
<5> adb push /home/bolt/model-tools/tools/tensorflow2cafee/tinybert/sequence.seq /data/local/bolt_data/nlp/tinybert/data/input/0.seq
<6> adb shell "./data/local/bolt/bin/tinybert /data/local/bolt_model/caffe/tinybert/tinybert_f16.bolt /data/local/bolt_data/nlp/tinybert/data CPU_AFFINITY_HIGH_PERFORMANCE"
```

  After running, you should be able to see the labels for each sequence calculated according to the model, and the execution time.

  Here we explain a little more for some of the parameters.

  - thread_affinity: When it is set to be CPU_AFFINITY_HIGH_PERFORMANCE, Bolt will look for a high-frequency core and bind to it. When it is set to be CPU_AFFINITY_LOW_POWER, Bolt will look for a low-frequency core. If the parameter is missing, the default value is "CPU_AFFINITY_HIGH_PERFORMANCE".



### API

Currently, we provide C and Java API. After installation, you can find the API documents docs/API/html/index.html.



# Advanced Features

### Graph Optimization

  By default, all graph optimizers that we have implemented are activated during model conversion. In the converters (caffe2bolt, onnx2bolt), you can find a function call:
  ```c++
  ms_optimizer.suggest();
  ```
  If you wish to turn them off, you can adjust the suggest() function, or simply call:
  ```c++
  ms_optimizer.empty();
  ```
  However, some of the optimizers are essential, which will be marked with * below.

  - *DeprecatedOpOptimizer: This optimizer removes the deprecated layers from the model
  - *ConvBNOptimizer: This optimizer folds BN parameters into the weight and bias of convolution.
  - *BNScaleOptimizer: When a BN layer is not precedented by a convolution layer, we will fold it into the following scale layer.
  - *ConvScaleOptimizer: This optimizer folds scale parameters into the weight and bias of convolution.
  - InPlaceOptimizer: If the input and output of a layer are identical in dimensions, they might share the same tensor name. Typical layers include the Activation Layer.
  - ConvActivationOptimizer: This optimizer fuses convolution and activation layers
  - *ChannelPaddingOptimizer: This optimizer will pad the output channels to a multiple of 8 for convolution layers. This increases the model compatibility.
  - DepthwisePointwiseOptimizer: This optimizers fuses depthwise conv and pointwise conv for computation efficiency.
  - TransposeMulToScaleOptimizer: This is useful for some NLP models.
  - *MemoryReuseOptimizer: When a feature map tensor is no longer needed as input or output, the storage that it occupies can be reused by other feature maps. This saves on average **two-thirds** of feature map storage for networks that we have tested.



### INT8 Post-Training Quantization

  If quantization is activated, the second convolution layer will quantize the tensors to 8-bit integers. For now, int8 operators include Convolution, Pooling and Concatenation (end-to-end support for Squeezenet). If your network includes other operators, you may need to add type casting in the front of those operators. The quantization method is symmetrical for both activation and weight.

  If you want to activate the quantization, pass "INT8_Q" as the precision parameter to caffe2bolt or onnx2bolt during model conversion.



### BNN Network Support

  Bolt supports both XNOR-style and DoReFa-style BNN networks. Just save the binary weights as FP32 in an Onnx model, and onnx2bolt will automatically convert the storage to 1-bit representations. So far, the floating-point portion of the BNN network can only be FP16 operations, so pass "FP16" as the precision parameter to onnx2bolt. The number of output channels for BNN convolution layers should be divisible by 32.



### Layer Performance Benchmark

  If you target device is an Android phone connected to your compilation server, you can call "make test" to run a quick verification test, which runs the [quick_benchmark.sh](../quick_benchmark.sh). For more details, please refer to the individual unit test programs under [tests](../tests).



### Algorithm Tuning for Key Layers

   Bolt provides tensor_computing_library_search program for performance tuning of the operator library. Bolt currently supports convolution layer algorithm tuning.

   <1> Push tensor_computing_library_search to the phone;

   <2> Set Bolt_TensorComputing_LibraryAlgoritmMap shell environment variable

   <3> Run library tuning program.
        
   <4> Use *CONVOLUTION_LIBRARY_SEARCH* convolution policy during model inference.

Example:

```shell
<1> adb push /home/bolt/inference/tools/tensor_computing_library_search /data/local/bolt/tools/tensor_computing_library_search
<2> adb shell "export Bolt_TensorComputing_LibraryAlgoritmMap=/data/local/bolt/tensor_computing_library_algorithm_map.txt && ./data/local/bolt/tools/tensor_computing_library_search"
```

  After running, you should be able to get algorithm map file on device.



# Feedback

  If you have encountered any difficulty, feel free to reach out to us by summitting issues. You are also encouraged to contribute your implementations. Please refer to [DEVELOPER.md](DEVELOPER.md).
