# Bolt

## 1 Introduction

Bolt is a light-weight inference toolbox for mobile devices. Bolt, as a universal deployment tool for all kinds of neural networks, aims to minimize the inference runtime as much as possible. Higher speed, better security and more efficient memory management are the advantages that Bolt strives to provide.



## 2 Features

### 2.1 Supported Deep Learning Platform

caffe, onnx, tflite, pytorch (via onnx), tensorflow (via onnx). 

### 2.2 Supported Operators

- Attention
- BatchNorm
- Clip
- Concat
- Convolution
- Eltwise
- Embedding
- FullyConnected
- Gelu
- HSigmoid
- HSwish
- LayerNorm
- LSTM
- MatMul
- Multiply
- Pad
- Pooling
- Relu
- Relu6
- Reshape
- Scale
- Sigmoid
- Slice
- Softmax
- TanH
- Transpose

### 2.3 Supported Inference Precision Types

fp16, int8, binary

### 2.4 Verified Networks

Bolt supports common neural networks such as Sequential, CNN, LSTM etc.

Verified CV models include [squeezenet](https://github.com/forresti/SqueezeNet), [resnet50](https://github.com/KaimingHe/deep-residual-networks#models), [mobilenet_v1](https://github.com/shicai/MobileNet-Caffe), [mobilenet_v2](https://github.com/shicai/MobileNet-Caffe), [mobilenet_v3](https://github.com/jixing0415/caffe-mobilenet-v3), [birealnet18](https://github.com/JDAI-CV/dabnn) etc.

Verified NLP models include lstm, [bert](https://github.com/google-research/bert), tinybert, [albert](https://github.com/google-research/google-research/tree/master/albert) etc. 



## 3 Compilation and Installation

Before compilation,  you need to install some dependencies and set environment variables accordingly.

Two ways of compilation are provided. For direct compilation, you can compile Bolt on arm devices directly binding the dependent libraries as dynamic libraries. For cross compilation, you can compile Bolt on x86 devices binding the dependent libraries as static libraries.

More compilation details, please refer to [INSTALL.md](https://github.com/huawei-noah/bolt/blob/master/INSTALL.md).



## 4 User Guide

The typical use case of Bolt can be summarized into the following 3 steps:

(1) Compile Bolt. Two sets of executables shall be generated. The first set is for model converting, such as `caffe2bolt`, `onnx2bolt`, `tflite2bolt` etc. The other set is for the inference tasks, such as `classification`, `bert` etc.  The following steps use `caffe2bolt` and `classification` as example.

(2) Use `caffe2bolt` to convert Caffe model (demo.prototxt / demo.caffemodel) to Bolt format (demo.bolt).

(3) Run `classification` with the Bolt model and target inputs.

More details can be found below in Section 4.2.

### 4.1 How to implement a sequential model

Sequential model is a linear model. You can self-define a personalized model and deploy it on Bolt. Here we take Lenet as a simple example:

```c++
int main(int argc, char* argv[]) {
    char* imageDir = (char*)"";
    if(argc != 2) {
        print_help(argv);
    }
    else
        imageDir = argv[1];

    const Arch A = ARM_A76;
    DataType dt = DT_F16;
    auto model = Sequential<A>(dt, "lenet");

    auto op = Factory::createConvolution<A>(dt, 8, 5, 1, 2, ACTIVATION_NULL, ACTIVATION_NULL, Convolution_Pointwise, 1, 1);
    model.add(op);

    op = Factory::createPooling<A>(PoolingMode::Max, 2, 2, 0, RoundMode::CEIL);
    model.add(op);

    op = Factory::createConvolution<A>(dt, 8, 3, 1, 1, ACTIVATION_NULL, ACTIVATION_NULL, Convolution_Pointwise, 1, 1);
    model.add(op);

    op = Factory::createPooling<A>(PoolingMode::Max, 2, 2, 0, RoundMode::CEIL);
    model.add(op);

    op = Factory::createFullyConnectedEltwise<A>(dt, 10);
    model.add(op);

    op = Factory::createSoftmax<A>(dt);
    model.add(op);

    TensorDesc imageDesc = tensor4df(DT_F16, DF_NCHW, 1, 1, 8, 8);

    auto weight = (F16*)operator new(256*256*256*sizeof(F16));
    for (int i = 0; i < 256*256*256; i++) {
        weight[i] = 1;
    }
    U8* wPtr = (U8*)weight;
    std::shared_ptr<U8> modelPtr(wPtr);
    model.ready({imageDesc}, modelPtr);

    // load images
    Vec<Tensor> images;
    load_images(imageDir, imageDesc, &images, BGR, 1.0);

    for (auto image: images) {
        Vec<Tensor> input;
        input.push_back(image);
        model.set_input_tensors(input);

        model.run();

        auto outputs = model.get_output_tensors();
        outputs[0].print<F16>();
    }

    return 0;
}
```

You may also refer to `engine/tests/lenet.cpp` for details. When you compile the source code of Bolt, the `lenet` application will also be generated (`engine/bin/lenet` ).

### 4.2 How to convert and deploy a CNN model

You can also load a trained cnn model, and deploy it on bolt.

```c++
int main(int argc, char* argv[]){
    // pass the file parameter upon on personalized situation 

    const Arch A = NEON;
    ModelSpec ms;
    deserialize_model_from_file(model_path, &ms);
    auto cnn = createCNN<A>(&ms);
    
    // load images
    Vec<Tensor> images;
    HashMap<std::string, std::shared_ptr<Tensor>> in_map = cnn->get_inputs();
    TensorDesc image_desc = (*(in_map.begin()->second)).get_desc();
    Vec<std::string> iamge_paths = load_images(image_dir, image_desc, &image, scale_value);
    
    for(auto image: images){
        // set input
        Vec<Tensor> input;
        input.push_back(image);
        cnn->set_input_tensors(input);
        
        // run
        cnn->run();
        
        // get result
        HashMap<std::string, std::shared_ptr<Tensor>> out_map = cnn->get_outputs();
        Tensor result = *(out_map.begin()->second);
    }
    return 0;
}
```

As mentioned above, you can get the classification results in 3 steps.

- Compile Bolt and get `model-tools/bin/caffe2bolt` and `engine/bin/classification`.
- Secondly, you should convert the Caffe model like this:

```
./caffe2bolt /model_storage_path model_name
```

`caffe2bolt` takes at least two arguments. One is the storage path of the Caffe model files. The other is the model name, and `caffe2bolt`  will look for model_name.prototxt and model_name.caffemodel in the specified directory.

- Thirdly, set the Bolt model and the images as the inputs to `classification`, and run it like this:

```
./classification  bolt_model_path  input_data_directory_path  image_style scale_value  TOPK  correct_label
```

`classification` takes 6 arguments. In addition to the paths  for the Bolt model and the image folder, you can select the preprocessing style required by the model. For example, you should set image_style to BGR for Caffe models, and set scale_value to 1 for resnet50 and 0.017 for mobilenets. If you want to get TOP5 accuracy, please set TOPK to 5. Lastly, please specify the correct label number for the input image folder.



## 5 Benchmark

### 5.1 Accuracy

| model\acc        | top1(official) | top1(bolt) | top5(official) | top5(bolt) |
| ---------------- | -------------- | ---------- | -------------- | ---------- |
| resnet50         | 75.30%         | 75.60%     | 92.20%         | 95.51%     |
| mobilenet_v1     | 70.81%         | 70.13%     | 89.85%         | 92.23%     |
| squeezenet       | 57.50%         | 61.61%     | 80.30%         | 87.69%     |
| Birealnet18(BNN) | 56.40%         | 54.95%     | 79.50%         | 81.61%     |



### 5.2 speed

Here we list the single-thread execution time measured on Kirin 810.

| model\speed  | fp16 on A55 | fp16 on A76 | int8 on A55   | int8 on A76  |
| ------------ | ----------- | ----------- | ------------- | ------------ |
| resnet50     | 393.89 ms   | 95.86 ms    | 289.95 ms (*) | /            |
| mobilenet_v1 | 70.38 ms    | 19.85 ms    | /             | /            |
| mobilenet_v2 | 69.4 ms     | 18.27 ms    | /             | /            |
| squeezenet   | 46.97 ms    | 12.16 ms    | 40.15 ms      | 12.15 ms (*) |
| bert         | 5359.9 ms   | 1520.26 ms  | /             | /            |
| tinybert     | 45.63 ms    | 12.25 ms    | /             | /            |
| albert_tiny  | 143 ms      | 39 ms       | /             | /            |
| albert       | 1972 ms     | 488 ms      | /             | /            |

| model\speed | BNN on A55 | BNN on A76 |
| ----------- | ---------- | ---------- |
| Birealnet18 | 77.66 ms   | 30.70 ms   |

(*) Experimental support without mature optimization



## 6 Developer Guide

Everyone can self-define new operators in Bolt. We welcome the community to contribute functionalities in tensor_computing, engine and model-tools to make Bolt more and more versatile.

For more details, you can refer to [DEVELOPER.md](https://github.com/huawei-noah/bolt/blob/master/DEVELOPER.md). We appreciate your contributions! Anyone who has contributed to Bolt will be recorded into the [CONTRIBUTORS.md](https://github.com/huawei-noah/bolt/blob/master/CONTRIBUTORS.md).



## 7 FAQ

(1) Q : What are the dependent libraries?

A : The two major dependencies are Protobuf and CImg. Please refer to `model-tools/dependency/` and `image/dependency/` for more details.



(2) Q : Requirements on tensor dimensions?

A : For optimal performance, Bolt requires the number of output channels to be divisible by 8.



(3) Q : Restrictions for BNN?

A : For BNN layers, the number of output channels must be divisible by 32.



(4) Q : Restrictions on convolution and pooling?

A : Currently, Bolt requires that the kernel_size / stride / padding should be the same in height and width dimension.



(5) Q : Restrictions on quantization (int8)?

A: For the time being, Bolt only supports post-training int8 quantization. If quantization is activated,  the second convolution layer will quantize the tensors to 8-bit integers. For now, int8 operators include Convolution, Pooling and Concatenation (end-to-end support for Squeezenet). If your network includes other operators, you may need to add type casting in the front of those operators. The quantization method is symmetrical for both activation and weight.



## 8 Acknowledgement

Bolt refers to the following projects: [caffe](https://github.com/BVLC/caffe), [onnx](https://github.com/onnx/onnx), [protobuf](https://github.com/protocolbuffers/protobuf), [flatbuffers](https://github.com/google/flatbuffers), [ncnn](https://github.com/Tencent/ncnn), [mnn](https://github.com/alibaba/MNN), [dabnn](https://github.com/JDAI-CV/dabnn).

## QQ Technology Group
833345709

## License

The MIT License(MIT)
