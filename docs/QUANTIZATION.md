# Quantization Toolchain

Bolt currently supports various modes of post-training quantization, including quantized storage, dynamic quantization inference, offline calibration, etc. Bolt will provide quantization aware training tools in the future.

## post training quantization

Please refer to [model_tools/tools/quantization/post_training_quantization.cpp](../model_tools/tools/quantization/post_training_quantization.cpp). All post-training quantization utilities are covered in this tool.

Before using this tool, you need to first produce the input model with X2bolt using the "-i PTQ" option. Later, you can use the tool:

```
./post_training_quantization --help
./post_training_quantization -p model_ptq_input.bolt
```

Different options of the tool are explained below. The default setting will produce model_int8_q.bolt which will be executed with dynamic int8 quantization. INT8_FP16 is for machine(ARMv8.2+) that supports fp16 to compute non quantized operators. INT8_FP32 is for machine(ARMv7, v8. Intel AVX512) that supports fp32 to compute non quantized operators. The command above is equivalent to this one:

```
./post_training_quantization -p model_ptq_input.bolt -i INT8_FP16 -b true -q NOQUANT -c 0 -o false
```

Here are the list of covered utilities:

* **Quantized Storage**: If you would like to compress your model, use the -q option. Choose from {FP16, INT8, MIX}. INT8 storage could lead to accuracy drop, so we provided the MIX mode which will try to avoid accuracy-critical layers. Note that this option is independent from the -i option, which sets the inference precision. For example, if you want to run model with FP32 inference but store it using int8 weights, use this command:

    ```
    ./post_training_quantization -p model_ptq_input.bolt -i FP32 -q INT8
    ```

* **Global Clipping of GEMM Inputs**: In some cases of quantization-aware training (QAT), GEMM inputs will be clipped so that they can be better quantized symmetrically. For example, if the QAT uses a global clipping value of 2.5 for int8 inference, use this command:

    ```
    ./post_training_quantization -p model_ptq_input.bolt -i INT8_FP16 -c 2.5
    ```

* **Ad-Hoc Clipping with Quantization Aware Training**: In some other cases, the clip value is a trainable parameter for individual layers. Please use the -s option. The parameter **scaleFileDirectory** is the directory of your scale table file (.json). Note that text format of the file is like the following codes, the scale table contains all the layers that need to be quantized, and you can specify the clipvalue of any tensor. If the clipvalue of a output tensor is not set, the tensor will not be quantized during inference. In our tool, we will calculate true scales of each tensor with the equation clipvalue/127.0 and store them into the created int8 model. The example command is also given below.
    ```
    {
        "quantized_layer_name": {
            "inputs": {
                "tensor0": clipvalue_of_tensor0,
                "tensor1": clipvalue_of_tensor1
            },
            "weights": {
                "tensor2": clipvalue_of_tensor2
            },
            "outputs": {
                "tensor3": clipvalue_of_tensor3
            }
        },
        ...
    }

    ./post_training_quantization -p model_ptq_input.bolt -i INT8 -s /path/to/scale.json
    ```

* **Offline Calibration**: To use this mode, a calibration dataset should be provided. Currently bolt provides calibration of int8 models with KL-divergence. Please use the -o option to turn on this mode. Currently we accept a directory of jpeg images as the dataset, which should be specified with the -d option. The -f option sets the preprocessing style of inputs, and the -m option sets the multiplying scale used in preprocessing. Example usage:
    ```
    ./post_training_quantization -p model_ptq_input.bolt -i INT8_FP16 -o true -d calibration_dataset/ -f BGR -m 0.017
    ```

* More options:

    - The -b option sets whether to fuse BatchNorm parameters with weight of convolution, etc. Default value is true for highest inference speed. This option is useful for the following scenario: Usually in quantization-aware training, FakeQuant nodes are inserted before each convolution layer, but BatchNorm is not handled. In this case, if we fuse BatchNorm and then quantize the convolution weight, it will create a difference between training and inference. When you find that this difference leads to accuracy drop, you can set the -b option to false:

        ```
        ./post_training_quantization -p model_ptq_input.bolt -i INT8_FP16 -b false
        ```

    - The -V option triggers verbose mode that prints detailed information.

        ```
        ./post_training_quantization -V -p model_ptq_input.bolt
        ```
