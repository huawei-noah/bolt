# Quantization Toolchain

So far bolt supports various modes of post-training quantization, including quantized storage, dynamic quantization inference, calibration, etc. In the future, we will also provide quantization training tools.

## post_training_quantization

Please refer to [model_tools/tools/quantization/post_training_quantization.cpp](../model_tools/tools/quantization/post_training_quantization.cpp). All post-training quantization utilities are covered in this tool, except the calibration, which will also be merged into this tool in the future.

Before using this tool, you need to first produce the input model with X2bolt using the "-i PTQ" option. Later, you can use the tool:

```
./post_training_quantization -p model_ptq_input.bolt
```

Different options of the tool are explained below. The default setting will produce model_int8_q.bolt which will be executed with dynamic int8 quantization.

Here are the list of covered utilities:

1. **Quantized Storage**: If you would like to compress your model, use the -q option. Choose from {FP16, INT8, MIX}. INT8 storage could lead to accuracy drop, so we provided the MIX mode which will try to avoid accuracy-critical layers. Note that this option is independent from the -i option, which sets the inference precision.
2. **Global Clipping of GEMM Inputs**: In some cases of quantization-aware training, GEMM inputs will be clipped so that they can be better quantized symmetrically. Please use the -c option is necessary.
3. **Ad-Hoc Clipping of Feature Maps**: In some other cases, the clip value is a trainable parameter for individual layers. Please use the -s option. The parameter **scaleFileDirectory** is the directory of your scale table file(.txt). Note that text format of the file is like the following codes, and **clipvalue** is the clip value of each feature map in your model. In our tool, we will calculate true scales of each tensor with the equation clipvalue/127.0 and store them into the created int8 model.
```
tensor_name_0 clipvalue
tensor_name_1 clipvalue
tensor_name_2 clipvalue
```

## Calibration tool

The post training quantization calibration tool is in the directory [inference/engine/tools/ptq_calibration/ptq_calibration.cpp](../inference/engine/tools/ptq_calibration/ptq_calibration.cpp). The command to use this tool is :
```
./ptq_calibration modelPath dataDirectory dataFormat scaleValue affinityPolicyName algorithmMapPath 
```
So these parameters are :

1. **modelPath** : the directory of your int8 Bolt model, make sure that you get your int8 Bolt model with our converter tool X2Bolt and then you can use this post training quantization calibration tool with your own related calibration datasets.
2. **dataDirectory** : the directory of your calibration datasets, note that the structure of the folder is :
```
HWSEA:/data/local/tmp/test # cd calibration_dataset
HWSEA:/data/local/tmp/test # ls
XXXXX.JPEG XXXXX.JPEG XXXXX.JPEG XXXXX.JPEG XXXXX.JPEG
```
3. **dataFormat** : specific imageFormat : BGR/RGB/RGB_SC/BGR_SC_RAW/BGR_SC_R
4. **scaleValue** : specific scaleValue for image classification, the default value is 1
5. **affinityPolicyName** : specific running mode: CPU_AFFINITY_HIGH_PERFORMANCE/CPU_AFFINITY_LOW_POWER/GPU, the default value is CPU_AFFINITY_HIGH_PERFORMANCE.
6. **algorithmMapPath** : specific file path to read or write algorithm auto tunning result

After running this post training quantization calibration tool, you will get a int8-KL Bolt model named by **_int8_q_KL.bolt** in the directory of the folder which stores your original int8 model. 
