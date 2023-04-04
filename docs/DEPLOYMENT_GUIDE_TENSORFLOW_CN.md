快速上手用Bolt部署Tensorflow模型

# 目录
---
&nbsp;&nbsp;&nbsp;&nbsp;[方法1：通过转换成tflite模型部署](#方法1：通过转换成tflite模型部署) 
&nbsp;&nbsp;&nbsp;&nbsp;[方法2：通过转换成onnx模型部署](#方法2：通过转换成onnx模型部署) 
&nbsp;&nbsp;&nbsp;&nbsp;[方法3：通过转换成caffe模型部署（不推荐）](#方法3：通过转换成caffe模型部署（不推荐）) 
&nbsp;&nbsp;&nbsp;&nbsp;[方法4：直接用bolt部署tensorflow模型（不推荐）](#方法4：直接用bolt部署tensorflow模型（不推荐）) 
&nbsp;&nbsp;&nbsp;&nbsp;[常见问题](#常见问题) 


# 方法1：通过转换成tflite模型部署

- ### 步骤1：从tensorflow格式模型导出成tflite格式模型

    详细可参考[官方说明文档](https://www.tensorflow.org/lite/convert?hl=zh-cn)导出tflite模型文件。

- ### 步骤2：用bolt部署tflite模型

    详细请参考[DEPLOYMENT_GUIDE_TFLITE_CN.md](DEPLOYMENT_GUIDE_TFLITE_CN.md)。


# 方法2：通过转换成onnx模型部署

- ### 步骤1： 从tensorflow格式模型导出成onnx格式模型

    使用[tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)导出onnx模型文件。

- ### 步骤2：用bolt部署onnx模型

    详细请参考[DEPLOYMENT_GUIDE_ONNX_CN.md](DEPLOYMENT_GUIDE_ONNX_CN.md)。


# 方法3：通过转换成caffe模型部署（不推荐）

- ### 步骤1：tensorflow格式模型转换成caffe格式模型

    * 方法1：使用第三方工具转换tensorflow模型到caffe模型

    * 方法2：使用bolt自带的半自动化工具转换tensorflow模型到caffe模型

        详细请参考[model_tools/tools/tensorflow2caffe](../model_tools/tools/tensorflow2caffe)。

- ### 步骤2：用bolt部署caffe模型

    详细请参考[DEPLOYMENT_GUIDE_CAFFE_CN.md](DEPLOYMENT_GUIDE_CAFFE_CN.md)。


# 方法4：直接用bolt部署tensorflow模型（不推荐）

   详细请参考[USER_HANDBOOK.md](USER_HANDBOOK.md#tensorflow-model-conversion)。


# 常见问题
