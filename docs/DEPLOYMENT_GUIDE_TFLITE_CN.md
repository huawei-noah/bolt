快速上手用Bolt部署Tflite模型

# 目录
---
&nbsp;&nbsp;&nbsp;&nbsp;[方法1：直接用bolt部署tflite模型](#方法1：直接用bolt部署tflite模型) 
&nbsp;&nbsp;&nbsp;&nbsp;[方法2：通过转换成onnx模型部署](#方法2：通过转换成onnx模型部署) 
&nbsp;&nbsp;&nbsp;&nbsp;[附：tflite部署](#附：tflite部署)
&nbsp;&nbsp;&nbsp;&nbsp;[常见问题](#常见问题) 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[bolt转换和推理添加新算子](#bolt转换和推理添加新算子)

# 方法1：直接用bolt部署tflite模型

  *对于常见的CNN模型可以采用直接部署的方式，对于含有复杂reshape/transpose等算子的网络，建议通过转换成onnx模型部署。*

- ### 下载和编译bolt项目

    请参考[INSTALL.md](INSTALL.md)。

- ### 使用X2bolt转换tflite格式模型到bolt格式模型

    详细请参考[USER_HANDBOOK.md](USER_HANDBOOK.md#model-conversion)或者 *--help*。

    * 示例：转换./example.tflite到./example_f32.bolt
    
    ```bash
    ./X2bolt -d ./ -m example -i FP32
    ```
    
- ### 通用benchmark测试
    
    详细请参考[USER_HANDBOOK.md](USER_HANDBOOK.md#model-inference)或者 *--help*。
    
    * 示例：CPU推理./example_f32.bolt，查看模型输入输出信息和推理时间。
    
    ```bash
    ./benchmark -m ./example_f32.bolt
    ```
    
- ### C/C++/Java API开发
    
    详细请参考[DEVELOPER.md](DEVELOPER.md##use-out-of-the-box-api-to-infer-your-model)。
    
    
# 方法2：通过转换成onnx模型部署
    
- ### tflite转onnx
    
    [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)提供了工具转换tflite模型到onnx模型。
    
- ### 用bolt部署onnx模型
    
    详细请参考[DEPLOYMENT_GUIDE_ONNX_CN.md](DEPLOYMENT_GUIDE_ONNX_CN.md)。


# 附：tflite部署

- ### 使用tflite python推理

    bolt提供了一个简单的示例运行脚本[inference/engine/tools/tflite_tools/benchmark.py](../inference/engine/tools/tflite_tools/benchmark.py)，支持单输入/多输入，支持指定输入文件或输入文件夹，支持添加新输出tensor(中间tensor)。
    
    * 示例：全1输入测试
    
    ```bash
    python inference/engine/tools/tflite_tools/benchmark.py ./example.tflite
    ```
    
    * 示例：当前目录./下单txt文件输入，模型有1个输入，名字为input，input.txt是输入input的数据内容，用空格分割。
    
    ```bash
    python inference/engine/tools/tflite_tools/benchmark.py ./example.tflite ./input.txt
    ```
    
    * 示例：当前目录./下多txt文件输入，模型有2个输入，名字分别为input0和input1，input0.txt是输入input0的数据内容，用空格分割，input0.txt是输入input1的数据内容。
    
    ```bash
    python inference/engine/tools/tflite_tools/benchmark.py ./example.tflite ./
    ```
    
    * 示例：当前目录./下多txt文件输入，输入维度是动态的，模型有2个输入，名字分别为input0和input1，shape.txt记录实际推理输入维度，input0.txt是输入input0的数据内容，用空格分割，input0.txt是输入input1的数据内容。
    
    shape.txt内容：维度用空格分割
    
    ```bash
    input0 1 3 224 224
    input1 1 1 224 224
    ```
    
    ```bash
    python inference/engine/tools/tflite_tools/benchmark.py ./example.tflite ./
    ```
    
    * 示例：全1输入测试，添加新输出，新添加的2个输出名字tensor0和tensor1存放在output.txt
    
    output.txt内容：
    
    ```bash
    tensor0
    tensor1
    ```
    
    ```bash
    python inference/engine/tools/tflite_tools/benchmark.py ./example.tflite None ./output.txt
    ```

# 常见问题

- ### bolt转换和推理添加新算子

    详细请参考[DEVELOPER.md](DEVELOPER.md#customize-models-with-unsupported-operators-step-by-step)。
