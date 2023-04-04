快速上手用Bolt部署Onnx模型

# 目录
---
&nbsp;&nbsp;&nbsp;&nbsp;[步骤1：简化onnx模型（推荐，可选）](#步骤1：简化onnx模型（推荐，可选）) 
&nbsp;&nbsp;&nbsp;&nbsp;[步骤2：bolt部署](#步骤2：bolt部署) 
&nbsp;&nbsp;&nbsp;&nbsp;[附：onnx部署](#附：onnx部署) 
&nbsp;&nbsp;&nbsp;&nbsp;[常见问题](#常见问题)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[onnx-simplifier处理动态输入](#onnx-simplifier处理动态输入)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[onnx-simplifier报错](#onnx-simplifier报错)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[修改onnx模型输入维度](#修改onnx模型输入维度)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[给onnx模型增加输出节点](#给onnx模型增加输出节点)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[修改onnx模型节点参数](#修改onnx模型节点参数)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[bolt转换和推理添加新算子](#bolt转换和推理添加新算子)  

# 步骤1：简化onnx模型（推荐，可选）

  应用[onnx-simplifier](https://github.com/daquexian/onnx-simplifier)简化onnx模型，输入旧的onnx模型，输出简化后的onnx模型。


# 步骤2：bolt部署

- ### 下载和编译bolt项目

    请参考[INSTALL.md](INSTALL.md)。

- ### 使用X2bolt转换onnx格式模型到bolt格式模型

    详细请参考[USER_HANDBOOK.md](USER_HANDBOOK.md#model-conversion)或者 *--help*。
    
    * 示例：转换./example.onnx到./example_f32.bolt
    
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


# 附：onnx部署

- ### 使用onnx runtime推理

    [onnx runtime](https://github.com/microsoft/onnxruntime) 是onnx的推理引擎，bolt提供了一个简单的示例运行脚本[inference/engine/tools/onnx_tools/benchmark.py](../inference/engine/tools/onnx_tools/benchmark.py)，支持单输入/多输入，支持指定输入文件或输入文件夹。

    * 示例：全1输入测试
    
    ```bash
    python inference/engine/tools/onnx_tools/benchmark.py ./example.onnx
    ```
    
    * 示例：当前目录./下单txt文件输入，模型有1个输入，名字为input，input.txt是输入input的数据内容，用空格分割。
    
    ```bash
    python inference/engine/tools/onnx_tools/benchmark.py ./example.onnx ./input.txt
    ```
    
    * 示例：当前目录./下多txt文件输入，模型有2个输入，名字分别为input0和input1，input0.txt是输入input0的数据内容，用空格分割，input0.txt是输入input1的数据内容。
    
    ```bash
    python inference/engine/tools/onnx_tools/benchmark.py ./example.onnx ./
    ```
    
    * 示例：当前目录./下多txt文件输入，输入维度是动态的，模型有2个输入，名字分别为input0和input1，shape.txt记录实际推理输入维度，input0.txt是输入input0的数据内容，用空格分割，input0.txt是输入input1的数据内容。
    
    shape.txt内容：维度用空格分割
    
    ```bash
    input0 1 3 224 224
    input1 1 1 224 224
    ```
    
    ```bash
    python inference/engine/tools/onnx_tools/benchmark.py ./example.onnx ./
    ```


# 常见问题

- ### onnx-simplifier处理动态输入

    onnx-simplifier参数支持 *--dynamic-input-shape* 或 *--input-shape*，可以通过 *--hep* 查看。
    
    * 示例：通过 *--input-shape* 设置多输入大小。
    
    ```bash
    python -m onnxsim old.onnx new.onnx --input-shape input0:1,3,224,224 input1:1,1,224,224
    ```
    
- ### onnx-simplifier报错
    
    onnx-simplifier 的shape inference错误可以用过使用 *--skip-shape-inference* 解决。
    
    其它错误可以通过 *--help* 查看是否有解决措施。
    
- ### 修改onnx模型输入维度
    
    bolt提供了一个简单的示例运行脚本[inference/engine/tools/onnx_tools/change_input_dim.py](../inference/engine/tools/onnx_tools/change_input_dim.py)，通过shape.txt指定各个输入信息。
    
    * 示例：修改./old.onnx维度信息，保存到./new.onnx。old.onnx有两个输入，名字分别为input0和input1。
    
    shape.txt内容：维度用空格分割
    
    ```bash
    input0 1 3 224 224
    input1 1 1 224 224
    ```
    
    ```bash
    python3 inference/engine/tools/onnx_tools/change_input_dim.py ./old.onnx ./new.onnx ./shape.txt
    ```
    
- ### 给onnx模型增加输出节点
    
    bolt提供了一个简单的示例运行脚本[inference/engine/tools/onnx_tools/add_output.py](../inference/engine/tools/onnx_tools/add_output.py)。
    
    * 示例：output3和output4是新添加的两个输出名字，多输出用逗号分隔。
    
    ```bash
    python inference/engine/tools/onnx_tools/add_output.py old.onnx new.onnx output3,output4
    ```
    
- ### 修改onnx模型节点参数
    
    * 示例：修改层名为440和660的reshape算子参数
    
    ```bash
    #!/usr/bin/python
    
    import onnx
    import numpy as np
    
    model_name = "../old.onnx"
    model_name_new = "./new.onnx"
    layer_name = {"440", "660"}
    
    onnx_model = onnx.load(model_name)
    graph = onnx_model.graph
    for i in range(graph.initializer.__len__()):
        name = graph.initializer[i].name
        if (name in layer_name):
            graph.initializer.remove(graph.initializer[i])
            new_param = onnx.helper.make_tensor(name, onnx.TensorProto.INT64, [2], [-1, 512])
            graph.initializer.insert(i, new_param)
            print(graph.initializer[i])
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, model_name_new)
    ```
    
- ### bolt转换和推理添加新算子
    
    详细请参考[DEVELOPER.md](DEVELOPER.md#customize-models-with-unsupported-operators-step-by-step)。
