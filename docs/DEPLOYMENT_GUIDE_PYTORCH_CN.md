快速上手用Bolt部署Pytorch模型

# 目录
---
&nbsp;&nbsp;&nbsp;&nbsp;[步骤1：从pytorch格式模型导出成onnx格式模型](#步骤1：从pytorch格式模型导出成onnx格式模型) 
&nbsp;&nbsp;&nbsp;&nbsp;[步骤2：用bolt部署onnx模型](#步骤2：用bolt部署onnx模型) 
&nbsp;&nbsp;&nbsp;&nbsp;[常见问题](#常见问题) 

# 步骤1：从pytorch格式模型导出成onnx格式模型

  加载pytorch格式模型，使用[torch.onnx.export](https://pytorch.org/docs/stable/onnx.html)导出成onnx模型文件。
    
  * 示例：简单的单输入单输出固定大小
    
    ```bash
    #!/usr/bin/python
    import io
    import torch
    import torch.onnx
    from models.Model import Model
    
    model = Model()
    pthfile = 'checkpoint.pth'
    loaded_model = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(loaded_model['state_dict'])
    
    dummy_input = torch.randn(1, 3, 64, 64)
    input_names = [ "input" ]
    output_names = [ "output" ]
    torch.onnx.export(model, dummy_input, "example.onnx", verbose=True, input_names=input_names, output_names=output_names)
    ```


# 步骤2：用bolt部署onnx模型

   详细请参考[DEPLOYMENT_GUIDE_ONNX_CN.md](DEPLOYMENT_GUIDE_ONNX_CN.md)。


# 常见问题

- ### 导出onnx使用动态输入
    
    torch.onnx.export支持设置动态输入维度，可以查阅相关手册。
    
- ### pytorch导出onnx遇到不支持算子
    
    onnx支持pytorch算子有限，如果遇到不支持算子导出，可以参考示例添加自定义算子，自定义算子只支持onnx存储，不支持onnx runtime推理。
    
    * 示例：inference/engine/tools/onnx_tools/custom_ops文件夹
        * [添加GridSample算子转换](../inference/engine/tools/onnx_tools/custom_ops/GridSample.py)
        * [添加PixelAdaConv算子转换](inference/engine/tools/onnx_tools/custom_ops/PixAdaConvNet.py)
