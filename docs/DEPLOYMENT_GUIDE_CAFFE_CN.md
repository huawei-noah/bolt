快速上手用Bolt部署Caffe模型

# 目录
---
&nbsp;&nbsp;&nbsp;&nbsp;[bolt部署](#bolt部署)
&nbsp;&nbsp;&nbsp;&nbsp;[常见问题](#常见问题)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[caffe添加自定义算子](#caffe添加自定义算子)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[bolt转换和推理添加新算子](#bolt转换和推理添加新算子)

# bolt部署
-  ### 下载和编译bolt项目

    请参考[INSTALL.md](INSTALL.md)。

-  ### 使用X2bolt转换caffe格式模型到bolt格式模型

    详细请参考[USER_HANDBOOK.md](USER_HANDBOOK.md#model-conversion)或者 *--help*。

    * 示例：转换./example.prototxt和./example.caffemodel到./example_f32.bolt

   ```bash
   ./X2bolt -d ./ -m example -i FP32
   ```

-  ###  通用benchmark测试

    详细请参考[USER_HANDBOOK.md](USER_HANDBOOK.md#model-inference)或者 *--help*。

    * 示例：CPU推理./example_f32.bolt，查看模型输入输出信息和推理时间。

    ```bash
    ./benchmark -m ./example_f32.bolt
    ```

-  ### C/C++/Java API开发

    详细请参考[DEVELOPER.md](DEVELOPER.md##use-out-of-the-box-api-to-infer-your-model)。


# 常见问题

-  ### caffe添加自定义算子

    caffe支持修改[third_party/proto/caffe.proto](third_party/proto/caffe.proto)添加自定义算子，可以参照开源项目或已有定义实现。

-  ### bolt转换和推理添加新算子

    详细请参考[DEVELOPER.md](DEVELOPER.md#customize-models-with-unsupported-operators-step-by-step)。
