| Operator                  | Description |
| ------------------------- | ----------- |
| Abs                       | y = (x > 0) ? x : -x |
| ArgMax                    | max value index |
| AttentionMask             | transformer local attention mask |
| Attention                 | transformer global attention mask |
| BatchNorm                 | y = (x - mean) / sqrt(variance + eps) per channel |
| BatchToSpaceNd            | tensorflow batch_to_space function |
| BilateralSliceApply       | hdrnet BilateralSliceApply function, you can use it by using caffe or onnx self-defined operator, please refer to [inference/examples/bilateral_slice_apply/README.md](../inference/examples/bilateral_slice_apply/README.md) |
| Cast                      | change tensor data type |
| Ceil                      | y = ceil(x) |
| ChannelResize             | channel padding or channel cut |
| Check                     | element level compare, same as onnx Greater, GreaterOrEqual, Equal, LowerOrEqual, Lower |
| Clip                      | y = clip(x, min, max) |
| Concat                    | many tensors concat on some axis |
| ConstantOfShape           | allocate memory(not implement) |
| Constant                  | onnx constant |
| ConvertColor              | YUV_NV21 <-> RGB,BGR,RGBA,BGRA, you can use it by using caffe or onnx self-defined operator, please refer to [inference/examples/convert_color/README.md](../inference/examples/convert_color/README.md) |
| Convolution               | common 1D&2D&3D convolution, dilated 1D&2D&3D convolution, group 1D&2D&3D convolution, depthwise 1D&2D convolution |
| Copy                      | memory copy |
| Cos                       | y = cos(x) |
| Cum                       | prefix function, currently support cumsum, cumprod |
| Deconvolution             | 1D&2D deconvolution, onnx ConvTranspose |
| Depth2Space               | tensorflow depth_to_space function |
| DetectionOutput           | SSD caffe DetectionOutput |
| Dropout                   | dropout function |
| Einsum                    | same as onnx einsum |
| Elu                       | elu activation function |
| Eltwise                   | sum, min, max, mul(prod), sub, div elementwise operation |
| Embedding                 | Caffe embedding |
| ~~Equal~~                 | elementwise tensor compare, same as onnx equal, this also support tflite NOT_EQUAL, Equal is replaced with Check |
| Erf                       | erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t^2) dt|
| Expand                    | onnx expand |
| Exp                       | y = exp(x) |
| Flatten                   | same as onnx flatten |
| Floor                     | y = floor(x) |
| FullyConnected            | onnx Gemm, Linear |
| GAT                       | graph attention module |
| Gather                    | onnx gather, gather_elements, gatherND, also same as embedding |
| Gelu                      | gelu activation |
| GenerateProposals         | same as tf tf.image.generate_bounding_box_proposals |
| Greater                   | elementwise tensor compare, same as onnx greater |
| GridSample                | same as onnx grid_sample |
| HSigmoid                  | hard sigmoid, y = clip((x + 1) / 2, 0, 1)|
| HSwishNoDiv               | y = x * relu6(x + 3) |
| HSwish                    | y = x * relu6(x + 3) / 6 |
| InstanceNorm              | Instance Normalization |
| Jump                      | if statement for dynamic control flow |
| L2Normalization           | L2 Normalization |
| LayerNorm                 | layernorm |
| LeakyRelu                 | relu(scale != 0 when x < 0) |
| LogSoftmax                | log softmax |
| Log                       | y = log(x) |
| Matmul                    | matrix multiply |
| Mish                      | y = x * tanh(log(1 + e ^ x)) |
| MultiHeadAttention        | transformer multi-head attention |
| Neg                       | y = -x |
| NonMaxSuppression         | same as onnx non max suppression |
| NonZero                   | same as onnx non zero |
| Not                       | y = ! (x), same as onnx not |
| OneHot                    | same as onnx one hot |
| Pad                       | constant(0), reflect, edge, symmetric padding |
| Pooling                   | max, mean pooling |
| Power                     | y = (scale * x + shift) ^ pow |
| PreAllocatedMemory        | allocate memory |
| Prelu                     | prelu activation |
| PriorBox                  | SSD caffe PriorBox |
| QuantizeLinear            | int8 quantization |
| Random                    | random function, currently support uniform and normal random |
| Range                     | same as onnx range |
| Reciprocal                | same as onnx reciprocal, y = 1 / x |
| Reduction                 | sum, min, max, mean reduction |
| RelativePositionEmbedding | self-defined relative position embedding operator |
| RelativeShift             | self-defined relative shift operator |
| Relu6                     | y = relu6(x) |
| Relu                      | relu(scale = 0 when x < 0) |
| Repeat                    | do while loop for dynamic control flow |
| Reshape                   | change dimension |
| Resize                    | linear, nearest, cubic mode resize, same as onnx Resize, Upsample |
| RNN                       | LSTM, PLSTM, GRU, onnx LBR GRU, onnx Scan, also supports bi-direction |
| RoIAlign                  | same as onnx RoIAlign |
| Round                     | y = round(x) |
| Scale                     | y = alpha * x + beta per channel |
| Scatter                   | onnx scatter, scatter_elements, scatterND |
| Select                    | y = choice ? a : b, same as tflite select |
| Shape                     | get tensor shape |
| SharedWeight              | used to represent onnx/tflite operator input that is not generated by another operator |
| Sigmoid                   | sigmoid activation |
| Sign                      | y = sign(x) |
| Sin                       | y = sin(x) |
| Slice                     | caffe slice |
| SoftmaxWithLoss           | softmax with loss(not implement) |
| Softmax                   | y = exp(x - max(x)) / sum(exp(x - max(x))) |
| SoftPlus                  | y = log(1 + e ^ x)|
| Space2Depth               | tensorflow space_to_depth function |
| SpaceToBatchNd            | tensorflow space_to_batch function |
| Splice                    | Kaldi extract feature function, same as Gather |
| Split                     | same as onnx split |
| SqDiff                    | tflite squared difference |
| Squeeze                   | remove 1 dimension |
| Swish                     | y = x * exp(x) |
| TanH                      | y = tanh(x) |
| Tdnn                      | Kaldi tdnn operator(Splice + Linear) |
| TfSlice                   | onnx or tflite slice, strided slice |
| Tile                      | onnx tile |
| TopK                      | same as onnx topk |
| Transpose                 | transpose data, same as caffe permute |
| UnPooling                 | same as onnx unpooling |
| Unsqueeze                 | add 1 dimension |
| Where                     | same as onnx where|
| Yolov3DetectionOutput     | Yolov3 caffe detectionOutput |
