from __future__ import absolute_import
from . import caffe_pb2 as pb


class LayerParameter():
    def __init__(self, name='', type='', top=(), bottom=()):
        self.layerParameter = pb.LayerParameter()
        self.layerName = self.layerParameter.name = name
        self.layerType = self.layerParameter.type = type
        self.outputBlobName = self.layerParameter.top
        self.outputBlobName.extend(top)
        self.inputBlobName = self.layerParameter.bottom
        self.inputBlobName.extend(bottom)

    def to_scalar(self, values, duplicate):
        if hasattr(values, '__iter__'):
            return values[0]
        else:
            return values

    def to_array(self, values):
        if hasattr(values, '__iter__'):
            return values
        else:
            return [values]

    def inner_product_param(self, num_output, weight_filler='xavier', bias_filler='constant', bias_term=True):
        if self.layerType != 'InnerProduct':
            raise TypeError('[ERROR] this is not InnerProduct')
        inner_product_param = pb.InnerProductParameter()
        inner_product_param.num_output = num_output
        inner_product_param.weight_filler.type = weight_filler
        inner_product_param.bias_term = bias_term
        if bias_term:
            inner_product_param.bias_filler.type = bias_filler
        self.layerParameter.inner_product_param.CopyFrom(inner_product_param)

    def input_param(self, shape):
        input_param = pb.InputParameter()
        blob = input_param.shape.add()
        for i in shape:
            blob.dim.append(i)
        self.layerParameter.input_param.CopyFrom(input_param)

    def embed_param(self, input_dim, embedding_dim, transpose=False, weight_filler='xavier', bias_filler='constant', bias_term=False):
        embed_param = pb.EmbedParameter()
        embed_param.input_dim = input_dim
        embed_param.num_output = embedding_dim
        embed_param.transpose = transpose
        embed_param.weight_filler.type = weight_filler
        embed_param.bias_term = bias_term
        if bias_term:
            embed_param.bias_filler.type = bias_filler
        self.layerParameter.embed_param.CopyFrom(embed_param)

    def reshape_param(self, shape, axis=0, num_axes=-1):
        reshape_param = pb.ReshapeParameter()
        for i in shape:
            reshape_param.shape.dim.append(i)
        reshape_param.axis = axis
        reshape_param.num_axes = num_axes
        self.layerParameter.reshape_param.CopyFrom(reshape_param)

    def slice_param(self, axis, slice_point):
        slice_param = pb.SliceParameter()
        slice_param.axis = axis
        for i in slice_point:
            slice_param.slice_point.append(i)
        self.layerParameter.slice_param.CopyFrom(slice_param)

    def convolution_param(self, num_output, kernel_size, stride=(1), pad=(0,),
                   weight_filler_type='xavier', bias_filler_type='constant',
                   bias_term=True, dilation=None,groups=None):
        if self.layerType not in ['Convolution','Deconvolution']:
            raise TypeError('[ERROR] this is not Convolution or Deconvolution')
        convolution_param=pb.ConvolutionParameter()
        convolution_param.num_output = num_output
        convolution_param.kernel_size.extend(to_array(kernel_size))
        convolution_param.stride.extend(to_array(stride))
        convolution_param.pad.extend(to_array(pad))
        convolution_param.bias_term = bias_term
        convolution_param.weight_filler.type=weight_filler_type
        if bias_term:
            convolution_param.bias_filler.type = bias_filler_type
        if dilation:
            convolution_param.dilation.extend(to_array(dilation))
        if groups:
            convolution_param.group = groups
        self.layerParameter.convolution_param.CopyFrom(convolution_param)

    def pool_param(self, type='MAX', kernel_size=2, stride=2, pad=None, ceil_mode=True):
        pool_param = pb.PoolingParameter()
        pool_param.pool = pool_param.PoolMethod.Value(type)
        pool_param.kernel_size = to_scalar(kernel_size)
        pool_param.stride = to_scalar(stride)
        pool_param.ceil_mode = ceil_mode
        if pad:
            if isinstance(pad, tuple):
                pool_param.pad_h = pad[0]
                pool_param.pad_w = pad[1]
            else:
                pool_param.pad = pad
        self.layerParameter.pooling_param.CopyFrom(pool_param)

    def batch_norm_param(self, use_global_stats=0, moving_average_fraction=None, eps=None):
        batch_norm_param = pb.BatchNormParameter()
        batch_norm_param.use_global_stats = use_global_stats
        if moving_average_fraction:
            batch_norm_param.moving_average_fraction = moving_average_fraction
        if eps:
            batch_norm_param.eps = eps
        self.layerParameter.batch_norm_param.CopyFrom(batch_norm_param)

    def eltwise_param(self, operation):
        eltwise_param = pb.EltwiseParameter()
        eltwise_param.operation = operation
        self.layerParameter.eltwise_param.CopyFrom(eltwise_param)

    def argmax_param(self, axis):
        argmax_param = pb.ArgMaxParameter()
        argmax_param.axis = axis
        self.layerParameter.argmax_param.CopyFrom(argmax_param)

    def add_data(self, *datas):
        del self.layerParameter.blobs[:]
        for data in datas:
            new_blob = self.layerParameter.blobs.add()
            for dim in data.shape:
                new_blob.shape.dim.append(dim)
            new_blob.data.extend(data.flatten().astype(float))

    def set_params_by_dict(self,dic):
        pass

    def copy_from(self,layer_param):
        pass

    # caffe.proto
    #    optional TransposeParameter transpose_param = 100010;
    #    message TransposeParameter {
    #      optional BlobShape dim = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "layer0_out_att_self_query_t"
    #       type: "Transpose"
    #       bottom: "layer0_out_att_self_query_r"
    #       top: "layer0_out_att_self_query_t"
    #       transpose_param {
    #         dim {
    #           dim: 0
    #           dim: 2
    #           dim: 1
    #           dim: 3
    #         }
    #       }
    #     }
    def transpose_param(self, dim):
        transpose_param = pb.TransposeParameter()
        for i in dim:
            transpose_param.dim.dim.append(i)
        self.layerParameter.transpose_param.CopyFrom(transpose_param)

    def convert_data_type(self, type_str):
        if (type_str == "FLOAT32"):
            type = 0;
        elif (type_str == "INT32"):
            type = 1;
        elif (type_str == "UINT32"):
            type = 2;
        else:
            print("[ERROR] data type %s NOT_SUPPORTED" % (type_str))
            exit(0)
        return type

    # caffe.proto
    #    optional SharedWeightParameter shared_weight_param = 100011;
    #    message SharedWeightParameter {
    #      enum DataType {
    #        FLOAT32 = 0;
    #        INT32 = 1;
    #        UINT32 = 2;
    #      }
    #      optional DataType data_type = 1;
    #      repeated BlobShape shape = 2;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "weight_0"
    #       type: "Weight"
    #       bottom: ""
    #       top: "weight_0"
    #       shared_weight_param {
    #         data_type: 0
    #         shape {
    #           dim: 100
    #           dim: 200
    #         }
    #       }
    #     }
    def weight_param(self, shape, data_type):
        shared_weight_param = pb.SharedWeightParameter()
        shared_weight_param.data_type = self.convert_data_type(data_type)
        for i in shape:
            shared_weight_param.shape.dim.append(i)
        self.layerParameter.shared_weight_param.CopyFrom(shared_weight_param)

    # caffe.proto
    #    optional PreAllocatedMemory preallocated_memory_param = 100012;
    #    message PreAllocatedMemory {
    #      enum DataType {
    #        FLOAT32 = 0;
    #        INT32 = 1;
    #        UINT32 = 2;
    #      }
    #      optional DataType data_type = 1;
    #      optional BlobShape shape = 2;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "memory_0"
    #       type: "Memory"
    #       bottom: ""
    #       top: "memory_0"
    #       preallocated_memory_param {
    #         data_type: 0
    #         shape {
    #           dim: 100
    #           dim: 200
    #         }
    #       }
    #     }
    def memory_param(self, shape, data_type):
        preallocated_memory_param = pb.PreAllocatedMemoryParameter()
        preallocated_memory_param.data_type = self.convert_data_type(data_type)
        for i in shape:
            preallocated_memory_param.shape.dim.append(i)
        self.layerParameter.preallocated_memory_param.CopyFrom(preallocated_memory_param)

    # caffe.proto
    #    optional MultiplyParameter multiply_param = 100013;
    #    message MultiplyParameter {
    #      optional float scale = 1 [default = 1];
    #      optional float bias = 2 [default = 0];
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "layer0_out_att_self_qks"
    #       type: "Multiply"
    #       bottom: "layer0_out_att_self_qk"
    #       top: "layer0_out_att_self_qks"
    #       multiply_param {
    #         scale: 0.125
    #         bias: 0
    #       }
    #     }
    def multiply_param(self, scale, bias):
        multiply_param = pb.MultiplyParameter()
        multiply_param.scale = scale
        multiply_param.bias = bias
        self.layerParameter.multiply_param.CopyFrom(multiply_param)

    # caffe.proto
    #    optional AttentionParameter attention_param = 100014;
    #    message AttentionParameter {
    #      optional uint32 num_heads = 1;
    #      optional uint32 from_sequence_length = 2;
    #      optional uint32 to_sequence_length = 3;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "attention"
    #       type: "Attention"
    #       bottom: "input_mask"
    #       top: "attention"
    #       attention_param {
    #         num_heads: 12
    #         from_sequence_length: 120
    #         to_sequence_length: 120
    #       }
    #     }
    def attention_param(self, num_heads, from_seq_length, to_seq_length):
        attention_param = pb.AttentionParameter()
        attention_param.num_heads = num_heads
        attention_param.from_sequence_length = from_seq_length
        attention_param.to_sequence_length = to_seq_length
        self.layerParameter.attention_param.CopyFrom(attention_param)

    # caffe.proto
    #    optional SqueezeParameter squeeze_param = 100015;
    #    message SqueezeParameter {
    #      optional int axis = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "squeeze"
    #       type: "Squeeze"
    #       bottom: "squeeze_input"
    #       top: "squeeze_output"
    #       squeeze_param {
    #         axis: 1
    #       }
    #     }
    def squeeze_param(self, axis):
        squeeze_param = pb.SqueezeParameter()
        squeeze_param.axis = axis
        self.layerParameter.squeeze_param.CopyFrom(squeeze_param)


    # caffe.proto
    #    optional AxisMeanParameter axis_mean_param = 100016;
    #    message AxisMeanParameter {
    #      optional int axis = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "axis_mean"
    #       type: "AxisMean"
    #       bottom: "axis_mean_input"
    #       top: "axis_mean_output"
    #       axis_mean_param {
    #         axis: 1
    #       }
    #     } 
    def axis_mean_param(self, axis):
        axis_mean_param = pb.AxisMeanParameter()
        axis_mean_param.axis = axis
        self.layerParameter.axisMean_param.CopyFrom(axis_mean_param)

    # caffe.proto
    #    optional UnsqueezeParameter unsqueeze_param = 100017;
    #    message UnsqueezeParameter {
    #      optional int axis = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "unsqueeze"
    #       type: "Unsqueeze"
    #       bottom: "unsqueeze_input"
    #       top: "unsqueeze_output"
    #       unsqueeze_param {
    #         axis: 1
    #       }
    #     }
    def unsqueeze_param(self, axis):
        unsqueeze_param = pb.UnsqueezeParameter()
        unsqueeze_param.axis = axis
        self.layerParameter.unsqueeze_param.CopyFrom(unsqueeze_param)

    # caffe.proto
    #    optional RepeatParameter repeat_param = 100018;
    #    message RepeatParameter {
    #      optional int axis = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "repeat"
    #       type: "Repeat"
    #       bottom: "repeat_input"
    #       top: "repeat_output"
    #       repeat_param {
    #         loops: 1
    #       }
    #     }
    def repeat_param(self, loops):
        repeat_param = pb.RepeatParameter()
        repeat_param.loops = loops
        self.layerParameter.repeat_param.CopyFrom(repeat_param)


    # caffe.proto
    #    optional CheckParameter check_param = 100019;
    #    message CheckParameter {
    #      optional int axis = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "check"
    #       type: "Check"
    #       bottom: "check_input"
    #       top: "check_output"
    #       check_param {
    #         operation: 0
    #       }
    #     }
    def check_param(self, operation):
        check_param = pb.CheckParameter()
        if operation == "equal":
            check_param.operation = 0
        else:
            print("[ERROR] unsupported check condition %s" % (operation))
            exit(0)
        self.layerParameter.check_param.CopyFrom(check_param)


    # caffe.proto
    #    optional CopyParameter copy_param = 100020;
    #    message CopyParameter {
    #      optional uint32 src_batch_stride = 1;
    #      optional uint32 src_stride = 2;
    #      optional uint32 src_offset = 3;
    #      optional uint32 dst_batch_stride = 4;
    #      optional uint32 dst_stride = 5;
    #      optional uint32 dst_offset = 6;
    #      optional uint32 length = 7;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "copy"
    #       type: "Copy"
    #       bottom: "copy_input"
    #       bottom: "copy_input_index"
    #       bottom: "copy_output_index"
    #       top: "copy_output"
    #       copy_param {
    #         src_batch_stride: 10
    #         src_stride: 1
    #         src_offset: 0
    #         dst_batch_stride: 10
    #         dst_stride: 1
    #         dst_offset: 0
    #         length: 1
    #       }
    #     }
    def copy_param(self, src_batch_stride, src_stride, src_offset,
                   dst_batch_stride, dst_stride, dst_offset,
                   length):
        copy_param = pb.CopyParameter()
        copy_param.src_batch_stride = src_batch_stride
        copy_param.src_stride = src_stride
        copy_param.src_offset = src_offset
        copy_param.dst_batch_stride = dst_batch_stride
        copy_param.dst_stride = dst_stride
        copy_param.dst_offset = dst_offset
        copy_param.length = length
        self.layerParameter.copy_param.CopyFrom(copy_param)

    # caffe.proto
    #    optional LSTMParameter lstm_param = 100020;
    #    message LSTMParameter {
    #      optional uint32 num_output = 1;
    #      optional int32 steps = 2 [default = -1];
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "lstm_output"
    #       type: "LSTM"
    #       bottom: "lstm_input"
    #       top: "lstm_output"
    #       lstm_param {
    #         num_output: 1024
    #         steps: -1
    #       }
    #     }
    def lstm_param(self, num_output, steps):
        lstm_param = pb.LSTMParameter()
        lstm_param.num_output = num_output
        lstm_param.steps = steps
        self.layerParameter.lstm_param.CopyFrom(lstm_param)

    # caffe.proto
    #    optional MatMulParameter matmul_param = 100020;
    #    message MatMulParameter {
    #      optional bool transpose_a = 1 [default = false]; // Whether to use transpose matrixA
    #      optional bool transpose_b = 2 [default = false]; // Whether to use transpose matrixB
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "matrixC"
    #       type: "MatMul"
    #       bottom: "matrixA"
    #       bottom: "matrixB"
    #       top: "matrixC"
    #       matmul_param {
    #         transpose_a: false
    #         transpose_b: false
    #       }
    #     }
    def matmul_param(self, transpose_a, transpose_b):
        matmul_param = pb.MatMulParameter()
        matmul_param.transpose_a = transpose_a
        matmul_param.transpose_b = transpose_b
        self.layerParameter.matmul_param.CopyFrom(matmul_param)
