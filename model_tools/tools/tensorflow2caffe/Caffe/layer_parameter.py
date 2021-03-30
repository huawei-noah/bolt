from __future__ import absolute_import
from . import caffe_pb2 as pb
import numpy as np


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

    def embed_param(self, input_dim, embedding_dim,
            transpose=False, weight_filler='xavier', bias_filler='constant', bias_term=False):
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

    def tile_param(self, axis=1, tiles=2):
        tile_param = pb.TileParameter()
        tile_param.axis = axis
        tile_param.tiles = tiles
        self.layerParameter.tile_param.CopyFrom(tile_param)

    def slice_param(self, axis, slice_point):
        slice_param = pb.SliceParameter()
        slice_param.axis = axis
        for i in slice_point:
            slice_param.slice_point.append(i)
        self.layerParameter.slice_param.CopyFrom(slice_param)

    def convolution_param(self, num_output, kernel_size, stride=(1), pad=(0,),
                   bias_term=True, dilation=None, groups=None,
                   weight_filler_type='xavier', bias_filler_type='constant'):
        if self.layerType not in ['Convolution','Deconvolution']:
            raise TypeError('[ERROR] this is not Convolution or Deconvolution')
        convolution_param=pb.ConvolutionParameter()
        convolution_param.num_output = num_output
        convolution_param.kernel_size.extend(self.to_array(kernel_size))
        convolution_param.stride.extend(self.to_array(stride))
        convolution_param.pad.extend(self.to_array(pad))
        convolution_param.bias_term = bias_term
        convolution_param.weight_filler.type=weight_filler_type
        if bias_term:
            convolution_param.bias_filler.type = bias_filler_type
        if dilation:
            convolution_param.dilation.extend(self.to_array(dilation))
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

    def batch_norm_param(self, use_global_stats=0, moving_average_fraction=None, axis=1, eps=None):
        batch_norm_param = pb.BatchNormParameter()
        batch_norm_param.use_global_stats = use_global_stats
        if moving_average_fraction:
            batch_norm_param.moving_average_fraction = moving_average_fraction
        if eps:
            batch_norm_param.eps = eps
        batch_norm_param.axis = axis
        self.layerParameter.batch_norm_param.CopyFrom(batch_norm_param)

    def scale_param(self, axis=1, num_axes=1, filler='xavier', bias_term=False, bias_filler='constant'):
        scale_param = pb.ScaleParameter()
        scale_param.axis = axis
        scale_param.num_axes = num_axes
        scale_param.filler.type = filler
        scale_param.bias_term = bias_term
        scale_param.bias_filler.type = bias_filler
        self.layerParameter.scale_param.CopyFrom(scale_param)

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

    def permute_param(self, dim):
        permute_param = pb.PermuteParameter()
        for i in dim:
            permute_param.order.append(i)
        self.layerParameter.permute_param.CopyFrom(permute_param)

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

    def power_param(self, scale, shift, power):
        power_param = pb.PowerParameter()
        power_param.scale = scale
        power_param.shift = shift
        power_param.power = power
        self.layerParameter.power_param.CopyFrom(power_param)

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
    #    optional ReductionParameter reduction_param = 100016;
    #    message ReductionParameter {
    #      enum ReductionOp {
    #        SUM = 1;
    #        ASUM = 2;
    #        SUMSQ = 3;
    #        MEAN = 4;
    #      }
    #      optional ReductionOp operation = 1 [default = SUM]; // reduction operation
    #      optional int axis = 2 [default = 0];
    #      optional float coeff = 3 [default = 1.0]; // coefficient for output
    #      optional bool keep_dim = 4 [default = false];
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "reduction"
    #       type: "Reduction"
    #       bottom: "reduction_input"
    #       top: "reduction_output"
    #       reduction_param {
    #         operation: SUM
    #         axis: 1
    #         coeff: 1.0
    #         keep_dim: false
    #       }
    #     } 
    def reduction_param(self, operation, axis, keep_dim, coeff=1.0):
        reduction_param = pb.ReductionParameter()
        reduction_param.operation = operation
        reduction_param.axis = axis
        reduction_param.coeff = coeff
        reduction_param.keep_dim = keep_dim
        self.layerParameter.reduction_param.CopyFrom(reduction_param)

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
    #      optional int loops = 1;
    #      optional int axis = 2;
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
    #         loops: -1
    #       }
    #     }
    def repeat_param(self, loops, axis):
        repeat_param = pb.RepeatParameter()
        repeat_param.loops = loops
        repeat_param.axis = axis
        self.layerParameter.repeat_param.CopyFrom(repeat_param)


    # caffe.proto
    #    optional CheckParameter check_param = 100019;
    #    message CheckParameter {
    #      enum CheckOp {
    #        EQUAL = 0;
    #        GREATEQUAL = 0;
    #      }
    #      optional CheckOp operation = 1;
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
        elif operation == "great":
            check_param.operation = 1
        elif operation == "greatequal":
            check_param.operation = 2
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
    #    optional LSTMParameter lstm_param = 100021;
    #    message LSTMParameter {
    #      optional uint32 num_output = 1;
    #      optional int32 steps = 2 [default = -1];
    #      optional int32 num_proj = 3 [default = 0];
    #      optional float zoneout_cell = 4 [default = 0];
    #      optional float zoneout_output = 5 [default = 0];
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
    #         num_proj: 0
    #         zoneout_cell: 0
    #         zoneout_output: 0
    #       }
    #     }
    def lstm_param(self, num_output, steps, num_proj, zoneout_cell, zoneout_output):
        lstm_param = pb.LSTMParameter()
        lstm_param.num_output = num_output
        lstm_param.steps = steps
        lstm_param.num_proj = num_proj
        lstm_param.zoneout_cell = zoneout_cell
        lstm_param.zoneout_output = zoneout_output
        self.layerParameter.lstm_param.CopyFrom(lstm_param)

    # caffe.proto
    #    optional MatMulParameter matmul_param = 100022;
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

    # caffe.proto
    #    optional ConcatParameter concat_param = 100023;
    #    message ConcatParameter {
    #      optional int axis = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "concat"
    #       type: "Concat"
    #       bottom: "concat_input"
    #       top: "concat_output"
    #       concat_param {
    #         axis: 1
    #       }
    #     }
    def concat_param(self, axis):
        concat_param = pb.ConcatParameter()
        concat_param.axis = axis
        self.layerParameter.concat_param.CopyFrom(concat_param)

    # caffe.proto
    #     optional RelativePositionEmbedParameter relative_position_embed_param = 100024;
    #     message RelativePositionEmbedParameter {
    #       optional uint32 num_output = 1; // The number of outputs for the layer
    #       optional uint32 input_dim = 2;
    #       optional bool transpose = 3 [default = false]; // Whether to use transpose dict
    #       optional int32 axis = 4 [default = 1];
    #     }
    #
    # prototxt example
    #     layer {
    #       name: "relative_position_embedding"
    #       type: "RelativePositionEmbed"
    #       bottom: ""
    #       top: "relative_embedding"
    #       concat_param {
    #         num_output: 512
    #         input_dim: 10
    #         transpose: false
    #         axis: 1
    #       }
    #     }
    def relative_position_embed_param(self, input_dim, embedding_dim,
            transpose=False, axis=1, weight_filler='xavier', bias_filler='constant', bias_term=False):
        embed_param = pb.RelativePositionEmbedParameter()
        embed_param.input_dim = input_dim
        embed_param.num_output = embedding_dim
        embed_param.transpose = transpose
        embed_param.axis = axis
        embed_param.weight_filler.type = weight_filler
        embed_param.bias_term = bias_term
        if bias_term:
            embed_param.bias_filler.type = bias_filler
        self.layerParameter.relative_position_embed_param.CopyFrom(embed_param)

    # caffe.proto
    #    optional AttentionMaskParameter attention_mask_param = 100025;
    #    message AttentionMaskParameter {
    #      optional int32 attention_length = 1 [default = -1];
    #      optional bool same_length = 2 [default = false];
    #      optional float mask = 3 [default = 10000];
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "attention_mask"
    #       type: "AttentionMask"
    #       bottom: "attention_mask_input"
    #       top: "attention_mask_output"
    #       attention_mask_param {
    #         attention_length: -1
    #         same_langth: false
    #         mask: 10000
    #       }
    #     }
    def attention_mask_param(self, attention_length, same_length, mask):
        attention_mask_param = pb.AttentionMaskParameter()
        attention_mask_param.attention_length = attention_length
        attention_mask_param.same_length = same_length
        attention_mask_param.mask = mask
        self.layerParameter.attention_mask_param.CopyFrom(attention_mask_param)

    # caffe.proto
    #    optional RelativeShiftParameter relative_shift_param = 100026;
    #    message RelativeShiftParameter {
    #      optional int32 axis = 1 [default = 1];
    #      optional int32 shift_length = 2 [default = 1];
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "relative_shift"
    #       type: "RelativeShift"
    #       bottom: "relative_shift_input"
    #       top: "relative_shift_output"
    #       relative_shift_param {
    #         axis: 2
    #         shift_length: 1
    #       }
    #     }
    def relative_shift_param(self, axis, shift_length):
        relative_shift_param = pb.RelativeShiftParameter()
        relative_shift_param.axis = axis
        relative_shift_param.shift_length = shift_length
        self.layerParameter.relative_shift_param.CopyFrom(relative_shift_param)

    # caffe.proto
    #    optional PaddingParameter padding_param = 100026;
    #    message PaddingParameter {
    #      optional int32 axis = 1 [default = 1];
    #      optional int32 shift_length = 2 [default = 1];
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "padding"
    #       type: "Padding"
    #       bottom: "padding_input"
    #       top: "padding_output"
    #       padding_param {
    #         shape {
    #           dim: 1
    #           dim: 1
    #           dim: 1
    #           dim: 1
    #         }
    #         value {
    #           dim: 0
    #           dim: 0
    #           dim: 0
    #           dim: 0
    #         }
    #       }
    #     }
    def padding_param(self, shape, value=None):
        padding_param = pb.PaddingParameter()
        for i in np.array(shape).flatten():
            padding_param.shape.append(i)
        if (value is not None):
            for i in np.array(value).flatten():
                padding_param.value.append(i)
        self.layerParameter.padding_param.CopyFrom(padding_param)

    # caffe.proto
    #    optional SoftmaxParameter softmax_param = 100027;
    #    message SoftmaxParameter {
    #      optional int axis = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "softmax"
    #       type: "Softmax"
    #       bottom: "softmax_input"
    #       top: "softmax_output"
    #       softmax_param {
    #         axis: 1
    #       }
    #     }
    def softmax_param(self, axis):
        softmax_param = pb.SoftmaxParameter()
        softmax_param.axis = axis
        self.layerParameter.softmax_param.CopyFrom(softmax_param)

    # caffe.proto
    #    optional ClipParameter clip_param = 100027;
    #    message ClipParameter {
    #      required float min = 1;
    #      required float max = 2;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "clip"
    #       type: "Clip"
    #       bottom: "clip_input"
    #       top: "clip_output"
    #       clip_param {
    #         min: 0
    #         max: 1
    #       }
    #     }
    def clip_param(self, min_value, max_value):
        clip_param = pb.ClipParameter()
        clip_param.min = min_value
        clip_param.max = max_value
        self.layerParameter.clip_param.CopyFrom(clip_param)

    def exp_param(self, base, scale, shift):
        exp_param = pb.ExpParameter()
        exp_param.base = base
        exp_param.scale = scale
        exp_param.shift = shift
        self.layerParameter.exp_param.CopyFrom(exp_param)
