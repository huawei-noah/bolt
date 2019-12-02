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

    def embed_param(self, input_dim, embedding_dim, weight_filler='xavier', bias_filler='constant', bias_term=False):
        embed_param = pb.EmbedParameter()
        embed_param.input_dim = input_dim
        embed_param.num_output = embedding_dim
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

    # caffe.proto
    #    optional MultiplyParameter multiply_param = 100011;
    #    message MultiplyParameter {
    #      optional float scale = 1;
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
    #       }
    #     }
    def multiply_param(self, scale):
        multiply_param = pb.MultiplyParameter()
        multiply_param.scale = scale
        self.layerParameter.multiply_param.CopyFrom(multiply_param)

    # caffe.proto
    #    optional AttentionParameter attention_param = 100012;
    #    message AttentionParameter {
    #      optional int num_attention = 1;
    #    }
    #
    # prototxt example
    #     layer {
    #       name: "attention"
    #       type: "Attention"
    #       bottom: "input_mask"
    #       top: "attention"
    #       attention_param {
    #         num_attention: 12
    #       }
    #     }
    def attention_param(self, num_attention):
        attention_param = pb.AttentionParameter()
        attention_param.num_attention = num_attention
        self.layerParameter.attention_param.CopyFrom(attention_param)
