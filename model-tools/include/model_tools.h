// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_MODEL_TOOLS
#define _H_MODEL_TOOLS

#include "error.h"
#include "type.h"
#include "tensor_desc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NAME_LEN 128

    typedef enum{
        F32_to_F32,
        F32_to_F16,
        F32_to_I8
    } DataConvertType;

    // TODO: please add OperatorType and OperatorTypeName at the same time
    typedef enum {
        OT_Conv,
        OT_FC,
        OT_Pooling,
        OT_Relu,
        OT_Relu6,
        OT_HSwish,
        OT_HSigmoid,
        OT_Eltwise,
        OT_Softmax,
        OT_Concat,
        OT_MaxOut,
        OT_BatchNorm,
        OT_Sigmoid,
        OT_Scale,
        OT_Clip,
        OT_LSTM,
        OT_Embedding,
        OT_SoftmaxWithLoss,
        OT_Pad,
        OT_Gelu,
        OT_TanH,
        OT_LayerNorm, 
        OT_MatMul,
        OT_Multiply,
        OT_Reshape,
        OT_Slice,
        OT_Transpose,
        OT_Attention,
        OT_Input,
        OT_Squeeze,
        OT_Gather,
        OT_Unsqueeze,
        OT_Upsample,
        OT_Cast,
        OT_Logistic,
        OT_ResizeBilinear,
        OT_None
    } OperatorType;

    inline const char * const *OperatorTypeName() {
        static const char * const names[] = {
            "OT_Conv",
            "OT_FC",
            "OT_Pooling",
            "OT_Relu",
            "OT_Relu6",
            "OT_HSwish",
            "OT_HSigmoid",
            "OT_Eltwise",
            "OT_Softmax",
            "OT_Concat",
            "OT_MaxOut",
            "OT_BatchNorm",
            "OT_Sigmoid",
            "OT_Scale",
            "OT_Clip",
            "OT_LSTM",
            "OT_Embedding",
            "OT_SoftmaxWithLoss",
            "OT_Pad",
            "OT_Gelu",
            "OT_TanH",
            "OT_LayerNorm",
            "OT_MatMul",
            "OT_Multiply",
            "OT_Reshape", 
            "OT_Slice",
            "OT_Transpose",
            "OT_Attention",
            "OT_Input",
            "OT_Squeeze",
            "OT_Gather",
            "OT_Unsqueeze",
            "OT_Upsample",
            "OT_Cast",
            "OT_Logistic",
            "OT_ResizeBilinear",
            "OT_None"
        };
        return names;
    }

    typedef struct {    // 20191119
        int gather_axis;
    } GatherParamSpec;

    typedef struct {
        int unsqueeze_axes[8];
        int axes_num;
    } UnsqueezeParamSpec;

    typedef struct {
        char upsample_mode[NAME_LEN];
    } UpsampleParamSpec;

    typedef struct {
        int cast_to;
    } CastParamSpec;

    typedef struct{
        float neg_slope;
    }ReLUParamSpec;

    typedef struct{
        int coeff_size;
        float* coeff_values;
    }EltwiseSumSpec;

    typedef struct{
        EltwiseMode elt_mode;
        EltwiseSumSpec elt_sum_spec;   // only sum mode need to feed it
    }EltwiseParamSpec;

    typedef struct{
        float min;
        float max;
    }ClipParamSpec;

    typedef union{
        ReLUParamSpec relu_spec;
        ClipParamSpec clip_spec;
    }ActivationSpec;

    typedef struct {
        U32 num_kernels;
        U32 kernel_size;
        U32 stride;
        U32 padding;
        U32 group;
        U32 dilation;
        ConvolutionMode convolution_type;
        ActivationMode dw_activation_type;
        ActivationMode pw_activation_type;
        ActivationSpec activation_spec;
    } ConvolutionParamSpec;

    typedef struct {
        U32 kernel_size;
        U32 stride;
        U32 padding;
        RoundMode rm;
        PoolingMode mode;
    } PoolingParamSpec;

    typedef struct {
        U32 num_outputs;
    } FullyConnectedParamSpec;

    typedef struct{
        F32 eps;     // from prototxt
    } BatchNormParamSpec;

    typedef struct {
        U32 top;
        U32 bottom;
        U32 left;
        U32 right;
        F32 constant_value;
        PadMode pad_mode;
    } PadParamSpec;

    typedef struct {
        U32 input_dim;
        U32 num_output;
        bool bias_term;
    } EmbedParamSpec;

    typedef struct {
        float scale;
    } MultiplyParamSpec;

    typedef struct {
        I32 shape_dims[8];
        I32 shape_size;
        I32 axis;
        I32 num_axes;
    } ReshapeParamSpec;

    typedef struct {
        U32 slice_points[8];
        U32 slice_size;
        U32 axis;
    } SliceParamSpec;

    typedef struct {
        U32 trans_dims[8];
        U32 trans_size;
    } TransposeParamSpec;

    typedef struct {
        int num_attention;
    } AttentionParamSpec;

    typedef struct {
        U32 num_output;
    } LstmParamSpec;

    typedef union {
        ConvolutionParamSpec conv_param_spec;
        PoolingParamSpec pooling_param_spec;
        FullyConnectedParamSpec ip_param_spec;
        BatchNormParamSpec bn_param_spec;
        EltwiseParamSpec eltwise_param_spec;
        ReLUParamSpec relu_spec;
        ClipParamSpec clip_spec;
        PadParamSpec pad_spec;
        EmbedParamSpec embed_spec;
        MultiplyParamSpec multiply_spec;
        ReshapeParamSpec reshape_spec;
        SliceParamSpec slice_spec;
        TransposeParamSpec transpose_spec;
        AttentionParamSpec attention_spec;
        LstmParamSpec lstm_spec;
        GatherParamSpec gather_spec;
        UnsqueezeParamSpec unsqueeze_spec;
        UpsampleParamSpec upsample_spec;
        CastParamSpec cast_spec;
    } ParameterSpec;

    typedef struct {
        I8 name[NAME_LEN];
        OperatorType type;
        U32 num_inputs;
        I8 **input_tensors_name;
        U32 num_outputs;
        I8 **output_tensors_name;
        ParameterSpec ps;
    } OperatorSpec;

    typedef struct {
        I8 op_name[NAME_LEN];
        DataType mdt;
        U32 bytes_of_weight;
        U8* weight;
        U32 bytes_of_vec;
        U8* vec;
    } WeightSpec;

    typedef struct {
        I8 op[NAME_LEN];
        U32 num_inputs;
        I8 **input_op_names;
        U32 num_outputs;
        I8 **output_op_names;
    } OperatorRelationshipMapEntry;

    typedef struct {
        I32 version;
        I32 magic_number;

        I8 model_name[NAME_LEN];
        DataType dt;

        I32 num_inputs;
        I8 **input_names;
        TensorDesc *input_dims;

        I32 num_outputs;
        I8 **output_names;

        I32 num_operator_specs;
        OperatorSpec *ops;

        I32 num_weight_specs;
        WeightSpec *ws;

        I32 num_op_tensor_entries;
        OperatorRelationshipMapEntry *op_relationship_entries;
    } ModelSpec;
/****
 * @breif
 *
 * @param dir
 * @param mfn model file name without extension
 *
 **/
    inline I32 mt_version(){
        static const I32 version = 190930;
        return version;
    }

    inline I32 mt_magic_number(){
        static const I32 magic_number = 1141119;   // bolt
        return magic_number;
    }

    //you must invoke this before use md
    EE mt_create_model(ModelSpec* md);
    EE mt_load(CI8* dir, CI8* mfn, ModelSpec* md);
    EE mt_store(CI8* dir, CI8* mfn, const ModelSpec* md);
    //you must invoke this before exit to clean up resource usage
    EE mt_destroy_model(ModelSpec* md);

    
#ifdef __cplusplus
}
#endif

#endif
