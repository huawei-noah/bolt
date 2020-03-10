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
#include "op_type.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef enum {
        F32_to_F32,
        F32_to_F16,
        F32_to_I8
    } DataConvertType;

    typedef struct {
        U32 height;
        U32 width;
    } InterpParamSpec;

    typedef struct {
        int axis;
    } FlattenParamSpec;

    typedef struct {
        int gather_axis;
    } GatherParamSpec;

    typedef struct {
        int axis;
        int squeeze_axes[8];
        int axes_num;
    } SqueezeParamSpec;

    typedef struct {
        int axis;
        int unsqueeze_axes[8];
        int axes_num;
    } UnsqueezeParamSpec;

    typedef struct {
        char upsample_mode[NAME_LEN];
        F32 scale[4];
    } UpsampleParamSpec;

    typedef struct {
        int cast_to;
    } CastParamSpec;

    typedef struct {
        int num_concat;
    } ScaleParamSpec;

    typedef struct {
        float neg_slope;
    } ReLUParamSpec;

    typedef struct {
        int coeff_size;
        float* coeff_values;
    } EltwiseSumSpec;

    typedef struct {
        EltwiseMode elt_mode;
        EltwiseSumSpec elt_sum_spec;
    } EltwiseParamSpec;

    typedef struct {
        float min;
        float max;
    } ClipParamSpec;

    typedef union {
        ReLUParamSpec relu_spec;
        ClipParamSpec clip_spec;
    } ActivationSpec;

    typedef struct {
        U32 num_kernels;
        U32 kernel_size_h;
        U32 kernel_size_w;
        U32 stride_h;
        U32 stride_w;
        U32 padding_top;
        U32 padding_bottom;
        U32 padding_left;
        U32 padding_right;
        U32 group;
        U32 dilatedRate_h;
        U32 dilatedRate_w;
        ConvolutionMode convolution_type;
        ActivationMode dw_activation_type;
        ActivationMode pw_activation_type;
        ActivationSpec activation_spec;
    } ConvolutionParamSpec;

    typedef struct {
        U32 kernel_size_h;
        U32 kernel_size_w;
        U32 stride_h;
        U32 stride_w;
        U32 padding_top;
        U32 padding_bottom;
        U32 padding_left;
        U32 padding_right;
        RoundMode rm;
        PoolingMode mode;
    } PoolingParamSpec;

    typedef struct {
        U32 num_outputs;
    } FullyConnectedParamSpec;

    typedef struct{
        F32 eps;
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
        bool transpose;
    } EmbedParamSpec;

    typedef struct {
        float scale;
        float bias;
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
        U32 num_heads;
        U32 from_sequence_length;
        U32 to_sequence_length;
    } AttentionParamSpec;

    typedef struct {
        U32 num_output;
        I32 steps;
    } LSTMParamSpec;

    typedef struct {
        U32 coefficient_len;
        bool has_offset;
        BilateralSliceApplyMode mode;
    } BilateralSliceApplyParamSpec;
    typedef struct {
        I32 axis;
    } AxisMeanParamSpec;

    typedef struct {
        I32 axis;
    } ArgMaxParamSpec;

    typedef struct {
        U32 src_dims[3];
        U32 dst_dims[3];
        U32 length;
    } CopyParamSpec;

    typedef struct {
        CheckMode check_mode;
    } CheckParamSpec;

    typedef struct {
        int loops;
    } RepeatParamSpec;

    typedef struct {
        TensorDesc desc;
    } PreAllocatedMemoryParamSpec;

    typedef struct {
        TensorDesc desc;
    } SharedWeightParamSpec;
    
    typedef struct {
        bool transpose_a;
        bool transpose_b;
    } MatMulParamSpec;

    typedef union ParameterSpec {
        ParameterSpec() {}
        ConvolutionParamSpec conv_spec;
        PoolingParamSpec pooling_spec;
        FullyConnectedParamSpec fc_spec;
        BatchNormParamSpec bn_spec;
        EltwiseParamSpec eltwise_spec;
        ReLUParamSpec relu_spec;
        ClipParamSpec clip_spec;
        PadParamSpec pad_spec;
        EmbedParamSpec embed_spec;
        MultiplyParamSpec multiply_spec;
        ReshapeParamSpec reshape_spec;
        SliceParamSpec slice_spec;
        TransposeParamSpec transpose_spec;
        AttentionParamSpec attention_spec;
        LSTMParamSpec lstm_spec;
        GatherParamSpec gather_spec;
        UnsqueezeParamSpec unsqueeze_spec;
        SqueezeParamSpec squeeze_spec;
        UpsampleParamSpec upsample_spec;
        CastParamSpec cast_spec;
        BilateralSliceApplyParamSpec bilateral_slice_apply_spec;
        ScaleParamSpec scale_spec;
        AxisMeanParamSpec axis_mean_spec;
        CopyParamSpec copy_spec;
        CheckParamSpec check_spec;
        RepeatParamSpec repeat_spec;
        PreAllocatedMemoryParamSpec preallocated_memory_spec;
        SharedWeightParamSpec shared_weight_spec;
        ArgMaxParamSpec argmax_spec;
        MatMulParamSpec matmul_spec;
        InterpParamSpec interp_spec;
        FlattenParamSpec flatten_spec;
    } ParameterSpec;

    typedef struct {
        I8 name[NAME_LEN];
        OperatorType type;
        U32 num_inputs;
        I8 **input_tensors_name;
        U32 num_outputs;
        I8 **output_tensors_name;
        I32 *tensor_positions;
        ParameterSpec ps;
    } OperatorSpec;

    typedef struct {
        I8 op_name[NAME_LEN];
        DataType mdt = DT_U8;
        U32 bytes_of_weight = 0;
        U8* weight;
        U32 bytes_of_vec = 0;
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
        static const I32 magic_number = 1141119;
        return magic_number;
    }

    // you must invoke this before use md
    EE mt_create_model(ModelSpec* md);
    EE mt_load(CI8* dir, CI8* mfn, ModelSpec* md);
    EE mt_store(CI8* dir, CI8* mfn, const ModelSpec* md);
    // you must invoke this before exit to clean up resource usage
    EE mt_destroy_model(ModelSpec* md);

    
#ifdef __cplusplus
}
#endif

#endif
