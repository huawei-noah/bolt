// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TYPES
#define _H_TYPES

#include <math.h>
#include "tensor_desc.h"
#include "op_type.h"
#ifdef __cplusplus
extern "C" {
#endif

static const int sg_boltVersion = 20201120;
static const int sg_magicNumber = 1141119;

typedef enum { POOLING_MAX, POOLING_MEAN } PoolingMode;

typedef enum { CEIL, FLOOR } RoundMode;

typedef enum {
    ELTWISE_SUM,
    ELTWISE_MAX,
    ELTWISE_MIN,
    ELTWISE_PROD,
    ELTWISE_SUB,
    ELTWISE_DIV,
    ELTWISE_SQRT,
    ELTWISE_ERF
} EltwiseMode;

typedef enum {
    ACTIVATION_NULL,
    ACTIVATION_RELU,
    ACTIVATION_RELU6,
    ACTIVATION_H_SWISH,
    ACTIVATION_H_SIGMOID,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_GELU,
    ACTIVATION_MISH,
    ACTIVATION_GREATER
} ActivationMode;

typedef enum { BSliceApply_NULL, BSliceApply_CONV } BilateralSliceApplyMode;

typedef enum {
    Convolution_Pointwise,
    Convolution_Dilation,
    Convolution_Depthwise,
    Convolution_Depthwise_Pointwise,
    Convolution_Deconvolution,
    Convolution_Depthwise_Deconvolution
} ConvolutionMode;

typedef enum { Pad_Constant, Pad_Reflect, Pad_Edge, Pad_Symmetric } PadMode;

typedef enum { CHECK_EQUAL, CHECK_GREATEQUAL, CHECK_GREAT } CheckMode;

typedef enum {
    REDUCTION_SUM,
    REDUCTION_MEAN,
    REDUCTION_STD_DEVIATION,
    REDUCTION_SCALAR_PRODUCT,
    REDUCTION_MAX
} ReductionMode;

typedef enum { KeepPrecision, ToFloat, ToInt } CastPrecisionMode;

typedef enum { F32_to_F32, F32_to_F16, F32_to_I8 } DataConvertType;

typedef enum { RNN_RNN, RNN_LSTM, RNN_GRU, RNN_GRU_LBR } RNNMode;

#pragma pack(8)
typedef struct {
    ActivationMode mode;
    float value[4] = {0, 0, 0, 0};
} ActivationParamSpec;

typedef struct {
    bool propagate_down;
} PReLUParamSpec;

typedef enum {
    CONVOLUTION_NO_TMP_MEM,
    CONVOLUTION_FASTEST,
    CONVOLUTION_TUNNING,
    CONVOLUTION_LIBRARY_SEARCH,
} ConvolutionPolicy;

typedef enum {
    CONVOLUTION_ALGORITHM_POINTWISE,
    CONVOLUTION_ALGORITHM_DIRECT,
    CONVOLUTION_ALGORITHM_IM2COL_GEMM,
    CONVOLUTION_ALGORITHM_GEMM,
    CONVOLUTION_ALGORITHM_GEMM_ICNCHW,
    CONVOLUTION_ALGORITHM_WINOGRAD,
    CONVOLUTION_ALGORITHM_BNN,
    CONVOLUTION_ALGORITHM_DIRECT_SPE_CK,
    CONVOLUTION_ALGORITHM_GROUP_DECONV,
    CONVOLUTION_ALGORITHM_NULL
} ConvolutionForwardAlgorithm;

typedef struct {
    F32 xmin;
    F32 ymin;
    F32 xmax;
    F32 ymax;
    U32 label;
} BoxRect;

typedef struct {
    U32 label;
    I64 box_index;
} BoxInfo;

typedef struct {
    U32 max_output_boxes_per_class;
    F32 iou_threshold;
    F32 score_threshold;
} NonMaxSuppressionParamSpec;

typedef struct {
    U32 output_h;
    U32 output_w;
    U32 sampling_ratio;
    F32 spatial_scale;
} RoiAlignParamSpec;

typedef enum {
    DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM,
    DEPTHWISE_CONVOLUTION_ALGORITHM_NULL
} DepthwiseConvolutionForwardAlgorithm;

typedef struct {
    char mode[NAME_LEN];
    U32 sizes[2];
    float scales[4];
    U32 num_sizes;
    U32 num_scales;
} ResizeParamSpec;

typedef struct {
    int gather_axis;
} GatherParamSpec;

typedef struct {
    int axes[8];
    int axes_num;
} SqueezeParamSpec;

typedef struct {
    int axes[8];
    int axes_num;
} UnsqueezeParamSpec;

typedef struct {
    CastPrecisionMode castPrecision;
} CastParamSpec;

typedef struct {
    int axis;
    int num_concat;
} ScaleParamSpec;

typedef struct {
    float neg_slope;
} ReLUParamSpec;

typedef struct {
    float coeff_values[8];
    int coeff_size;
} EltwiseSumSpec;

typedef struct {
    float min;
    float max;
} ClipParamSpec;

typedef union {
    ReLUParamSpec relu_spec;
    ClipParamSpec clip_spec;
} ActivationSpec;

typedef struct {
    EltwiseMode elt_mode;
    EltwiseSumSpec elt_sum_spec;
    ActivationMode activation_type;
    ActivationSpec activation_spec;
} EltwiseParamSpec;

typedef struct {
    U32 num_outputs;
    U32 kernel_t;
    U32 kernel_h;
    U32 kernel_w;
    U32 stride_t;
    U32 stride_h;
    U32 stride_w;
    U32 padding_before;
    U32 padding_after;
    U32 padding_top;
    U32 padding_bottom;
    U32 padding_left;
    U32 padding_right;
    U32 group;
    U32 dilatedRate_t;
    U32 dilatedRate_h;
    U32 dilatedRate_w;
    U32 num_outputs_origin;
    ConvolutionMode convolution_type;
    ActivationMode dw_activation_type;
    ActivationMode pw_activation_type;
    ActivationSpec activation_spec;
} ConvolutionParamSpec;

typedef struct {
    U32 kernel_t;
    U32 kernel_h;
    U32 kernel_w;
    U32 stride_t;
    U32 stride_h;
    U32 stride_w;
    U32 padding_before;
    U32 padding_after;
    U32 padding_top;
    U32 padding_bottom;
    U32 padding_left;
    U32 padding_right;
    RoundMode rm;
    PoolingMode mode;
} PoolingParamSpec;

typedef struct {
    U32 num_outputs;
    U32 num_slices;
    I32 slice_point[32];
} FullyConnectedParamSpec;

typedef struct {
    int axis;
    F32 eps;
    F32 gama;
    F32 momentum;
} BatchNormParamSpec;

typedef struct {
    U32 before;
    U32 after;
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
    int axis;
} EmbedParamSpec;

typedef struct {
    float scale;
    float shift;
    float power;
} PowerParamSpec;

typedef struct {
    I32 shape_dims[8];
    I32 shape_size;
    I32 axis;
    I32 num_axes;
} ReshapeParamSpec;

typedef struct {
    I32 slice_points[8];
    U32 slice_size;
    I32 axis;
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
    RNNMode mode;
    U32 numOutput;
    I32 steps;
    I32 numProjection;
    float zoneoutCell;
    float zoneoutOutput;

    bool biDirection;
    float forgetBias;
    ActivationMode activationMode;
} RNNParamSpec;

typedef struct {
    U32 coefficient_len;
    BilateralSliceApplyMode mode;
    bool has_offset;
} BilateralSliceApplyParamSpec;

typedef struct {
    I32 axes[8];
    I32 axes_num;
    ReductionMode reduction_mode;
    float coeff;
    bool keep_dim;
} ReductionParamSpec;

typedef struct {
    I32 axis;
} ArgMaxParamSpec;

typedef struct {
    I32 src_dims[3];
    I32 dst_dims[3];
    I32 length;
} CopyParamSpec;

typedef struct {
    CheckMode check_mode;
} CheckParamSpec;

typedef struct {
    int loops;
    int axis;
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

typedef struct {
    int attention_length;
    float mask;
    bool same_length;
} AttentionMaskParamSpec;

typedef struct {
    int axis;
    int shift_length;
} RelativeShiftParamSpec;

typedef struct {
    int axis;
    int num_concat;
} ConcatParamSpec;

typedef struct {
    int axis;
} SoftmaxParamSpec;

typedef struct {
    int begin[8];
    int end[8];
    int strides[8];
    char begin_mask[8];
    char end_mask[8];
    char ellipsis_mask[8];
    char new_axis_mask[8];
    char shrink_axis_mask[8];
    U32 dim_size;
} TfSliceParamSpec;

typedef struct {
    F32 min_sizes[2];
    F32 max_sizes[2];
    F32 aspect_ratios[2];
    U32 flip;
    U32 clip;
    F32 variances[4];
    U32 image_h;
    U32 image_w;
    F32 step_h;
    F32 step_w;
    F32 offset;
} PriorBoxParamSpec;

typedef struct {
    U32 num_class;
    F32 nms_threshold;
    U32 nms_top_k;
    U32 keep_top_k;
    F32 confidence_threshold;
} DetectionOutputParamSpec;

typedef struct {
    U32 num_class;
    U32 num_box;
    F32 confidence_threshold;
    F32 nms_threshold;
    F32 biases[18];
    U32 anchors_scale[3];
    U32 mask_group_num;
    U32 mask[9];
} Yolov3DetectionOutputParamSpec;

typedef struct {
    char symmetric[NAME_LEN];
    int group;
    int channel_before;
    int channel_after;
} ChannelResizeParamSpec;

typedef struct {
    int blockSize;
} Space2DepthParamSpec;

typedef struct {
    int blockSize;
    I8 reMode[8];
} Depth2SpaceParamSpec;

typedef struct {
    int repeatsInfo[8];
    int dimsSize;
    int axis;
} TileParamSpec;

typedef struct {
    U32 numIndices;
    int outputDim;
} SpliceParamSpec;

typedef struct {
    FullyConnectedParamSpec fc_desc[6];
    PowerParamSpec power_spec;
    bool eltwiseWithLayerNormIn[2];
    ActivationMode actiMode;
    ReshapeParamSpec reshapeDesc[4];
    EltwiseParamSpec eltwiseDesc[2];
} MultiheadAttentionParamSpec;

typedef union ParameterSpec {
    ParameterSpec()
    {}
    ConvolutionParamSpec conv_spec;
    FullyConnectedParamSpec fc_spec;
    RNNParamSpec rnn_spec;
    MatMulParamSpec matmul_spec;
    ResizeParamSpec resize_spec;
    BilateralSliceApplyParamSpec bilateral_slice_apply_spec;
    PoolingParamSpec pooling_spec;
    ScaleParamSpec scale_spec;
    BatchNormParamSpec bn_spec;
    ReductionParamSpec reduction_spec;
    ArgMaxParamSpec argmax_spec;
    SoftmaxParamSpec softmax_spec;
    ClipParamSpec clip_spec;
    PowerParamSpec power_spec;
    ReLUParamSpec relu_spec;
    GatherParamSpec gather_spec;
    EmbedParamSpec embed_spec;
    PadParamSpec pad_spec;
    EltwiseParamSpec eltwise_spec;
    ConcatParamSpec concat_spec;
    SliceParamSpec slice_spec;
    TfSliceParamSpec tfslice_spec;
    CastParamSpec cast_spec;
    TransposeParamSpec transpose_spec;
    ReshapeParamSpec reshape_spec;
    SqueezeParamSpec squeeze_spec;
    UnsqueezeParamSpec unsqueeze_spec;
    Space2DepthParamSpec space2depth_spec;
    Depth2SpaceParamSpec depth2space_spec;
    ChannelResizeParamSpec channel_resize_spec;
    PreAllocatedMemoryParamSpec preallocated_memory_spec;
    SharedWeightParamSpec shared_weight_spec;
    CopyParamSpec copy_spec;
    CheckParamSpec check_spec;
    RepeatParamSpec repeat_spec;
    AttentionParamSpec attention_spec;
    AttentionMaskParamSpec attention_mask_spec;
    RelativeShiftParamSpec relative_shift_spec;
    PriorBoxParamSpec prior_box_spec;
    DetectionOutputParamSpec detection_output_spec;
    Yolov3DetectionOutputParamSpec yolov3_detection_output_spec;
    MultiheadAttentionParamSpec multiheadAttention_spec;
    TileParamSpec tile_spec;
    SpliceParamSpec splice_spec;
} ParameterSpec;

typedef struct {
    int num_scale;
    F32 *scale;
} QuantSpec;

typedef struct {
    I8 name[NAME_LEN];
    OperatorType type;
    U32 num_inputs;
    I8 **input_tensors_name;
    U32 num_outputs;
    I8 **output_tensors_name;
    I32 *tensor_positions;
    U32 num_quant_feature;
    QuantSpec *feature_scale;
    ParameterSpec ps;
} OperatorSpec;

typedef struct {
    I8 op_name[NAME_LEN];
    DataType mdt = DT_U8;
    U32 bytes_of_weight = 0;
    U8 *weight;
    U32 bytes_of_vec = 0;
    U8 *vec;
    U32 num_quant_scale;  // Merged FC may have multiple weight scales
    QuantSpec *weight_scale;
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
#pragma pack()

#ifdef __cplusplus
}
#endif

OperatorSpec mt_create_operator(
    const char *name, OperatorType type, U32 num_inputs, U32 num_outputs);

EE mt_insert_operator(ModelSpec *ms, int index, OperatorSpec newOperator);

WeightSpec mt_create_weight(
    const char *name, DataType dataType, U32 bytesOfWeight, U32 bytesOfVec, U32 numQuantScale);

bool isDeprecatedOp(OperatorType opType);

bool isDeprecatedOpWeight(const ModelSpec *spec, int index);

EE str_copy(I8 *dst, const I8 *src, I32 src_len, I32 dst_len = NAME_LEN);

void *mt_new_storage(size_t size);

inline INT8 round_towards_zero(F32 num, bool clamp = true)
{
    INT8 ret;
    if (clamp) {
        if (num > 127.0) {
            return 127;
        } else if (num < -127.0) {
            return -127;
        }
    }
    if (num > 0) {
        ret = floor(num);
    } else {
        ret = ceil(num);
    }
    return ret;
}

#endif
