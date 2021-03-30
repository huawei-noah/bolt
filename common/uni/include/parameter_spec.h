// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_PARAMETER_SPEC
#define _H_PARAMETER_SPEC

#include <map>
#include "operator_type.h"
#include "tensor_desc.h"

#define NAME_LEN 128

typedef enum { POOLING_MAX, POOLING_MEAN } PoolingMode;

typedef enum { CEIL, FLOOR, TF_SAME, TF_VALID, ROUND_PREFER_FLOOR, ROUND_PREFER_CEIL } RoundMode;

typedef enum { LINEAR, NEAREST, CUBIC } ResizeMode;

typedef enum { ALIGN_CORNERS, HALF_PIXEL } ResizeCoordinateTransMode;

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
    ACTIVATION_GREATER,
    ACTIVATION_SOFTPLUS,
    ACTIVATION_EXP,
    ACTIVATION_ABS,
    ACTIVATION_SIGN,
    ACTIVATION_H_SWISH_NODIV
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
    REDUCTION_MAX,
    REDUCTION_MIN
} ReductionMode;

typedef enum { F32_to_F32, F32_to_F16, F32_to_I8 } DataConvertType;

typedef enum { RNN_RNN, RNN_LSTM, RNN_GRU, RNN_GRU_LBR } RNNMode;

typedef enum {
    RGB_SC = 0,  // scale and center crop
    RGB = 1,
    BGR = 2,
    RGB_RAW = 3,
    RGB_SC_RAW = 4,
    BGR_SC_RAW = 5
} ImageFormat;

#pragma pack(8)
typedef struct ActivationParamSpec {
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
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    unsigned int label;
} BoxRect;

typedef struct {
    unsigned int label;
    I64 box_index;
} BoxInfo;

typedef struct {
    unsigned int max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
} NonMaxSuppressionParamSpec;

typedef struct {
    unsigned int output_h;
    unsigned int output_w;
    unsigned int sampling_ratio;
    float spatial_scale;
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
    unsigned int sizes[2];
    float scales[4];
    unsigned int num_sizes;
    unsigned int num_scales;
    ResizeMode mode;
    ResizeCoordinateTransMode trans_mode;
    RoundMode round_mode;
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
    DataType targetDt;
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
    unsigned int num_outputs;
    unsigned int kernel_t;
    unsigned int kernel_h;
    unsigned int kernel_w;
    unsigned int stride_t;
    unsigned int stride_h;
    unsigned int stride_w;
    unsigned int padding_before;
    unsigned int padding_after;
    unsigned int padding_top;
    unsigned int padding_bottom;
    unsigned int padding_left;
    unsigned int padding_right;
    unsigned int group;
    unsigned int dilatedRate_t;
    unsigned int dilatedRate_h;
    unsigned int dilatedRate_w;
    unsigned int num_outputs_origin;
    ConvolutionMode convolution_type;
    ActivationMode dw_activation_type;
    ActivationMode pw_activation_type;
    ActivationSpec activation_spec;
    RoundMode rm;
} ConvolutionParamSpec;

typedef struct {
    unsigned int kernel_t;
    unsigned int kernel_h;
    unsigned int kernel_w;
    unsigned int stride_t;
    unsigned int stride_h;
    unsigned int stride_w;
    unsigned int padding_before;
    unsigned int padding_after;
    unsigned int padding_top;
    unsigned int padding_bottom;
    unsigned int padding_left;
    unsigned int padding_right;
    RoundMode rm;
    PoolingMode mode;
} PoolingParamSpec;

typedef struct {
    unsigned int num_outputs;
    unsigned int num_slices;
    int slice_point[32];
} FullyConnectedParamSpec;

typedef struct {
    int axis;
    float eps;
    float gama;
    float momentum;
} BatchNormParamSpec;

typedef struct {
    // padding on time dimension
    unsigned int before;
    unsigned int after;
    // padding on channel dimension
    unsigned int front;
    unsigned int back;
    // padding on height dimension
    unsigned int top;
    unsigned int bottom;
    // padding on width dimension
    unsigned int left;
    unsigned int right;
    float constant_value;
    PadMode pad_mode;
} PadParamSpec;

typedef struct {
    unsigned int input_dim;
    unsigned int num_output;
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
    int shape_dims[8];
    int shape_size;
    int axis;
    int num_axes;
} ReshapeParamSpec;

typedef struct {
    int slice_points[8];
    unsigned int slice_size;
    int axis;
} SliceParamSpec;

typedef struct {
    unsigned int trans_dims[8];
    unsigned int trans_size;
} TransposeParamSpec;

typedef struct {
    unsigned int num_heads;
    unsigned int from_sequence_length;
    unsigned int to_sequence_length;
} AttentionParamSpec;

typedef struct {
    RNNMode mode;
    unsigned int numOutput;
    // steps >= 0 for multi-steps RNN
    // steps = -1 for RNNCell
    int steps;
    int numProjection;
    float zoneoutCell;
    float zoneoutOutput;

    bool biDirection;
    float forgetBias;
    ActivationMode activationMode;
} RNNParamSpec;

typedef struct {
    unsigned int coefficient_len;
    BilateralSliceApplyMode mode;
    bool has_offset;
} BilateralSliceApplyParamSpec;

typedef struct {
    int axes[8];
    int axes_num;
    ReductionMode reduction_mode;
    float coeff;
    bool keep_dim;
} ReductionParamSpec;

typedef struct {
    int axis;
} ArgMaxParamSpec;

typedef struct {
    int src_dims[3];
    int dst_dims[3];
    int length;
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
    unsigned int dim_size;
} TfSliceParamSpec;

typedef struct {
    float min_sizes[2];
    float max_sizes[2];
    float aspect_ratios[2];
    unsigned int flip;
    unsigned int clip;
    float variances[4];
    unsigned int image_h;
    unsigned int image_w;
    float step_h;
    float step_w;
    float offset;
} PriorBoxParamSpec;

typedef struct {
    unsigned int num_class;
    float nms_threshold;
    unsigned int nms_top_k;
    unsigned int keep_top_k;
    float confidence_threshold;
} DetectionOutputParamSpec;

typedef struct {
    unsigned int num_class;
    unsigned int num_box;
    float confidence_threshold;
    float nms_threshold;
    float biases[18];
    unsigned int anchors_scale[3];
    unsigned int mask_group_num;
    unsigned int mask[9];
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
    int context[8];
    int num_context;
    int index_min;
    int index_max;
} SpliceParamSpec;

typedef struct {
    int context[8];
    int num_context;
    int num_outputs;
    ActivationMode activation_type;
    ActivationSpec activation_spec;
} TdnnParamSpec;

typedef struct {
    FullyConnectedParamSpec fc_desc[6];
    PowerParamSpec power_spec;
    bool eltwiseWithLayerNormIn[2];
    ActivationMode actiMode;
    ReshapeParamSpec reshapeDesc[4];
    EltwiseParamSpec eltwiseDesc[2];
} MultiheadAttentionParamSpec;

typedef struct {
    int axis;
    int largest;
    int sorted;
    int topk;
} TopKParamSpec;

typedef struct {
    TensorDesc conditionDesc;
    TensorDesc yDesc;
} WhereParamSpec;

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
    TdnnParamSpec tdnn_spec;
    TopKParamSpec topk_spec;
    WhereParamSpec where_spec;
} ParameterSpec;

typedef struct {
    int num_scale;
    float *scale;
} QuantSpec;
#pragma pack()

inline int get_operator_parameter_size(OperatorType operatorType)
{
    std::map<OperatorType, int> operatorParameterSizeMap = {{OT_Conv, sizeof(ConvolutionParamSpec)},
        {OT_Deconvolution, sizeof(ConvolutionParamSpec)}, {OT_FC, sizeof(FullyConnectedParamSpec)},
        {OT_RNN, sizeof(RNNParamSpec)}, {OT_MatMul, sizeof(MatMulParamSpec)},
        {OT_Resize, sizeof(ResizeParamSpec)},
        {OT_BilateralSliceApply, sizeof(BilateralSliceApplyParamSpec)},
        {OT_Pooling, sizeof(PoolingParamSpec)}, {OT_Scale, sizeof(ScaleParamSpec)},
        {OT_BatchNorm, sizeof(BatchNormParamSpec)}, {OT_Reduction, sizeof(ReductionParamSpec)},
        {OT_ArgMax, sizeof(ArgMaxParamSpec)}, {OT_Softmax, sizeof(SoftmaxParamSpec)},
        {OT_Clip, sizeof(ClipParamSpec)}, {OT_Power, sizeof(PowerParamSpec)},
        {OT_Relu, sizeof(ReLUParamSpec)}, {OT_Gather, sizeof(GatherParamSpec)},
        {OT_Embedding, sizeof(EmbedParamSpec)}, {OT_Pad, sizeof(PadParamSpec)},
        {OT_Eltwise, sizeof(EltwiseParamSpec)}, {OT_Concat, sizeof(ConcatParamSpec)},
        {OT_Slice, sizeof(SliceParamSpec)}, {OT_TfSlice, sizeof(TfSliceParamSpec)},
        {OT_Cast, sizeof(CastParamSpec)}, {OT_Transpose, sizeof(TransposeParamSpec)},
        {OT_Reshape, sizeof(ReshapeParamSpec)}, {OT_Squeeze, sizeof(SqueezeParamSpec)},
        {OT_Unsqueeze, sizeof(UnsqueezeParamSpec)}, {OT_Space2Depth, sizeof(Space2DepthParamSpec)},
        {OT_Depth2Space, sizeof(Depth2SpaceParamSpec)},
        {OT_ChannelResize, sizeof(ChannelResizeParamSpec)},
        {OT_PreAllocatedMemory, sizeof(PreAllocatedMemoryParamSpec)},
        {OT_SharedWeight, sizeof(SharedWeightParamSpec)}, {OT_Copy, sizeof(CopyParamSpec)},
        {OT_Check, sizeof(CheckParamSpec)}, {OT_Repeat, sizeof(RepeatParamSpec)},
        {OT_Attention, sizeof(AttentionParamSpec)},
        {OT_AttentionMask, sizeof(AttentionMaskParamSpec)},
        {OT_RelativePositionEmbedding, sizeof(EmbedParamSpec)},
        {OT_RelativeShift, sizeof(RelativeShiftParamSpec)}, {OT_PriorBox, sizeof(PriorBoxParamSpec)},
        {OT_DetectionOutput, sizeof(DetectionOutputParamSpec)},
        {OT_Yolov3DetectionOutput, sizeof(Yolov3DetectionOutputParamSpec)},
        {OT_MultiHeadAttention, sizeof(MultiheadAttentionParamSpec)},
        {OT_Tile, sizeof(TileParamSpec)}, {OT_Splice, sizeof(SpliceParamSpec)},
        {OT_Tdnn, sizeof(TdnnParamSpec)}, {OT_TopK, sizeof(TopKParamSpec)},
        {OT_Where, sizeof(WhereParamSpec)}};
    int size;
    if (operatorParameterSizeMap.find(operatorType) == operatorParameterSizeMap.end()) {
        size = 0;
    } else {
        size = operatorParameterSizeMap[operatorType];
    }
    return size;
}

inline ConvolutionParamSpec createConvolutionParamSpec(unsigned int group,
    unsigned int kernel_t,
    unsigned int kernel_h,
    unsigned int kernel_w,
    unsigned int stride_t,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int padding_before,
    unsigned int padding_after,
    unsigned int padding_top,
    unsigned int padding_bottom,
    unsigned int padding_left,
    unsigned int padding_right,
    unsigned int dilateRate_t,
    unsigned int dilateRate_h,
    unsigned int dilateRate_w,
    unsigned int num_outputs,
    ConvolutionMode convMode)
{
    ConvolutionParamSpec p;
    p.group = group;
    p.kernel_t = kernel_t;
    p.kernel_h = kernel_h;
    p.kernel_w = kernel_w;
    p.stride_t = stride_t;
    p.stride_h = stride_h;
    p.stride_w = stride_w;
    p.padding_before = padding_before;
    p.padding_after = padding_after;
    p.padding_top = padding_top;
    p.padding_bottom = padding_bottom;
    p.padding_left = padding_left;
    p.padding_right = padding_right;
    p.dilatedRate_t = dilateRate_t;
    p.dilatedRate_h = dilateRate_h;
    p.dilatedRate_w = dilateRate_w;
    p.num_outputs = num_outputs;
    p.convolution_type = convMode;
    return p;
}

inline FullyConnectedParamSpec createFullyConnectedParamSpec(
    unsigned int num_outputs, unsigned int num_slices, I32 *slice_point)
{
    FullyConnectedParamSpec p;
    p.num_outputs = num_outputs;
    p.num_slices = num_slices;
    if (num_slices > 1 && slice_point != nullptr) {
        for (int i = 0; i < (int)num_slices; i++) {
            p.slice_point[i] = slice_point[i];
        }
    }
    return p;
}

inline PoolingParamSpec createPoolingParamSpec(PoolingMode pm,
    unsigned int kernel_t,
    unsigned int kernel_h,
    unsigned int kernel_w,
    unsigned int stride_t,
    unsigned int stride_h,
    unsigned int stride_w,
    unsigned int padding_before,
    unsigned int padding_after,
    unsigned int padding_top,
    unsigned int padding_bottom,
    unsigned int padding_left,
    unsigned int padding_right,
    RoundMode rm)
{
    PoolingParamSpec p;
    p.mode = pm;
    p.kernel_t = kernel_t;
    p.kernel_h = kernel_h;
    p.kernel_w = kernel_w;
    p.stride_t = stride_t;
    p.stride_h = stride_h;
    p.stride_w = stride_w;
    p.padding_before = padding_before;
    p.padding_after = padding_after;
    p.padding_top = padding_top;
    p.padding_bottom = padding_bottom;
    p.padding_left = padding_left;
    p.padding_right = padding_right;
    p.rm = rm;
    return p;
}

inline ReshapeParamSpec createReshapeParamSpec(int *shape_dims, int shape_size, int axis, int num_axes)
{
    ReshapeParamSpec p;
    p.shape_size = shape_size;
    p.axis = axis;
    p.num_axes = num_axes;
    if (shape_dims != nullptr && shape_size != 0) {
        for (int i = 0; i < shape_size; i++) {
            p.shape_dims[i] = shape_dims[i];
        }
    }
    return p;
}

inline ClipParamSpec createClipParamSpec(float min, float max)
{
    ClipParamSpec p;
    p.min = min;
    p.max = max;
    return p;
}

inline SqueezeParamSpec createSqueezeParamSpec(int *axes, int axes_num)
{
    SqueezeParamSpec p;
    p.axes_num = axes_num;
    if (axes != nullptr && axes_num != 0) {
        for (int i = 0; i < axes_num; i++) {
            p.axes[i] = axes[i];
        }
    }
    return p;
}
#endif
