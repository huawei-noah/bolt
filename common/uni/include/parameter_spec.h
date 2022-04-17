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

typedef enum {
    ROUND_CEIL,
    ROUND_FLOOR,
    ROUND_TF_SAME,
    ROUND_TF_VALID,
    ROUND_PREFER_FLOOR,
    ROUND_PREFER_CEIL
} RoundMode;

typedef enum { RESIZE_LINEAR, RESIZE_NEAREST, RESIZE_CUBIC } ResizeMode;

typedef enum {
    COORDINATE_TRANS_ALIGN_CORNERS,
    COORDINATE_TRANS_HALF_PIXEL,
    COORDINATE_TRANS_PYTORCH_HALF_PIXEL,
    COORDINATE_TRANS_ASYMMETRIC,
    COORDINATE_TRANS_OUTPUT_HALF_PIXEL
} CoordinateTransMode;

typedef enum {
    ELTWISE_SUM,
    ELTWISE_MAX,
    ELTWISE_MIN,
    ELTWISE_PROD,
    ELTWISE_SUB,
    ELTWISE_DIV,
    ELTWISE_SQRT,
    ELTWISE_ERF,
    ELTWISE_AND,
    ELTWISE_OR,
    ELTWISE_XOR
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
    ACTIVATION_H_SWISH_NODIV,
    ACTIVATION_LOG,
    ACTIVATION_NOT,
    ACTIVATION_NEG,
    ACTIVATION_ROUND,
    ACTIVATION_FLOOR,
    ACTIVATION_CEIL,
    ACTIVATION_SWISH,
    ACTIVATION_RECIPROCAL
} ActivationMode;

typedef enum { BSLICE_APPLY_NULL, BSLICE_APPLY_CONV } BilateralSliceApplyMode;

typedef enum {
    CONVOLUTION_POINTWISE,
    CONVOLUTION_DILATION,
    CONVOLUTION_DEPTHWISE,
    CONVOLUTION_DEPTHWISE_POINTWISE,
    CONVOLUTION_DECONVOLUTION,
    CONVOLUTION_DEPTHWISE_DECONVOLUTION
} ConvolutionMode;

typedef enum { PAD_CONSTANT, PAD_REFLECT, PAD_EDGE, PAD_SYMMETRIC } PadMode;

typedef enum {
    CHECK_EQUAL,
    CHECK_GREATER_EQUAL,
    CHECK_GREATER,
    CHECK_LESS,
    CHECK_LESS_EQUAL,
    CHECK_NOT_EQUAL
} CheckMode;

typedef enum {
    REDUCTION_SUM,
    REDUCTION_MEAN,
    REDUCTION_STD_DEVIATION,
    REDUCTION_SCALAR_PRODUCT,
    REDUCTION_MAX,
    REDUCTION_MIN,
    REDUCTION_L2
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
    CONVOLUTION_ALGORITHM_INVGEMM,
    CONVOLUTION_ALGORITHM_GROUP_DECONV,
    CONVOLUTION_ALGORITHM_NULL
} ConvolutionForwardAlgorithm;

typedef enum {
    DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM,
    DEPTHWISE_CONVOLUTION_ALGORITHM_NULL
} DepthwiseConvolutionForwardAlgorithm;

#pragma pack(8)
typedef struct ActivationParamSpec {
    ActivationMode mode;
    float value[4] = {0, 0, 0, 0};
} ActivationParamSpec;

typedef struct {
    bool propagate_down;
} PReLUParamSpec;

typedef struct {
    int center_point_box;
    unsigned int max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
} NonMaxSuppressionParamSpec;

typedef struct {
    // save h, w
    unsigned int sizes[2];
    // save n, c, h, w
    float scales[4];
    unsigned int num_sizes;
    unsigned int num_scales;
    ResizeMode mode;
    CoordinateTransMode trans_mode;
    RoundMode round_mode;
} ResizeParamSpec;

typedef struct {
    int axes[8];
    int num_axes;
} SqueezeParamSpec;

typedef struct {
    int axes[8];
    int num_axes;
} UnsqueezeParamSpec;

typedef struct {
    DataType dt;
} CastParamSpec;

typedef struct {
    int axis;
    int num_concat;
} ScaleParamSpec;

typedef struct {
    float neg_slope;
} ReLUParamSpec;

typedef struct {
    float coeff[8];
    int num_coeff;
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
    EltwiseMode mode;
    EltwiseSumSpec sum_spec;
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
    unsigned int pad_before;
    unsigned int pad_after;
    unsigned int pad_top;
    unsigned int pad_bottom;
    unsigned int pad_left;
    unsigned int pad_right;
    unsigned int group;
    unsigned int dilatedRate_t;
    unsigned int dilatedRate_h;
    unsigned int dilatedRate_w;
    unsigned int num_outputs_origin;
    ConvolutionMode convolution_type;
    ActivationMode dw_activation_type;
    ActivationMode pw_activation_type;
    ActivationSpec activation_spec;
    RoundMode round_mode;
    unsigned int output_pad_t;
    unsigned int output_pad_h;
    unsigned int output_pad_w;
} ConvolutionParamSpec;

typedef struct {
    unsigned int kernel_t;
    unsigned int kernel_h;
    unsigned int kernel_w;
    unsigned int stride_t;
    unsigned int stride_h;
    unsigned int stride_w;
    unsigned int pad_before;
    unsigned int pad_after;
    unsigned int pad_top;
    unsigned int pad_bottom;
    unsigned int pad_left;
    unsigned int pad_right;
    RoundMode round_mode;
    PoolingMode mode;
    bool count_include_pad;
} PoolingParamSpec;

// FC's weight is reordered to NxK, K is removed dimension.
// slice parameter is for multi FC merge optimizer, default is 1.
typedef struct FullyConnectedParamSpec {
    unsigned int num_outputs;
    unsigned int num_slices = 1;
    int slice_point[32];
} FullyConnectedParamSpec;

typedef struct {
    int axis;
    float eps;
    float gama;
    float momentum;
} BatchNormParamSpec;

typedef struct {
    float eps;
    int axis;
    int axis_dim;
} InstanceNormParamSpec;

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
    unsigned int num_inputs;
    unsigned int num_outputs;
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
    int shape[8];
    int num_shape;
    int axis;
    int num_axes;
} ReshapeParamSpec;

typedef struct {
    int slice_points[8];
    unsigned int num_slice;
    int axis;
} SliceParamSpec;

typedef struct TransposeParamSpec {
    unsigned int axes[8];
    unsigned int num_axes;
    DataFormat df = DF_NCHW;
} TransposeParamSpec;

typedef struct {
    unsigned int num_heads;
    unsigned int from_sequence_length;
    unsigned int to_sequence_length;
} AttentionParamSpec;

typedef struct {
    RNNMode mode;
    unsigned int num_outputs;
    // steps >= 0 for multi-steps RNN
    // steps = -1 for RNNCell
    int steps;
    int num_projection;
    float zoneout_cell;
    float zoneout_output;

    bool bi_direction;
    float forget_bias;
    ActivationMode activation_type;
} RNNParamSpec;

typedef struct {
    unsigned int coefficient;
    BilateralSliceApplyMode mode;
    bool has_offset;
} BilateralSliceApplyParamSpec;

typedef struct {
    int axes[8];
    int num_axes;
    ReductionMode mode;
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
    CheckMode mode;
} CheckParamSpec;

typedef struct {
    int loops;
    int axis;
} RepeatParamSpec;

typedef struct PreAllocatedMemoryParamSpec {
    TensorDesc desc;
    float value = 0;
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
    unsigned int num_dims;
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
    int block_size;
} Space2DepthParamSpec;

typedef struct {
    int block_size;
    I8 mode[8];
} Depth2SpaceParamSpec;

typedef struct {
    int repeats[8];
    int num_repeats;
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
    ActivationMode activation_type;
    ReshapeParamSpec reshapeDesc[4];
    EltwiseParamSpec eltwiseDesc[2];
} MultiHeadAttentionParamSpec;

typedef struct {
    int axis;
    int largest;
    int sorted;
    int k;
} TopKParamSpec;

typedef struct {
    int shape[8];
    int num_shape;
} ExpandParamSpec;

typedef struct ScatterParamSpec {
    TensorDesc data_desc;
    TensorDesc index_desc;
    TensorDesc update_desc;

    // axis is used for ScatterElemnts, else axis = INT_MAX
    int axis = INT_MAX;
} ScatterParamSpec;

typedef struct GatherParamSpec {
    TensorDesc data_desc;
    TensorDesc index_desc;

    // axis is used for Gather/GatherElemnts, else axis = INT_MAX
    int axis = INT_MAX;
    // data dimension is 7x10, index content is 6;
    // index_scalar = false, index = [6], result dimension is 1 x 10
    // index_scalar = true, index = 6, result dimension is 10
    bool index_scalar = false;
    // element_level is used for GatherElemnts(true), else false
    bool element_level = false;
    // batch_dims for GatherND
    int batch_dims = 0;
} GatherParamSpec;

typedef struct {
    unsigned int num_heads;
    ActivationParamSpec activation_type;
} GATParamSpec;

typedef struct RoIAlignParamSpec {
    CoordinateTransMode trans_mode;
    PoolingMode mode;
    unsigned int output_h;
    unsigned int output_w;
    int sampling_ratio;
    float spatial_scale;
} RoIAlignParamSpec;

typedef struct GenerateProposalsParamSpec {
    int angle_bound_hi;
    int angle_bound_lo;
    int angle_bound_on;
    float clip_angle_thresh;
    int legacy_plus_one;
    float min_size;
    float nms_thresh;
    int post_nms_topN;
    int pre_nms_topN;
    float spatial_scale;
} GenerateProposalsParamSpec;

typedef struct QuantizeLinearParamSpec {
    // get the scales from input tensor
    int axis;
    DataType dt;
} QuantizeLinearParamSpec;

typedef struct {
    int axis;
    float eps;
} LayerNormParamSpec;

typedef struct RandomUniformParamSpec {
    DataType dt;
    float low;
    float high;
    float seed;
    int shape[8];
    int num_shape;
} RandomUniformParamSpec;

typedef struct CumSumParamSpec {
    bool exclusive;
    bool reverse;
    bool axis;
} CumSumParamSpec;

typedef struct GridSampleParamSpec {
    ResizeMode mode;
    PadMode pad_mode;
    float constant_value = 0;
    bool align_corners;
} GridSampleParamSpec;

typedef struct OneHotParamSpec {
    int axis;
    int depth;
    float values[2];
} OneHotParamSpec;

typedef struct ConstantOfShapeParamSpec {
    DataType dt;
    float value = 0;
} ConstantOfShapeParamSpec;

typedef struct RangeParamSpec {
    DataType dt;
    float start;
    float limit;
    float delta;
} RangeParamSpec;

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
    InstanceNormParamSpec in_spec;
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
    MultiHeadAttentionParamSpec multihead_attention_spec;
    TileParamSpec tile_spec;
    SpliceParamSpec splice_spec;
    TdnnParamSpec tdnn_spec;
    TopKParamSpec topk_spec;
    ExpandParamSpec expand_spec;
    ScatterParamSpec scatter_spec;
    RoIAlignParamSpec roialign_spec;
    GenerateProposalsParamSpec generate_proposals_spec;
    GATParamSpec gat_spec;
    QuantizeLinearParamSpec quant_spec;
    LayerNormParamSpec ln_spec;
    RandomUniformParamSpec random_uniform_spec;
    CumSumParamSpec cumsum_spec;
    GridSampleParamSpec grid_sample_spec;
    OneHotParamSpec onehot_spec;
    NonMaxSuppressionParamSpec non_max_suppression_spec;
    ConstantOfShapeParamSpec constant_of_shape_spec;
    RangeParamSpec range_spec;
} ParameterSpec;

typedef struct {
    int num_scale;
    float *scale;
} QuantSpec;
#pragma pack()

inline int get_operator_parameter_size(int version, OperatorType operatorType)
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
        {OT_MultiHeadAttention, sizeof(MultiHeadAttentionParamSpec)},
        {OT_Tile, sizeof(TileParamSpec)}, {OT_Splice, sizeof(SpliceParamSpec)},
        {OT_Tdnn, sizeof(TdnnParamSpec)}, {OT_TopK, sizeof(TopKParamSpec)},
        {OT_Expand, sizeof(ExpandParamSpec)}, {OT_InstanceNorm, sizeof(InstanceNormParamSpec)},
        {OT_Scatter, sizeof(ScatterParamSpec)}, {OT_LogSoftmax, sizeof(SoftmaxParamSpec)},
        {OT_GenerateProposals, sizeof(GenerateProposalsParamSpec)},
        {OT_RoIAlign, sizeof(RoIAlignParamSpec)}, {OT_GAT, sizeof(GATParamSpec)},
        {OT_QuantizeLinear, sizeof(QuantizeLinearParamSpec)},
        {OT_LayerNorm, sizeof(LayerNormParamSpec)},
        {OT_QuantizeLinear, sizeof(QuantizeLinearParamSpec)}, {OT_CumSum, sizeof(CumSumParamSpec)},
        {OT_RandomUniform, sizeof(RandomUniformParamSpec)},
        {OT_GridSample, sizeof(GridSampleParamSpec)}, {OT_OneHot, sizeof(OneHotParamSpec)},
        {OT_NonMaxSuppression, sizeof(NonMaxSuppressionParamSpec)},
        {OT_Range, sizeof(RangeParamSpec)}, {OT_ConstantOfShape, sizeof(ConstantOfShapeParamSpec)}};
    int size;
    if (operatorParameterSizeMap.find(operatorType) == operatorParameterSizeMap.end()) {
        size = 0;
    } else {
        size = operatorParameterSizeMap[operatorType];
    }
    if (version == 20201120) {
        if (operatorType == OT_Conv || operatorType == OT_Deconvolution) {
            size -= 3 * sizeof(unsigned int);
        }
        if (operatorType == OT_LayerNorm) {
            size = 0;
        }
    } else {
        size = (size + 3) / 4 * 4;
    }
    if (version == 20201120 || version == 20211021) {
        if (operatorType == OT_Transpose) {
            size -= sizeof(DataFormat);
        }
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
    unsigned int pad_before,
    unsigned int pad_after,
    unsigned int pad_top,
    unsigned int pad_bottom,
    unsigned int pad_left,
    unsigned int pad_right,
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
    p.pad_before = pad_before;
    p.pad_after = pad_after;
    p.pad_top = pad_top;
    p.pad_bottom = pad_bottom;
    p.pad_left = pad_left;
    p.pad_right = pad_right;
    p.dilatedRate_t = dilateRate_t;
    p.dilatedRate_h = dilateRate_h;
    p.dilatedRate_w = dilateRate_w;
    p.num_outputs = num_outputs;
    p.convolution_type = convMode;
    p.output_pad_t = 0;
    p.output_pad_h = 0;
    p.output_pad_w = 0;
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
    unsigned int pad_before,
    unsigned int pad_after,
    unsigned int pad_top,
    unsigned int pad_bottom,
    unsigned int pad_left,
    unsigned int pad_right,
    RoundMode round_mode)
{
    PoolingParamSpec p;
    p.mode = pm;
    p.kernel_t = kernel_t;
    p.kernel_h = kernel_h;
    p.kernel_w = kernel_w;
    p.stride_t = stride_t;
    p.stride_h = stride_h;
    p.stride_w = stride_w;
    p.pad_before = pad_before;
    p.pad_after = pad_after;
    p.pad_top = pad_top;
    p.pad_bottom = pad_bottom;
    p.pad_left = pad_left;
    p.pad_right = pad_right;
    p.round_mode = round_mode;
    return p;
}

inline ReshapeParamSpec createReshapeParamSpec(int *shape, int num_shape, int axis, int num_axes)
{
    ReshapeParamSpec p;
    p.num_shape = num_shape;
    p.axis = axis;
    p.num_axes = num_axes;
    if (shape != nullptr && num_shape != 0) {
        for (int i = 0; i < num_shape; i++) {
            p.shape[i] = shape[i];
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

inline SqueezeParamSpec createSqueezeParamSpec(int *axes, int num_axes)
{
    SqueezeParamSpec p;
    p.num_axes = num_axes;
    if (axes != nullptr && num_axes != 0) {
        for (int i = 0; i < num_axes; i++) {
            p.axes[i] = axes[i];
        }
    }
    return p;
}
#endif
