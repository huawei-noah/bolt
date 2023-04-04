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

typedef enum PoolingMode : ENUM_TYPE { POOLING_MAX, POOLING_MEAN } PoolingMode;

typedef enum RandomMode : ENUM_TYPE { RANDOM_NORMAL, RANDOM_UNIFORM } RandomMode;

typedef enum RoundMode : ENUM_TYPE {
    ROUND_CEIL,
    ROUND_FLOOR,
    ROUND_TF_SAME,
    ROUND_TF_VALID,
    ROUND_PREFER_FLOOR,
    ROUND_PREFER_CEIL,
    ROUND_SAME_UPPER,
    ROUND_SAME_LOWER
} RoundMode;

typedef enum ResizeMode : ENUM_TYPE { RESIZE_LINEAR, RESIZE_NEAREST, RESIZE_CUBIC } ResizeMode;

typedef enum CoordinateTransMode : ENUM_TYPE {
    COORDINATE_TRANS_ALIGN_CORNERS,
    COORDINATE_TRANS_HALF_PIXEL,
    COORDINATE_TRANS_PYTORCH_HALF_PIXEL,
    COORDINATE_TRANS_ASYMMETRIC,
    COORDINATE_TRANS_OUTPUT_HALF_PIXEL
} CoordinateTransMode;

typedef enum EltwiseMode : ENUM_TYPE {
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

typedef enum ActivationMode : ENUM_TYPE {
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
    ACTIVATION_RECIPROCAL,
    ACTIVATION_SIN,
    ACTIVATION_COS,
    ACTIVATION_ELU,
} ActivationMode;

typedef enum BilateralSliceApplyMode : ENUM_TYPE {
    BILATERAL_SLICE_APPLY_NULL,
    BILATERAL_SLICE_APPLY_CONV
} BilateralSliceApplyMode;

typedef enum ConvolutionMode : ENUM_TYPE {
    CONVOLUTION_POINTWISE,
    CONVOLUTION_DILATION,
    CONVOLUTION_DEPTHWISE,
    CONVOLUTION_DEPTHWISE_POINTWISE,
    CONVOLUTION_DECONVOLUTION,
    CONVOLUTION_DEPTHWISE_DECONVOLUTION
} ConvolutionMode;

typedef enum PadMode : ENUM_TYPE { PAD_CONSTANT, PAD_REFLECT, PAD_EDGE, PAD_SYMMETRIC } PadMode;

typedef enum CheckMode : ENUM_TYPE {
    CHECK_EQUAL,
    CHECK_GREATER_EQUAL,
    CHECK_GREATER,
    CHECK_LESS,
    CHECK_LESS_EQUAL,
    CHECK_NOT_EQUAL
} CheckMode;

typedef enum ReductionMode : ENUM_TYPE {
    REDUCTION_SUM,
    REDUCTION_MEAN,
    REDUCTION_STD_DEVIATION,
    REDUCTION_SCALAR_PRODUCT,
    REDUCTION_MAX,
    REDUCTION_MIN,
    REDUCTION_L2
} ReductionMode;

typedef enum DataConvertType : ENUM_TYPE { F32_to_F32, F32_to_F16, F32_to_I8 } DataConvertType;

typedef enum RNNMode : ENUM_TYPE { RNN_RNN, RNN_LSTM, RNN_GRU, RNN_GRU_LBR } RNNMode;

typedef enum ColorSpace : ENUM_TYPE {
    RGB_0_255 = 10,
    RGB_0_1 = 11,
    BGR_0_255 = 12,
    BGR_0_1 = 13,
    RGBA_0_255 = 20,
    RGBA_0_1 = 21,
    BGRA_0_255 = 22,
    BGRA_0_1 = 23,
    YUV_NV21 = 41,
    YUV_NV12 = 42,
} ColorSpace;

typedef enum ImageFormat : ENUM_TYPE {
    RGB_SC = 0,  // scale and center crop
    RGB = 1,
    BGR = 2,
    RGB_RAW = 3,
    RGB_SC_RAW = 4,
    BGR_SC_RAW = 5
} ImageFormat;

typedef enum ConvolutionPolicy : ENUM_TYPE {
    CONVOLUTION_NO_TMP_MEM,
    CONVOLUTION_FASTEST,
    CONVOLUTION_TUNNING,
    CONVOLUTION_LIBRARY_SEARCH,
} ConvolutionPolicy;

typedef enum ConvolutionForwardAlgorithm : ENUM_TYPE {
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

typedef enum DepthwiseConvolutionForwardAlgorithm : ENUM_TYPE {
    DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM,
    DEPTHWISE_CONVOLUTION_ALGORITHM_NULL
} DepthwiseConvolutionForwardAlgorithm;

#pragma pack(8)
typedef struct ActivationParamSpec {
    ActivationMode mode = ACTIVATION_NULL;
    float value[4] = {0, 0, 0, 0};
} ActivationParamSpec;

typedef struct {
    bool propagate_down;
} PReLUParamSpec;

typedef struct {
    I32 center_point_box;
    U32 max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
} NonMaxSuppressionParamSpec;

typedef struct {
    // save h, w
    U32 sizes[2];
    // save n, c, h, w
    float scales[4];
    U32 num_sizes;
    U32 num_scales;
    ResizeMode mode;
    CoordinateTransMode trans_mode;
    RoundMode round_mode;
    float zoom_factor;
    I32 pad_begin;
    I32 pad_end;
} ResizeParamSpec;

typedef struct {
    I32 axes[8];
    I32 num_axes;
} SqueezeParamSpec;

typedef struct {
    I32 axes[8];
    I32 num_axes;
} UnsqueezeParamSpec;

typedef struct {
    DataType dt;
} CastParamSpec;

typedef struct {
    I32 axis;
    I32 num_concat;
} ScaleParamSpec;

typedef struct {
    float neg_slope;
} ReLUParamSpec;

typedef struct {
    float coeff[8];
    I32 num_coeff;
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
    U32 num_outputs;
    U32 kernel_t;
    U32 kernel_h;
    U32 kernel_w;
    U32 stride_t;
    U32 stride_h;
    U32 stride_w;
    U32 pad_before;
    U32 pad_after;
    U32 pad_top;
    U32 pad_bottom;
    U32 pad_left;
    U32 pad_right;
    U32 group;
    U32 dilatedRate_t;
    U32 dilatedRate_h;
    U32 dilatedRate_w;
    U32 num_outputs_origin;
    ConvolutionMode convolution_type;
    ActivationMode dw_activation_type;
    ActivationMode pw_activation_type;
    ActivationSpec activation_spec;
    RoundMode round_mode;
    U32 output_pad_t;
    U32 output_pad_h;
    U32 output_pad_w;
} ConvolutionParamSpec;

typedef struct {
    U32 kernel_t;
    U32 kernel_h;
    U32 kernel_w;
    U32 stride_t;
    U32 stride_h;
    U32 stride_w;
    U32 pad_before;
    U32 pad_after;
    U32 pad_top;
    U32 pad_bottom;
    U32 pad_left;
    U32 pad_right;
    RoundMode round_mode;
    PoolingMode mode;
    bool count_include_pad;
} PoolingParamSpec;

// FC's weight is reordered to NxK, K is removed dimension.
// slice parameter is for multi FC merge optimizer, default is 1.
typedef struct FullyConnectedParamSpec {
    U32 num_outputs;
    U32 num_slices = 1;
    I32 slice_point[32];
} FullyConnectedParamSpec;

typedef struct {
    I32 axis;
    float eps;
    float gama;
    float momentum;
} BatchNormParamSpec;

typedef struct {
    float eps;
    I32 axis;
    I32 axis_dim;
} InstanceNormParamSpec;

typedef struct {
    // padding on time dimension
    U32 before;
    U32 after;
    // padding on channel dimension
    U32 front;
    U32 back;
    // padding on height dimension
    U32 top;
    U32 bottom;
    // padding on width dimension
    U32 left;
    U32 right;
    float constant_value;
    PadMode pad_mode;
} PadParamSpec;

typedef struct {
    U32 num_inputs;
    U32 num_outputs;
    bool bias_term;
    bool transpose;
    I32 axis;
} EmbedParamSpec;

typedef struct {
    float scale;
    float shift;
    float power;
} PowerParamSpec;

typedef struct {
    I32 shape[8];
    I32 num_shape;
    I32 axis;
    I32 num_axes;
} ReshapeParamSpec;

typedef struct {
    I32 slice_points[8];
    U32 num_slice;
    I32 axis;
} SliceParamSpec;

typedef struct TransposeParamSpec {
    U32 axes[8];
    U32 num_axes;
    DataFormat df = DF_NCHW;
} TransposeParamSpec;

typedef struct {
    U32 num_heads;
    U32 from_sequence_length;
    U32 to_sequence_length;
} AttentionParamSpec;

typedef struct {
    RNNMode mode;
    U32 num_outputs;
    // steps >= 0 for multi-steps RNN
    // steps = -1 for RNNCell
    I32 steps;
    I32 num_projection;
    float zoneout_cell;
    float zoneout_output;

    bool bi_direction;
    float forget_bias;
    ActivationMode activation_type;
} RNNParamSpec;

typedef struct {
    BilateralSliceApplyMode mode;
} BilateralSliceApplyParamSpec;

typedef struct {
    I32 axes[8];
    I32 num_axes;
    ReductionMode mode;
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
    CheckMode mode;
} CheckParamSpec;

typedef struct {
    I32 loops;
    I32 axis;
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
    I32 attention_length;
    float mask;
    bool same_length;
} AttentionMaskParamSpec;

typedef struct {
    I32 axis;
    I32 shift_length;
} RelativeShiftParamSpec;

typedef struct {
    I32 axis;
    I32 num_concat;
} ConcatParamSpec;

typedef struct {
    I32 axis;
} SoftmaxParamSpec;

typedef struct {
    I32 begin[8];
    I32 end[8];
    I32 strides[8];
    char begin_mask[8];
    char end_mask[8];
    char ellipsis_mask[8];
    char new_axis_mask[8];
    char shrink_axis_mask[8];
    U32 num_dims;
} TfSliceParamSpec;

typedef struct {
    float min_sizes[2];
    float max_sizes[2];
    float aspect_ratios[2];
    U32 flip;
    U32 clip;
    float variances[4];
    U32 image_h;
    U32 image_w;
    float step_h;
    float step_w;
    float offset;
} PriorBoxParamSpec;

typedef struct {
    U32 num_class;
    float nms_threshold;
    U32 nms_top_k;
    U32 keep_top_k;
    float confidence_threshold;
} DetectionOutputParamSpec;

typedef struct {
    U32 num_class;
    U32 num_box;
    float confidence_threshold;
    float nms_threshold;
    float biases[18];
    U32 anchors_scale[3];
    U32 mask_group_num;
    U32 mask[9];
} Yolov3DetectionOutputParamSpec;

typedef struct {
    char symmetric[NAME_LEN];
    I32 group;
    I32 channel_before;
    I32 channel_after;
} ChannelResizeParamSpec;

typedef struct {
    I32 block_size;
} Space2DepthParamSpec;

typedef struct {
    I32 block_size;
    I8 mode[8];
} Depth2SpaceParamSpec;

typedef struct {
    I32 repeats[8];
    I32 num_repeats;
    I32 axis;
} TileParamSpec;

typedef struct {
    I32 context[8];
    I32 num_context;
    I32 index_min;
    I32 index_max;
} SpliceParamSpec;

typedef struct {
    I32 context[8];
    I32 num_context;
    I32 num_outputs;
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
    I32 axis;
    I32 largest;
    I32 sorted;
    I32 k;
} TopKParamSpec;

typedef struct {
    I32 shape[8];
    I32 num_shape;
} ExpandParamSpec;

typedef struct ScatterParamSpec {
    // axis is used for ScatterElemnts, else axis = INT_MAX
    I32 axis = INT_MAX;
} ScatterParamSpec;

typedef struct GatherParamSpec {
    // axis is used for Gather/GatherElemnts, else axis = INT_MAX
    I32 axis = INT_MAX;
    // data dimension is 7x10, index content is 6;
    // index_scalar = false, index = [6], result dimension is 1 x 10
    // index_scalar = true, index = 6, result dimension is 10
    //bool index_scalar = false;
    // element_level is used for GatherElemnts(true), else false
    bool element_level = false;
    // batch_dims for GatherND
    I32 batch_dims = 0;
} GatherParamSpec;

typedef struct {
    U32 num_heads;
    ActivationParamSpec activation_type;
} GATParamSpec;

typedef struct RoIAlignParamSpec {
    CoordinateTransMode trans_mode;
    PoolingMode mode;
    U32 output_h;
    U32 output_w;
    I32 sampling_ratio;
    float spatial_scale;
} RoIAlignParamSpec;

typedef struct GenerateProposalsParamSpec {
    I32 angle_bound_hi;
    I32 angle_bound_lo;
    I32 angle_bound_on;
    float clip_angle_thresh;
    I32 legacy_plus_one;
    float min_size;
    float nms_thresh;
    I32 post_nms_topN;
    I32 pre_nms_topN;
    float spatial_scale;
} GenerateProposalsParamSpec;

typedef struct QuantizeLinearParamSpec {
    // get the scales from input tensor
    I32 axis;
    DataType dt;
} QuantizeLinearParamSpec;

typedef struct {
    I32 axis;
    float eps;
} LayerNormParamSpec;

typedef struct RandomParamSpec {
    RandomMode mode;
    DataType dt;
    float value[2];
    float seed;
    I32 shape[8];
    I32 num_shape;
} RandomParamSpec;

typedef struct CumParamSpec {
    EltwiseMode mode;
    bool exclusive;
    bool reverse;
    I32 axis;
} CumParamSpec;

typedef struct GridSampleParamSpec {
    ResizeMode mode;
    PadMode pad_mode;
    float constant_value = 0;
    bool align_corners;
} GridSampleParamSpec;

typedef struct OneHotParamSpec {
    I32 axis;
    I32 depth;
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

typedef struct EinsumParamSpec {
    char equation_r[8];
    char equation_l[8];
    char equation_o[8];
    I32 num_equation_r;
    I32 num_equation_l;
    I32 num_equation_o;
} EinsumParamSpec;

typedef struct FlattenParamSpec {
    I32 axis;
} FlattenParamSpec;

typedef struct ConvertColorParamSpec {
    ColorSpace src;
    ColorSpace dst;
    DataType dt;
} ConvertColorParamSpec;

typedef struct LutParamSpec {
    ColorSpace type;
    ResizeMode mode;
} LutParamSpec;

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
    RandomParamSpec random_spec;
    CumParamSpec cum_spec;
    GridSampleParamSpec grid_sample_spec;
    OneHotParamSpec onehot_spec;
    NonMaxSuppressionParamSpec non_max_suppression_spec;
    ConstantOfShapeParamSpec constant_of_shape_spec;
    RangeParamSpec range_spec;
    EinsumParamSpec einsum_spec;
    FlattenParamSpec flatten_spec;
    ConvertColorParamSpec convert_color_spec;
    LutParamSpec lut_spec;
} ParameterSpec;

typedef struct {
    I32 num_scale;
    F32 *scale;
} QuantSpec;
#pragma pack()

inline I32 get_operator_parameter_size(I32 version, OperatorType operatorType)
{
    std::map<OperatorType, I32> operatorParameterSizeMap = {
        {OT_Input, sizeof(TensorDesc)},
        {OT_Conv, sizeof(ConvolutionParamSpec)},
        {OT_Deconvolution, sizeof(ConvolutionParamSpec)},
        {OT_FC, sizeof(FullyConnectedParamSpec)},
        {OT_RNN, sizeof(RNNParamSpec)},
        {OT_MatMul, sizeof(MatMulParamSpec)},
        {OT_Resize, sizeof(ResizeParamSpec)},
        {OT_BilateralSliceApply, sizeof(BilateralSliceApplyParamSpec)},
        {OT_Pooling, sizeof(PoolingParamSpec)},
        {OT_Scale, sizeof(ScaleParamSpec)},
        {OT_BatchNorm, sizeof(BatchNormParamSpec)},
        {OT_Reduction, sizeof(ReductionParamSpec)},
        {OT_ArgMax, sizeof(ArgMaxParamSpec)},
        {OT_Softmax, sizeof(SoftmaxParamSpec)},
        {OT_Clip, sizeof(ClipParamSpec)},
        {OT_Power, sizeof(PowerParamSpec)},
        {OT_Relu, sizeof(ReLUParamSpec)},
        {OT_Gather, sizeof(GatherParamSpec)},
        {OT_Embedding, sizeof(EmbedParamSpec)},
        {OT_Pad, sizeof(PadParamSpec)},
        {OT_Eltwise, sizeof(EltwiseParamSpec)},
        {OT_Concat, sizeof(ConcatParamSpec)},
        {OT_Slice, sizeof(SliceParamSpec)},
        {OT_TfSlice, sizeof(TfSliceParamSpec)},
        {OT_Cast, sizeof(CastParamSpec)},
        {OT_Transpose, sizeof(TransposeParamSpec)},
        {OT_Reshape, sizeof(ReshapeParamSpec)},
        {OT_Squeeze, sizeof(SqueezeParamSpec)},
        {OT_Unsqueeze, sizeof(UnsqueezeParamSpec)},
        {OT_Space2Depth, sizeof(Space2DepthParamSpec)},
        {OT_Depth2Space, sizeof(Depth2SpaceParamSpec)},
        {OT_ChannelResize, sizeof(ChannelResizeParamSpec)},
        {OT_PreAllocatedMemory, sizeof(PreAllocatedMemoryParamSpec)},
        {OT_SharedWeight, sizeof(SharedWeightParamSpec)},
        {OT_Copy, sizeof(CopyParamSpec)},
        {OT_Check, sizeof(CheckParamSpec)},
        {OT_Repeat, sizeof(RepeatParamSpec)},
        {OT_Attention, sizeof(AttentionParamSpec)},
        {OT_AttentionMask, sizeof(AttentionMaskParamSpec)},
        {OT_RelativePositionEmbedding, sizeof(EmbedParamSpec)},
        {OT_RelativeShift, sizeof(RelativeShiftParamSpec)},
        {OT_PriorBox, sizeof(PriorBoxParamSpec)},
        {OT_DetectionOutput, sizeof(DetectionOutputParamSpec)},
        {OT_Yolov3DetectionOutput, sizeof(Yolov3DetectionOutputParamSpec)},
        {OT_MultiHeadAttention, sizeof(MultiHeadAttentionParamSpec)},
        {OT_Tile, sizeof(TileParamSpec)},
        {OT_Splice, sizeof(SpliceParamSpec)},
        {OT_Tdnn, sizeof(TdnnParamSpec)},
        {OT_TopK, sizeof(TopKParamSpec)},
        {OT_Expand, sizeof(ExpandParamSpec)},
        {OT_InstanceNorm, sizeof(InstanceNormParamSpec)},
        {OT_Scatter, sizeof(ScatterParamSpec)},
        {OT_LogSoftmax, sizeof(SoftmaxParamSpec)},
        {OT_GenerateProposals, sizeof(GenerateProposalsParamSpec)},
        {OT_RoIAlign, sizeof(RoIAlignParamSpec)},
        {OT_GAT, sizeof(GATParamSpec)},
        {OT_QuantizeLinear, sizeof(QuantizeLinearParamSpec)},
        {OT_LayerNorm, sizeof(LayerNormParamSpec)},
        {OT_QuantizeLinear, sizeof(QuantizeLinearParamSpec)},
        {OT_Cum, sizeof(CumParamSpec)},
        {OT_GridSample, sizeof(GridSampleParamSpec)},
        {OT_OneHot, sizeof(OneHotParamSpec)},
        {OT_NonMaxSuppression, sizeof(NonMaxSuppressionParamSpec)},
        {OT_Range, sizeof(RangeParamSpec)},
        {OT_ConstantOfShape, sizeof(ConstantOfShapeParamSpec)},
        {OT_Elu, sizeof(ReLUParamSpec)},
        {OT_Einsum, sizeof(EinsumParamSpec)},
        {OT_UnPooling, sizeof(PoolingParamSpec)},
        {OT_Random, sizeof(RandomParamSpec)},
        {OT_Flatten, sizeof(FlattenParamSpec)},
        {OT_ConvertColor, sizeof(ConvertColorParamSpec)},
    };
    I32 size;
    if (operatorParameterSizeMap.find(operatorType) == operatorParameterSizeMap.end()) {
        size = 0;
    } else {
        size = operatorParameterSizeMap[operatorType];
    }
    if (version == 20201120) {
        if (operatorType == OT_Conv || operatorType == OT_Deconvolution) {
            size -= 3 * sizeof(U32);
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
    if (version < 20220126) {
        if (operatorType == OT_Pooling) {
            size -= sizeof(I32);
        }
    }
    if (version < 20220817) {
        if (operatorType == OT_Resize) {
            size = size - sizeof(float) - 2 * sizeof(I32);
        }
    }
    if (version < 20220831) {
        if (operatorType == OT_Input || operatorType == OT_SharedWeight) {
            size -= (DIM_LEN - 6) * sizeof(U32);
        }
        if (operatorType == OT_Scatter || operatorType == OT_Gather ||
            operatorType == OT_PreAllocatedMemory) {
            return -1;
        }
    }
    return size;
}

inline ConvolutionParamSpec createConvolutionParamSpec(U32 group,
    U32 kernel_t,
    U32 kernel_h,
    U32 kernel_w,
    U32 stride_t,
    U32 stride_h,
    U32 stride_w,
    U32 pad_before,
    U32 pad_after,
    U32 pad_top,
    U32 pad_bottom,
    U32 pad_left,
    U32 pad_right,
    U32 dilateRate_t,
    U32 dilateRate_h,
    U32 dilateRate_w,
    U32 num_outputs,
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
    U32 num_outputs, U32 num_slices, I32 *slice_point)
{
    FullyConnectedParamSpec p;
    p.num_outputs = num_outputs;
    p.num_slices = num_slices;
    if (num_slices > 1 && slice_point != nullptr) {
        for (U32 i = 0; i < num_slices; i++) {
            p.slice_point[i] = slice_point[i];
        }
    }
    return p;
}

inline PoolingParamSpec createPoolingParamSpec(PoolingMode pm,
    U32 kernel_t,
    U32 kernel_h,
    U32 kernel_w,
    U32 stride_t,
    U32 stride_h,
    U32 stride_w,
    U32 pad_before,
    U32 pad_after,
    U32 pad_top,
    U32 pad_bottom,
    U32 pad_left,
    U32 pad_right,
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

inline ReshapeParamSpec createReshapeParamSpec(I32 *shape, I32 num_shape, I32 axis, I32 num_axes)
{
    ReshapeParamSpec p;
    p.num_shape = num_shape;
    p.axis = axis;
    p.num_axes = num_axes;
    if (shape != nullptr && num_shape != 0) {
        for (I32 i = 0; i < num_shape; i++) {
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

inline SqueezeParamSpec createSqueezeParamSpec(I32 *axes, I32 num_axes)
{
    SqueezeParamSpec p;
    p.num_axes = num_axes;
    if (axes != nullptr && num_axes != 0) {
        for (I32 i = 0; i < num_axes; i++) {
            p.axes[i] = axes[i];
        }
    }
    return p;
}
#endif
