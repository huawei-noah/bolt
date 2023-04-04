// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_OPERATOR_TYPE
#define _H_OPERATOR_TYPE

#include <stdint.h>

// please add OperatorType and OperatorTypeName at the same time
typedef enum OperatorType : uint32_t {
    OT_None = 0,
    OT_Input = 1,
    OT_Conv = 2,
    OT_Deconvolution = 3,
    OT_FC = 4,
    OT_RNN = 5,
    OT_MatMul = 6,
    OT_Resize = 7,
    OT_BilateralSliceApply = 8,
    OT_Pooling = 9,
    OT_Scale = 10,
    OT_PRelu = 11,
    OT_BatchNorm = 12,
    OT_LayerNorm = 13,
    OT_L2Norm = 14,
    OT_Reduction = 15,
    OT_ArgMax = 16,
    OT_Softmax = 17,
    OT_SoftmaxWithLoss = 18,
    OT_LogSoftmax = 19,

    OT_Clip = 20,
    OT_Power = 21,
    OT_Sigmoid = 22,
    OT_Relu = 23,
    OT_Relu6 = 24,
    OT_HSwish = 25,
    OT_HSigmoid = 26,
    OT_Gelu = 27,
    OT_TanH = 28,
    OT_Mish = 29,
    OT_Erf = 30,

    OT_Gather = 31,
    OT_Embedding = 32,
    OT_Pad = 33,
    OT_Eltwise = 34,
    OT_Concat = 35,
    OT_Slice = 36,
    OT_TfSlice = 37,

    OT_Cast = 38,
    OT_Shape = 39,
    OT_ConstantOfShape = 40,
    OT_Transpose = 41,
    OT_Reshape = 42,
    OT_Squeeze = 43,
    OT_Unsqueeze = 44,
    OT_Space2Depth = 45,
    OT_Depth2Space = 46,
    OT_Constant = 47,

    OT_ChannelResize = 48,
    OT_PreAllocatedMemory = 49,
    OT_SharedWeight = 50,
    OT_Copy = 51,
    OT_Check = 52,
    OT_Repeat = 53,
    OT_Jump = 54,
    OT_Attention = 55,
    OT_AttentionMask = 56,
    OT_RelativePositionEmbedding = 57,
    OT_RelativeShift = 58,
    OT_PriorBox = 59,
    OT_DetectionOutput = 60,
    OT_Yolov3DetectionOutput = 61,
    OT_MultiHeadAttention = 62,
    OT_SqDiff = 63,
    OT_Tile = 64,
    OT_Splice = 65,
    OT_Neg = 66,
    OT_Greater = 67,
    OT_Where = 68,
    OT_Softplus = 69,

    OT_Exp = 70,
    OT_OneHot = 71,
    OT_Tdnn = 72,
    OT_Dropout = 73,
    OT_TopK = 74,
    OT_SpaceToBatchNd = 75,
    OT_BatchToSpaceNd = 76,
    OT_Abs = 77,
    OT_NonZero = 78,
    OT_Sign = 79,

    OT_HSwishNoDiv = 80,
    OT_InstanceNorm = 81,
    OT_Expand = 82,
    OT_Scatter = 83,
    OT_Select = 84,
    OT_Not = 85,
    OT_Reciprocal = 86,
    OT_Log = 87,
    OT_GenerateProposals = 88,
    OT_RoIAlign = 89,

    OT_GAT = 90,
    OT_QuantizeLinear = 91,
    OT_Round = 92,
    OT_Floor = 93,
    OT_Ceil = 94,
    OT_Random = 95,
    OT_Cum = 96,
    OT_GridSample = 97,
    OT_NonMaxSuppression = 98,
    OT_Range = 99,

    OT_Swish = 100,
    OT_Sin = 101,
    OT_Cos = 102,
    OT_Elu = 103,
    OT_Einsum = 104,
    OT_UnPooling = 105,
    OT_Flatten = 106,
    OT_ConvertColor = 107,
    OT_Lut = 108,
    OT_LutPreprocess = 109,
} OperatorType;

inline const char *const *OperatorTypeName()
{
    static const char *const names[] = {"OT_None", "OT_Input", "OT_Conv", "OT_Deconvolution",
        "OT_FC", "OT_RNN", "OT_MatMul", "OT_Resize", "OT_BilateralSliceApply", "OT_Pooling",

        "OT_Scale", "OT_PRelu", "OT_BatchNorm", "OT_LayerNorm", "OT_L2Norm",
        "OT_Reduction", "OT_ArgMax", "OT_Softmax", "OT_SoftmaxWithLoss", "OT_LogSoftmax",

        "OT_Clip", "OT_Power", "OT_Sigmoid", "OT_Relu", "OT_Relu6", "OT_HSwish", "OT_HSigmoid",
        "OT_Gelu", "OT_TanH", "OT_Mish",

        "OT_Erf", "OT_Gather", "OT_Embedding", "OT_Pad", "OT_Eltwise", "OT_Concat", "OT_Slice",
        "OT_TfSlice", "OT_Cast", "OT_Shape",

        "OT_ConstantOfShape", "OT_Transpose", "OT_Reshape", "OT_Squeeze", "OT_Unsqueeze",
        "OT_Space2Depth", "OT_Depth2Space", "OT_Constant", "OT_ChannelResize",
        "OT_PreAllocatedMemory",

        "OT_SharedWeight", "OT_Copy", "OT_Check", "OT_Repeat", "OT_Jump", "OT_Attention",
        "OT_AttentionMask", "OT_RelativePositionEmbedding", "OT_RelativeShift", "OT_PriorBox",

        "OT_DetectionOutput", "OT_Yolov3DetectionOutput", "OT_MultiHeadAttention", "OT_SqDiff",
        "OT_Tile", "OT_Splice", "OT_Neg", "OT_Greater", "OT_Where", "OT_Softplus", "OT_Exp",

        "OT_OneHot", "OT_Tdnn", "OT_Dropout", "OT_TopK", "OT_SpaceToBatchNd", "OT_BatchToSpaceNd",
        "OT_Abs", "OT_NonZero", "OT_Sign", "OT_HSwishNoDiv",

        "OT_InstanceNorm", "OT_Expand", "OT_Scatter", "OT_Select", "OT_Not", "OT_Reciprocal",
        "OT_Log", "OT_GenerateProposals", "OT_RoIAlign",

        "OT_GAT", "OT_QuantizeLinear", "OT_Round", "OT_Floor", "OT_Ceil", "OT_Random", "OT_Cum",
        "OT_GridSample", "OT_NonMaxSuppression", "OT_Range",

        "OT_Swish", "OT_Sin", "OT_Cos", "OT_Elu", "OT_Einsum", "OT_UnPooling", "OT_Flatten",
        "OT_ConvertColor", "OT_Lut", "OT_LutPreprocess"};
    return names;
}
#endif
