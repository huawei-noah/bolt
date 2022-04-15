// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MODELADAPTEE
#define _H_MODELADAPTEE

#include <string>
#include <fstream>
#include <map>
#include <vector>

#include "uni.h"
#include "model_common.h"

#define REGISTER_EMPTY_ADAPT_OPERATOR(name)                                                \
    virtual ParameterSpec name()                                                           \
    {                                                                                      \
        UNI_WARNING_LOG("%s use default(0) operator parmeter as return.\n", __FUNCTION__); \
        ParameterSpec ps;                                                                  \
        UNI_MEMSET(&ps, 0, sizeof(ps));                                                    \
        return ps;                                                                         \
    }

class ModelAdaptee {
public:
    virtual EE adapt(std::string dir, std::string mfn, ModelSpec *ms)
    {
        CHECK_STATUS(parse_file(dir, mfn));
        CHECK_STATUS(adapt_operators(ms));
        CHECK_STATUS(adapt_weights(ms));
        return SUCCESS;
    }

    ModelAdaptee()
    {}

    virtual ~ModelAdaptee()
    {}

protected:
    virtual EE parse_file(std::string dir, std::string mfn) = 0;

    virtual EE adapt_operators(ModelSpec *ms) = 0;

    virtual EE adapt_weights(ModelSpec *ms) = 0;

    virtual EE adapt_operator(OperatorType type, ParameterSpec *ps)
    {
        typedef ParameterSpec (ModelAdaptee::*AdaptOperatorFunction)();
        std::map<OperatorType, AdaptOperatorFunction> functions = {
            {OT_Conv, &ModelAdaptee::adapt_Conv},
            {OT_Deconvolution, &ModelAdaptee::adapt_Deconvolution},
            {OT_FC, &ModelAdaptee::adapt_Fc},
            {OT_RNN, &ModelAdaptee::adapt_RNN},
            {OT_MatMul, &ModelAdaptee::adapt_MatMul},
            {OT_Resize, &ModelAdaptee::adapt_Resize},
            {OT_Pooling, &ModelAdaptee::adapt_Pooling},
            {OT_Scale, &ModelAdaptee::adapt_Scale},
            {OT_PRelu, &ModelAdaptee::adapt_PRelu},
            {OT_BatchNorm, &ModelAdaptee::adapt_BatchNorm},
            {OT_InstanceNorm, &ModelAdaptee::adapt_InstanceNorm},
            {OT_LayerNorm, &ModelAdaptee::adapt_LayerNorm},
            {OT_Reduction, &ModelAdaptee::adapt_Reduction},
            {OT_ArgMax, &ModelAdaptee::adapt_ArgMax},
            {OT_Softmax, &ModelAdaptee::adapt_Softmax},
            {OT_LogSoftmax, &ModelAdaptee::adapt_Softmax},
            {OT_Clip, &ModelAdaptee::adapt_Clip},
            {OT_Power, &ModelAdaptee::adapt_Power},
            {OT_Relu, &ModelAdaptee::adapt_Relu},
            {OT_Gather, &ModelAdaptee::adapt_Gather},
            {OT_Embedding, &ModelAdaptee::adapt_Embedding},
            {OT_Pad, &ModelAdaptee::adapt_Pad},
            {OT_Eltwise, &ModelAdaptee::adapt_Eltwise},
            {OT_Concat, &ModelAdaptee::adapt_Concat},
            {OT_Slice, &ModelAdaptee::adapt_Slice},
            {OT_TfSlice, &ModelAdaptee::adapt_TfSlice},
            {OT_Cast, &ModelAdaptee::adapt_Cast},
            {OT_Transpose, &ModelAdaptee::adapt_Transpose},
            {OT_Reshape, &ModelAdaptee::adapt_Reshape},
            {OT_Squeeze, &ModelAdaptee::adapt_Squeeze},
            {OT_Unsqueeze, &ModelAdaptee::adapt_Unsqueeze},
            {OT_Space2Depth, &ModelAdaptee::adapt_Space2Depth},
            {OT_Depth2Space, &ModelAdaptee::adapt_Depth2Space},
            {OT_PreAllocatedMemory, &ModelAdaptee::adapt_PreAllocatedMemory},
            {OT_SharedWeight, &ModelAdaptee::adapt_SharedWeight},
            {OT_Copy, &ModelAdaptee::adapt_Copy},
            {OT_Check, &ModelAdaptee::adapt_Check},
            {OT_Repeat, &ModelAdaptee::adapt_Repeat},
            {OT_Attention, &ModelAdaptee::adapt_Attention},
            {OT_AttentionMask, &ModelAdaptee::adapt_AttentionMask},
            {OT_RelativePositionEmbedding, &ModelAdaptee::adapt_RelativePositionEmbedding},
            {OT_RelativeShift, &ModelAdaptee::adapt_RelativeShift},
            {OT_PriorBox, &ModelAdaptee::adapt_PriorBox},
            {OT_DetectionOutput, &ModelAdaptee::adapt_DetectionOutput},
            {OT_Yolov3DetectionOutput, &ModelAdaptee::adapt_Yolov3DetectionOutput},
            {OT_Tile, &ModelAdaptee::adapt_Tile},
            {OT_Splice, &ModelAdaptee::adapt_Splice},
            {OT_SoftPlus, &ModelAdaptee::adapt_SoftPlus},
            {OT_Exp, &ModelAdaptee::adapt_Exp},
            {OT_Tdnn, &ModelAdaptee::adapt_Tdnn},
            {OT_TopK, &ModelAdaptee::adapt_TopK},
            {OT_SpaceToBatchNd, &ModelAdaptee::adapt_SpaceToBatchNd},
            {OT_BatchToSpaceNd, &ModelAdaptee::adapt_BatchToSpaceNd},
            {OT_Where, &ModelAdaptee::adapt_Where},
            {OT_Expand, &ModelAdaptee::adapt_Expand},
            {OT_Scatter, &ModelAdaptee::adapt_Scatter},
            {OT_Select, &ModelAdaptee::adapt_Select},
            {OT_RoIAlign, &ModelAdaptee::adapt_RoIAlign},
            {OT_GridSample, &ModelAdaptee::adapt_GridSample},
            {OT_GenerateProposals, &ModelAdaptee::adapt_GenerateProposals},
            {OT_OneHot, &ModelAdaptee::adapt_OneHot},
            {OT_CumSum, &ModelAdaptee::adapt_CumSum},
            {OT_NonMaxSuppression, &ModelAdaptee::adapt_NonMaxSuppression},
            {OT_ConstantOfShape, &ModelAdaptee::adapt_ConstantOfShape},
            {OT_Range, &ModelAdaptee::adapt_Range},
        };
        if (functions.find(type) == functions.end()) {
            UNI_MEMSET(ps, 0, sizeof(*ps));
        } else {
            *ps = (this->*(functions[type]))();
        }
        return SUCCESS;
    }

    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Conv)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Deconvolution)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Fc)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_RNN)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_MatMul)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Resize)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Pooling)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Scale)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_PRelu)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_BatchNorm)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_InstanceNorm)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_LayerNorm)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Reduction)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_ArgMax)

    virtual ParameterSpec adapt_Softmax()
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        ps.softmax_spec.axis = -1;
        return ps;
    }

    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Clip)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Power)

    virtual ParameterSpec adapt_Relu()
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        ps.relu_spec.neg_slope = 0;
        return ps;
    }

    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Gather)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Embedding)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Pad)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Eltwise)

    virtual ParameterSpec adapt_Concat()
    {
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        ps.concat_spec.axis = 1;
        return ps;
    }

    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Slice)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_TfSlice)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Cast)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Transpose)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Reshape)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Squeeze)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Unsqueeze)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Space2Depth)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Depth2Space)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_PreAllocatedMemory)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_SharedWeight)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Copy)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Check)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Repeat)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Attention)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_AttentionMask)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_RelativePositionEmbedding)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_RelativeShift)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_PriorBox)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_DetectionOutput)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Yolov3DetectionOutput)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Tile)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Splice)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_SoftPlus)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Exp)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Tdnn)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_TopK)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_SpaceToBatchNd)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_BatchToSpaceNd)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Where)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Expand)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Scatter)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Select)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_RoIAlign)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_GenerateProposals)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_GridSample)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_OneHot)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_CumSum)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_NonMaxSuppression)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_ConstantOfShape)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Range)
};
#endif
