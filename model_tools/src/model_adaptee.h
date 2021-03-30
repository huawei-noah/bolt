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

#include "uni.h"
#include "model_common.h"

#define REGISTER_EMPTY_ADAPT_OPERATOR(name)                                                \
    virtual ParameterSpec name()                                                           \
    {                                                                                      \
        UNI_WARNING_LOG("%s use default(0) operator parmeter as return.\n", __FUNCTION__); \
        ParameterSpec curPs;                                                               \
        memset(&curPs, 0, sizeof(ParameterSpec));                                          \
        return curPs;                                                                      \
    }

class ModelAdaptee {
public:
    virtual EE adapt(std::string dir, std::string mfn, ModelSpec *ms)
    {
        EE ret = parse_file(dir, mfn);
        ret = adapt_operators(ms);
        ret = adapt_weights(ms);
        return ret;
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
        if (type == OT_Conv) {
            *ps = adapt_Conv();
        } else if (type == OT_Deconvolution) {
            *ps = adapt_Deconvolution();
        } else if (type == OT_FC) {
            *ps = adapt_Fc();
        } else if (type == OT_RNN) {
            *ps = adapt_RNN();
        } else if (type == OT_MatMul) {
            *ps = adapt_MatMul();
        } else if (type == OT_Resize) {
            *ps = adapt_Resize();
        } else if (type == OT_Pooling) {
            *ps = adapt_Pooling();
        } else if (type == OT_Scale) {
            *ps = adapt_Scale();
        } else if (type == OT_PRelu) {
            *ps = adapt_PRelu();
        } else if (type == OT_BatchNorm) {
            *ps = adapt_BatchNorm();
        } else if (type == OT_LayerNorm) {
            *ps = adapt_LayerNorm();
        } else if (type == OT_Reduction) {
            *ps = adapt_Reduction();
        } else if (type == OT_ArgMax) {
            *ps = adapt_ArgMax();
        } else if (type == OT_Softmax) {
            *ps = adapt_Softmax();
        } else if (type == OT_Clip) {
            *ps = adapt_Clip();
        } else if (type == OT_Power) {
            *ps = adapt_Power();
        } else if (type == OT_Relu) {
            *ps = adapt_Relu();
        } else if (type == OT_Gather) {
            *ps = adapt_Gather();
        } else if (type == OT_Embedding) {
            *ps = adapt_Embedding();
        } else if (type == OT_Pad) {
            *ps = adapt_Pad();
        } else if (type == OT_Eltwise) {
            *ps = adapt_Eltwise();
        } else if (type == OT_Concat) {
            *ps = adapt_Concat();
        } else if (type == OT_Slice) {
            *ps = adapt_Slice();
        } else if (type == OT_TfSlice) {
            *ps = adapt_TfSlice();
        } else if (type == OT_Cast) {
            *ps = adapt_Cast();
        } else if (type == OT_Transpose) {
            *ps = adapt_Transpose();
        } else if (type == OT_Reshape) {
            *ps = adapt_Reshape();
        } else if (type == OT_Squeeze) {
            *ps = adapt_Squeeze();
        } else if (type == OT_Unsqueeze) {
            *ps = adapt_Unsqueeze();
        } else if (type == OT_Space2Depth) {
            *ps = adapt_Space2Depth();
        } else if (type == OT_Depth2Space) {
            *ps = adapt_Depth2Space();
        } else if (type == OT_PreAllocatedMemory) {
            *ps = adapt_PreAllocatedMemory();
        } else if (type == OT_SharedWeight) {
            *ps = adapt_SharedWeight();
        } else if (type == OT_Copy) {
            *ps = adapt_Copy();
        } else if (type == OT_Check) {
            *ps = adapt_Check();
        } else if (type == OT_Repeat) {
            *ps = adapt_Repeat();
        } else if (type == OT_Attention) {
            *ps = adapt_Attention();
        } else if (type == OT_AttentionMask) {
            *ps = adapt_AttentionMask();
        } else if (type == OT_RelativePositionEmbedding) {
            *ps = adapt_RelativePositionEmbedding();
        } else if (type == OT_RelativeShift) {
            *ps = adapt_RelativeShift();
        } else if (type == OT_PriorBox) {
            *ps = adapt_PriorBox();
        } else if (type == OT_DetectionOutput) {
            *ps = adapt_DetectionOutput();
        } else if (type == OT_Yolov3DetectionOutput) {
            *ps = adapt_Yolov3DetectionOutput();
        } else if (type == OT_Tile) {
            *ps = adapt_Tile();
        } else if (type == OT_Splice) {
            *ps = adapt_Splice();
        } else if (type == OT_SoftPlus) {
            *ps = adapt_SoftPlus();
        } else if (type == OT_Exp) {
            *ps = adapt_Exp();
        } else if (type == OT_Tdnn) {
            *ps = adapt_Tdnn();
        } else if (type == OT_TopK) {
            *ps = adapt_TopK();
        } else if (type == OT_SpaceToBatchNd) {
            *ps = adapt_SpaceToBatchNd();
        } else if (type == OT_BatchToSpaceNd) {
            *ps = adapt_BatchToSpaceNd();
        } else if (type == OT_Where) {
            *ps = adapt_Where();
        } else {
            memset(ps, 0, sizeof(ParameterSpec));
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
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_LayerNorm)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Reduction)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_ArgMax)

    virtual ParameterSpec adapt_Softmax()
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(ParameterSpec));
        curPs.softmax_spec.axis = -1;
        return curPs;
    }

    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Clip)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Power)

    virtual ParameterSpec adapt_Relu()
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(ParameterSpec));
        curPs.relu_spec.neg_slope = 0;
        return curPs;
    }

    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Gather)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Embedding)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Pad)
    REGISTER_EMPTY_ADAPT_OPERATOR(adapt_Eltwise)

    virtual ParameterSpec adapt_Concat()
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(ParameterSpec));
        curPs.concat_spec.axis = 1;
        return curPs;
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
};
#endif
