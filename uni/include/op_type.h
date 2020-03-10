// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_OP_TYPE
#define _H_OP_TYPE

#ifdef __cplusplus
extern "C" {
#endif

// please add OperatorType and OperatorTypeName at the same time
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

        OT_BilateralSliceApply,
        OT_Resize,
        OT_Deconvolution,
        OT_Constant,
        OT_ResizeBilinear,
        OT_PreAllocatedMemory,
        OT_SharedWeight,
        OT_Copy,
        OT_Check,
        OT_Repeat,
        OT_AxisMean,
        OT_ArgMax,
        OT_None,
        OT_Interp,
        OT_Flatten
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

            "OT_BilateralSliceApply",
            "OT_Resize",
            "OT_Deconvolution",
            "OT_Constant",
            "OT_ResizeBilinear",
            "OT_PreAllocatedMemory",
            "OT_SharedWeight",
            "OT_Copy",
            "OT_Check",
            "OT_Repeat",
            "OT_AxisMean",
            "OT_ArgMax",
            "OT_None",
            "OT_Interp",
            "OT_Flatten"
        };
        return names;
    }
    
#ifdef __cplusplus
}
#endif

#endif
