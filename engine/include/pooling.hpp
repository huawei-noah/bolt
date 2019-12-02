// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _POOLING_H
#define _POOLING_H
#include <math.h>
#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "model_tools.h"

template <Arch A>
class Pooling: public Operator<A> {
public:

/**
 * @param mode
 * @param ks
 * @param stride
 * @param padding
 * @param name
 */
    Pooling(PoolingMode mode, U32 ks, U32 stride, U32 padding, RoundMode rm)
    {
        this->mode = mode;
        this->kernelSize = ks;
        this->stride = stride;
        this->padding = padding;
        this->rm = rm;
        this->set_op_type(OT_Pooling);
    }

    PoolingDesc create_PoolingDesc(PoolingMode pm, U32 stride, U32 padding, U32 kernelSize, RoundMode rm)
    {
        PoolingDesc poolingDesc;
        poolingDesc.pm = pm;
        poolingDesc.stride = stride;
        poolingDesc.padding = padding;
        poolingDesc.kernelSize = kernelSize;
        poolingDesc.rm = rm;
        return poolingDesc;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        PoolingDesc poolingDesc = create_PoolingDesc(this->mode, this->stride, this->padding, this->kernelSize, this->rm);
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        F16 scales[2];
        if (DT_I8 == inputDesc.dt) {
            scales[0] = inputTensor.get_scale();
        }
        CHECK_STATUS(pooling(inputDesc, inputTensor.get_val().get(),
                             poolingDesc, scales,
                             outputDesc, outputTensor.get_val().get(),
                             A));
        if (DT_I8 == inputDesc.dt) {
            outputTensor.set_scale(scales[1]);
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    void set_kernelSize(U32 globalKernelSize) {    
        this->kernelSize = globalKernelSize;
    }

    void set_stride(U32 globalStride) {
        this->stride = globalStride;
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        auto inDim = inDims[0];
        DataType dt;
        DataFormat df;
        U32 width ;
        U32 height;
        U32 numChannels;
        U32 num;
        CHECK_STATUS_WITH_RETURN(tensor4dGet(inDim, &dt, &df, &num, &numChannels, &height, &width));

        TensorDesc inputDesc = tensor4df(dt, df, num, numChannels, height, width);
        if (this->kernelSize == 0) {
            set_kernelSize(height);
            set_stride(1);
        }
        PoolingDesc poolingDesc = create_PoolingDesc(this->mode, this->stride, this->padding, this->kernelSize, this->rm);
        CHECK_STATUS_WITH_RETURN(pooling_infer_output_size(inputDesc, poolingDesc, &((*outDims)[0])));
        return SUCCESS;
    }


private:
    PoolingMode mode;
    RoundMode rm;
    U32 kernelSize;
    U32 stride;
    U32 padding;
};

#endif //_POOLING_H
