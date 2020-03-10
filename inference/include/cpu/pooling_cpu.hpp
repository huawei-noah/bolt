// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _POOLING_CPU_H
#define _POOLING_CPU_H
#include <math.h>
#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"
#include "pooling.hpp"

class PoolingCPU: public Pooling {
public:

/**
 * @param mode
 * @param ks
 * @param stride
 * @param padding
 * @param name
 */
    PoolingCPU(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW, U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm): 
        Pooling(mode, ksH, ksW, strideH, strideW, paddingT, paddingB, paddingL, paddingR, rm){}

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        PoolingDesc poolingDesc = Pooling::create_PoolingDesc(this->mode, this->kernelSizeH, this->kernelSizeW, this->strideH, this->strideW,
            this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->rm);
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        short scales[2];
#ifdef _USE_INT8
        if (DT_I8 == inputDesc.dt) {
            F16* scale = (F16*)scales;
            scale[0] = inputTensor.get_scale();
        }
#endif
        CHECK_STATUS(pooling(inputDesc, inputTensor.get_val(),
                             poolingDesc, scales,
                             outputDesc, outputTensor.get_val(),
                             this->schedule));
#ifdef _USE_INT8
        if (DT_I8 == inputDesc.dt) {
            F16 *scale = (F16*)scales;
            outputTensor.set_scale(scale[1]);
        }
#endif

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }


    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        auto inDim = inDims[0];
        DataType dt;
        DataFormat df;
        U32 width ;
        U32 height;
        U32 numChannels;
        U32 num;
        CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &num, &numChannels, &height, &width));

        TensorDesc inputDesc = tensor4df(dt, df, num, numChannels, height, width);
        if (this->kernelSizeH == 0 && this->kernelSizeW == 0) {
            Pooling::set_kernelSize(height, width);
            Pooling::set_stride(1, 1);
        }
        PoolingDesc poolingDesc = Pooling::create_PoolingDesc(this->mode, this->kernelSizeH, this->kernelSizeW, this->strideH, this->strideW,
            this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->rm);
        CHECK_STATUS(pooling_infer_output_size(inputDesc, poolingDesc, &((*outDims)[0]), this->schedule));
        return SUCCESS;
    }

};

#endif //_POOLINGCPU_H
