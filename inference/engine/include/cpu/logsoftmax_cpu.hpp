// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _LOGSOFTMAX_CPU_H
#define _LOGSOFTMAX_CPU_H

#include "cpu/softmax_cpu.hpp"

// LOGSOFTMAX_CPU_V1: y = log(softmax(x))
// LOGSOFTMAX_CPU_V2: y = (x - reduce_max) - log(reduce_sum(exp(x - reduce_max)))
class LogSoftmaxCPU : public SoftmaxCPU {
public:
    LogSoftmaxCPU(DataType dt, SoftmaxParamSpec p) : SoftmaxCPU(dt, p)
    {
#ifndef LOGSOFTMAX_CPU_V1
        TensorDesc maskDesc;
        maskDesc.nDims = 0;
        reductionMask.resize(maskDesc);
#endif
    }

    OperatorType get_type() override
    {
        return OT_LogSoftmax;
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<LogSoftmaxCPU> mem =
            std::shared_ptr<LogSoftmaxCPU>(new LogSoftmaxCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
#ifdef LOGSOFTMAX_CPU_V1
        ActivationParamSpec activationDesc;
        activationDesc.mode = ACTIVATION_LOG;
        CHECK_STATUS(
            softmax(inputTensors[0], this->p, this->temp, outputTensors[0], &this->archInfo));
        CHECK_STATUS(
            activation(outputTensors[0], activationDesc, outputTensors[0], &this->archInfo));
#else
        Tensor tmp, newInput;
        U8 *data = (U8 *)((CpuMemory *)(this->temp.get_memory()))->get_ptr();
        std::shared_ptr<U8> p1(data, [](U8 *ptr) {});
        newInput.resize(inputTensors[0].get_desc());
        ((CpuMemory *)(reductionResult.get_memory()))->set_shared_ptr(p1);
        std::shared_ptr<U8> p2(data + reductionResult.bytes(), [](U8 *ptr) {});
        ((CpuMemory *)(newInput.get_memory()))->set_shared_ptr(p2);
        std::shared_ptr<U8> p3(data + reductionResult.bytes() + newInput.bytes(), [](U8 *ptr) {});
        ((CpuMemory *)(tmp.get_memory()))->set_shared_ptr(p3);

        ReductionParamSpec reductionSpec = get_reduction_param();
        reductionSpec.reduction_mode = REDUCTION_MAX;
        CHECK_STATUS(reduction(
            inputTensors[0], reductionMask, reductionSpec, tmp, reductionResult, &this->archInfo));
        EltwiseParamSpec eltwiseSpec;
        eltwiseSpec.elt_mode = ELTWISE_SUB;
        eltwiseSpec.activation_type = ACTIVATION_NULL;
        std::vector<Tensor> tmpInput = {inputTensors[0], reductionResult};
        CHECK_STATUS(eltwise(tmpInput, eltwiseSpec, tmp, newInput, &this->archInfo));

        ActivationParamSpec activationSpec;
        activationSpec.mode = ACTIVATION_EXP;
        CHECK_STATUS(activation(newInput, activationSpec, outputTensors[0], &this->archInfo));

        CHECK_STATUS(reduction(outputTensors[0], reductionMask, get_reduction_param(), tmp,
            reductionResult, &this->archInfo));

        activationSpec.mode = ACTIVATION_LOG;
        CHECK_STATUS(activation(reductionResult, activationSpec, reductionResult, &this->archInfo));

        tmpInput = {newInput, reductionResult};
        CHECK_STATUS(eltwise(tmpInput, eltwiseSpec, tmp, outputTensors[0], &this->archInfo));
#endif
    }

#ifndef LOGSOFTMAX_CPU_V1
    ReductionParamSpec get_reduction_param()
    {
        ReductionParamSpec reductionSpec;
        reductionSpec.axes_num = 1;
        reductionSpec.axes[0] = this->p.axis;
        reductionSpec.reduction_mode = REDUCTION_SUM;
        reductionSpec.keep_dim = true;
        reductionSpec.coeff = 1;
        return reductionSpec;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes1 = 0, bytes2 = 0;
        CHECK_STATUS(reduction_infer_output_size(&(inputTensors[0]), reductionMask,
            get_reduction_param(), &reductionResult, &this->archInfo));

        CHECK_STATUS(reduction_infer_forward_tmp_bytes(
            inputTensors[0], get_reduction_param(), reductionResult, &bytes1, &this->archInfo));

        std::vector<Tensor> tmpInput = {inputTensors[0], reductionResult};
        CHECK_STATUS(
            eltwise_infer_forward_tmp_bytes(tmpInput, inputTensors[0], &bytes2, &this->archInfo));
        return inputTensors[0].bytes() + reductionResult.bytes() + UNI_MAX(bytes1, bytes2);
    }

private:
    Tensor reductionResult;
    Tensor reductionMask;
#endif
};
#endif  // LOGSOFTMAX_CPU_H
