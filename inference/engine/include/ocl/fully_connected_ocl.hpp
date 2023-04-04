// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _FULLY_CONNECTED_OCL_H
#define _FULLY_CONNECTED_OCL_H

#include "fully_connected.hpp"

class FullyConnectedOCL : public FullyConnected {
public:
    FullyConnectedOCL(DataType dt, FullyConnectedParamSpec p, U32 numInput)
        : FullyConnected(dt, p, numInput)
    {
        INIT_GPU_INFO(&this->runInfo)
    }

    ~FullyConnectedOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<FullyConnectedOCL> mem = std::shared_ptr<FullyConnectedOCL>(
            new FullyConnectedOCL(this->dt, this->p, this->numInput));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        if (this->ws.bytes_of_weight > 0) {
            TensorDesc weightDesc =
                tensor2df(this->dt, DF_NORMAL, this->p.num_outputs, this->numInput);
            Tensor modelWeightTensor = Tensor(OCLMem);
            modelWeightTensor.resize(weightDesc);
            this->weightTensors.push_back(modelWeightTensor);
        }
        if (this->ws.bytes_of_vec > 0) {
            TensorDesc biasDesc = tensor1d(this->dt, this->p.num_outputs);
            Tensor modelVectorTensor = Tensor(OCLMem);
            modelVectorTensor.resize(biasDesc);
            auto vectorMem = (OclMemory *)modelVectorTensor.get_memory();
            vectorMem->padding(0, 8, 0, 0);
            this->biasTensors.push_back(modelVectorTensor);
        }
        return SUCCESS;
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        I32 algo[4];
        std::string name = this->name + std::to_string(get_type());
        EE ret = SUCCESS;
        if (algorithmMap->getAlgorithmInfoFromMap(name, algo, 4)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_h[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
        } else {
            ret = fully_connected_infer_forward_algorithm(
                inputTensor, filterTensor, outputTensor, &this->archInfo);
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_h[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            algorithmMap->setAlgorithmInfoToMap(name, algo, 4);
        }
        return ret;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor weightTensor = this->weightTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        Tensor biasTensor = Tensor(OCLMem);
        if (biasTensors.size() > 0) {
            biasTensor = this->biasTensors[0];
        } else {
            if (this->inputTensors.size() > 1) {
                biasTensor = this->inputTensors[1];
            }
        }
        Tensor tmpTensor = Tensor(OCLMem);
        std::vector<Tensor> tmpTensors(2, tmpTensor);
        tmpTensors[0] = this->temp;
        get_tmp_image(0, bytes + 1, &tmpTensors[1]);

        CHECK_STATUS(fully_connected(
            inputTensor, weightTensor, biasTensor, tmpTensors, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        TensorDesc inputDesc = inTensors[0]->get_desc();
        U32 in, ic, ih, iw;
        tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
        if (tensorIs4d(inputDesc)) {
            this->numInput = ic * ih * iw;
        } else {
            this->numInput = iw;
        }
        TensorDesc filterDesc2D =
            tensor2df(this->dt, DF_NORMAL, this->p.num_outputs, this->numInput);
        Tensor filterTensor = Tensor(OCLMem);
        filterTensor.resize(filterDesc2D);
        EE ret = fully_connected_infer_output_size(
            inTensors[0], filterTensor, outTensors[0], &this->archInfo);
        U32 row = tensorNumElements(inputDesc) / this->numInput;
        if (ret == SUCCESS && check_tensors_image(inTensors) && row > 1) {
            ret = set_tensors_image(outTensors, inTensors.size());
        }
        if (inTensors.size() > 1) {
            auto biasMem = (OclMemory *)inTensors[1]->get_memory();
            biasMem->padding(0, 8, 0, 0);
        }
        CHECK_REQUIREMENT(this->p.num_slices == 1);
        return ret;
    }

    U32 infer_tmp_memory_size() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        for (U32 i = 0; i < 4; i++) {
            bytes[i] = 0;
        }
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(
            inputTensor, filterTensor, outputTensor, bytes, &this->archInfo));
        add_tmp_image(0, bytes + 1);
        return bytes[0];
    }

    EE alloc_wtm_memory()
    {
        Tensor filterTensor = this->weightTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        TensorDesc ftmDesc;

        EE ret = fully_connected_transform_filter_bytes(filterTensor, &ftmDesc, &this->archInfo);
        if (ret == SUCCESS) {
            this->wtm = std::shared_ptr<Tensor>(new Tensor(OCLMem));
            this->wtm->resize(ftmDesc);
            U32 row = outputDesc.dims[1];
            if (row > 1) {
                CHECK_STATUS(set_wtm_image(ftmDesc));
            }
            this->wtm->alloc();
        }
        return ret;
    }

    EE transform_filter() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        CHECK_REQUIREMENT(this->p.num_slices == 1);
        EE ret = alloc_wtm_memory();
        if (ret == SUCCESS) {
            ret = fully_connected_transform_filter(
                inputTensor, filterTensor, this->wtm.get(), &this->archInfo);
            this->weightTensors[0] = *(this->wtm.get());
        }
        return ret;
    }

    REGISTER_OCL_OPERATOR_RUN

protected:
    ForwardRunInfoMali runInfo;
    U32 bytes[4];
};

#endif  // _FULLY_CONNECTED_OCL_H
