// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _DECONVOLUTION_OCL_H
#define _DECONVOLUTION_OCL_H

#include "deconvolution.hpp"

class DeconvolutionOCL : public Deconvolution {
public:
    DeconvolutionOCL(DataType dt, ConvolutionParamSpec p, ActivationParamSpec activationDesc)
        : Deconvolution(dt, p, activationDesc)
    {
        if (this->p.group > 1) {
            UNI_ERROR_LOG("GPU currently not support depthwise deconvolution, please replace with deconvolution or resize.\n");
	}
        INIT_GPU_INFO(&this->runInfo)
    }

    ~DeconvolutionOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<DeconvolutionOCL> mem = std::shared_ptr<DeconvolutionOCL>(
            new DeconvolutionOCL(this->dt, this->p, this->activationDesc));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        DataType dtNoQ = noQuantDataType(this->dt);
        DataFormat df = DF_NCHW;
        U32 fh, fw, fc, fn;
        fn = this->numInputs;
        fc = this->p.num_outputs;
        fh = this->p.kernel_h;
        fw = this->p.kernel_w;
        U32 vectorLen = fn;
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        Tensor modelWeightTensor = Tensor(OCLMem);
        Tensor modelVectorTensor = Tensor(OCLMemImg1D);
        TensorDesc filterTensorDesc = tensor4df(dtNoQ, df, fn, fc, fh, fw);
        TensorDesc vectorTensorDesc = tensor1d(dtNoQ, vectorLen);
        modelWeightTensor.resize(filterTensorDesc);
        modelVectorTensor.resize(vectorTensorDesc);
        this->weightTensors.push_back(modelWeightTensor);
        this->biasTensors.push_back(modelVectorTensor);
        return SUCCESS;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        U8 *scalePtr = nullptr;
        Tensor biasTensor = this->biasTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        CHECK_STATUS(deconvolution(inputTensor, filterTensor, p, this->alg, scalePtr, biasTensor,
            this->temp, outputTensor, this->activationDesc, &this->archInfo));
    }

    EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap) override
    {
        OCLContext::getInstance().handle.get()->kernelVec = &this->opKernelVec;
        ConvolutionPolicy policy = CONVOLUTION_TUNNING;
        ((MaliPara_t)(this->archInfo.archPara))->forwardRunInfo->algorithm =
            CONVOLUTION_ALGORITHM_NULL;
        DataType targetType = DT_F16;

        I32 algo[4];
        std::string name = this->name + std::to_string(get_type());
        EE ret = SUCCESS;
        if (algorithmMap->getAlgorithmInfoFromMap(name, algo, 4)) {
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_h[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
            this->alg = (ConvolutionForwardAlgorithm)algo[0];
        } else {
            ret = deconvolution_infer_forward_algorithm(this->inputTensors[0],
                this->weightTensors[0], this->outputTensors[0], p, policy, &(this->alg), targetType,
                this->activationDesc, &this->archInfo);
            algo[0] = this->runInfo.algorithm;
            algo[1] = this->runInfo.best_h[0];
            algo[2] = this->runInfo.best_c[0];
            algo[3] = this->runInfo.best_k[0];
            this->alg = (ConvolutionForwardAlgorithm)algo[0];
            algorithmMap->setAlgorithmInfoToMap(name, algo, 4);
        }
        return ret;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        auto inputTensor = inTensors[0];
        TensorDesc inDim = inputTensor->get_desc();
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS(tensor4dGet(inDim, &idt, &idf, &in, &ic, &ih, &iw));
        this->numInputs = ic;

        TensorDesc filterDim = tensor4df(this->dt, DF_NCHW, this->numInputs, this->p.num_outputs,
            this->p.kernel_h, this->p.kernel_w);
        Tensor filterTensor = Tensor(OCLMem);
        filterTensor.resize(filterDim);

        DataType targetType = this->dt;
        EE ret = deconvolution_infer_output_size(
            inputTensor, filterTensor, p, outTensors[0], targetType, &this->archInfo);
        if (ret == SUCCESS && check_tensors_image(inTensors)) {
            ret = set_tensors_image(outTensors, inTensors.size());
        }
        return ret;
    }

    U32 infer_tmp_memory_size() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor filterTensor = this->weightTensors[0];
        Tensor outputTensor = this->outputTensors[0];

        U32 bytes = 0;
        CHECK_STATUS(deconvolution_infer_forward_tmp_bytes(
            inputTensor, filterTensor, outputTensor, p, this->alg, &bytes, &this->archInfo));
        return bytes;
    }

    EE alloc_wtm_memory()
    {
        TensorDesc ftmDesc;
        EE ret = deconvolution_transform_filter_bytes(
            this->weightTensors[0], this->p, this->alg, &ftmDesc, &this->archInfo);
        if (ret == SUCCESS) {
            this->wtm = std::shared_ptr<Tensor>(new Tensor(OCLMem));
            this->wtm->resize(ftmDesc);
            this->wtm->alloc();
        }
        return ret;
    }

    EE transform_filter() override
    {
        Tensor filterTensor = this->weightTensors[0];
        EE ret = alloc_wtm_memory();
        if (ret == SUCCESS) {
            ret = deconvolution_transform_filter(
                filterTensor, this->p, this->alg, this->temp, this->wtm.get(), &this->archInfo);
            this->weightTensors[0] = *(this->wtm.get());
        }
        return ret;
    }

    REGISTER_OCL_OPERATOR_RUN

protected:
    ForwardRunInfoMali runInfo;
};

#endif  // _DECONVOLUTION_OCL_H
