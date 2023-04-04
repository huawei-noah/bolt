// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _OPERATOR_H
#define _OPERATOR_H

#include <string>
#ifdef _USE_GPU
#include "gcl.h"
#include "gcl_engine.h"
#include "image_container.hpp"
#endif
#include "tensor.hpp"
#include "tensor_computing.h"
#include "algorithm_map.h"

class Operator {
public:
    Operator()
    {
        this->name = "";
        this->dt = DT_F32;
        this->archInfo.archPara = nullptr;
#ifdef _USE_GPU
        this->tempImages = nullptr;
#endif
    }

    virtual ~Operator()
    {}

    virtual std::shared_ptr<Operator> clone() = 0;

    virtual EE infer_output_tensors_size(std::vector<Tensor *>, std::vector<Tensor *>) = 0;

    virtual EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap)
    {
        return SUCCESS;
    }

    void set_algorithm_map(std::shared_ptr<AlgorithmMap> algorithmMap)
    {
        this->algorithmMap = algorithmMap;
    }

    virtual U32 infer_tmp_memory_size()
    {
        return 0;
    }

    void set_tmp_memory(Tensor temp)
    {
        this->temp = temp;
    }

    virtual void run() = 0;

    void set_name(const std::string &opName)
    {
        this->name = opName;
    }

    std::string get_name()
    {
        return this->name;
    }

    virtual OperatorType get_type() = 0;

    void set_input_tensors(std::vector<Tensor> &it)
    {
        this->inputTensors = it;
    }

    std::vector<Tensor> get_input_tensors()
    {
        return this->inputTensors;
    }

    void set_output_tensors(std::vector<Tensor> &ot)
    {
        this->outputTensors = ot;
    }

    std::vector<Tensor> get_output_tensors()
    {
        return this->outputTensors;
    }

    void set_tensor_positions(std::vector<I32> &pos)
    {
        this->tensorPos = pos;
    }

    std::vector<I32> &get_tensor_positions()
    {
        return this->tensorPos;
    }

    virtual bool is_weight()
    {
        return false;
    }

    void set_schedule(Arch opSchedule)
    {
        this->archInfo.arch = opSchedule;
    }

    virtual int get_next_operator_index()
    {
        return -1;
    }

    DataType get_activation_quant_data_type(F32 scale = -1)
    {
        DataType ret = DT_I8;
        if (IS_X86(this->archInfo.arch)) {
            if (scale == -3) {
                ret = DT_I8;
            } else {
                ret = DT_U8_Q;
            }
        }
        return ret;
    }

#ifdef _USE_INT8
    void init_feature_scale(U32 num, QuantSpec *qs)
    {
        if (1 == num && 0 == qs[0].scale[0]) {  // OP is labelled as no-quantization
            this->dt = noQuantDataType(this->dt);
            return;
        }
        featureScale.resize(num);
        for (U32 i = 0; i < num; i++) {
            featureScale[i].resize(qs[i].num_scale);
            UNI_MEMCPY(featureScale[i].data(), qs[i].scale, qs[i].num_scale * bytesOf(DT_F32));
        }
    }

    void set_feature_scale(const std::vector<std::vector<F32>> &fs)
    {
        this->featureScale = fs;
    }

    bool is_dynamic_scale()
    {
        OperatorType ot = this->get_type();
        if (OT_Conv != ot && OT_FC != ot && OT_MatMul != ot) {
            return false;
        }

        U32 numScale = featureScale.size();
        U32 numQuant = isQuantMixDataType(this->dt) ? inputTensors.size() : 0;

        if (0 != numScale && 0 == featureScale[0][0]) {  // OP is labelled as no-quantization
            return false;
        }

        // if (0 != numScale && -2 == (featureScale.back())[0]) {  // OP is labelled as fp-output
        //     numScale = 0;
        //     numQuant += 1;
        // }

        // for (auto tensor : outputTensors) {
        //     if (DT_I8 == tensor.get_desc().dt) {
        //         numQuant++;
        //     }
        // }

        // if (0 == numQuant) {
        //     return false;
        // }

        // if (0 == numScale) {
        //     return true;
        // }

        // CHECK_REQUIREMENT(numQuant == numScale);
        return true;
    }
#endif

#ifdef _USE_GPU
    void set_tmp_images(ImageContainer *tmpImageContainer)
    {
        this->tempImages = tmpImageContainer;
    }

    void add_tmp_image(U32 slot, U32 *size)
    {
        if (IS_QUALCOMM_GPU(this->archInfo.arch)) {
            this->tempImages->add(slot, size[0], size[1], size[2]);
        }
    }

    bool get_tmp_image(U32 slot, U32 *size, Tensor *tensor)
    {
        bool findMatchImage = false;
        if (IS_QUALCOMM_GPU(this->archInfo.arch)) {
            if (size[0] == 0 && size[1] == 0 && size[2] == 0) {
                return false;
            } else if (size[0] == 0 || size[1] == 0 || size[2] == 0) {
                UNI_ERROR_LOG("gpu tmp buffer(on image buffer) parameter is wrong.\n");
            }
            *tensor = this->tempImages->get(slot, size[0], size[1], size[2]);
            findMatchImage = true;
        }
        return findMatchImage;
    }

    bool check_tensors_image(std::vector<Tensor *> tensors, I32 tensorId = -1)
    {
        if (IS_QUALCOMM_GPU(this->archInfo.arch)) {
            bool isImage = true;
            U32 be = (tensorId >= 0) ? tensorId : 0;
            U32 end = (tensorId >= 0) ? tensorId + 1 : tensors.size();
            for (U32 i = be; i < end; i++) {
                if (tensors[i]->get_mem_type() == OCLMem) {
                    isImage = false;
                    break;
                }
            }
            return isImage;
        }
        return false;
    }

    EE set_tensors_image(std::vector<Tensor *> tensors, U32 tensorPosOff, I32 tensorId = -1)
    {
        if (IS_QUALCOMM_GPU(this->archInfo.arch)) {
            U32 be = (tensorId >= 0) ? tensorId : 0;
            U32 end = (tensorId >= 0) ? tensorId + 1 : tensors.size();
            if (tensorPosOff + end - be > tensorPos.size()) {
                return NOT_MATCH;
            }
            for (U32 i = be; i < end; i++) {
                if (this->tensorPos[tensorPosOff + i] != -2) {
                    auto mem = std::shared_ptr<OclMemoryImg>(new OclMemoryImg());
                    TensorDesc desc = tensors[i]->get_desc();
                    mem->resize(desc);
                    U32 str[3] = {0};
                    mem->stride(str);
                    if (gcl_check_meet_device_image3d_limits(
                            OCLContext::getInstance().handle.get(), str[0], str[1], str[2])) {
                        tensors[i]->set_shared_memory(mem);
                    }
                }
            }
        }
        return SUCCESS;
    }

    virtual void update_kernel()
    {}
#endif

    int is_shape(const std::vector<Tensor *> &tensors)
    {
        int count = 0;
        for (U32 i = 0; i < tensors.size(); i++) {
            count += tensorIsShape(tensors[i]->get_desc());
        }
        return count;
    }

    TensorDesc tensor_shape(Tensor tensor)
    {
        TensorDesc desc = tensor.get_desc();
        U32 *ptr = (U32 *)((CpuMemory *)(tensor.get_memory()))->get_ptr();
        for (U32 i = 0; i < tensor.length() && desc.nDims + i < DIM_LEN; i++) {
            desc.dims[desc.nDims + i] = ptr[i];
        }
        return desc;
    }

protected:
    std::string name;
    DataType dt;

    std::vector<Tensor> inputTensors;
    std::vector<Tensor> outputTensors;
    std::vector<I32> tensorPos;

    Tensor temp;

    ArchInfo archInfo;
    std::shared_ptr<AlgorithmMap> algorithmMap;

    std::vector<std::vector<F32>> featureScale;
#ifdef _USE_GPU
    ImageContainer *tempImages;
#endif
};

#endif  // _OPERATOR_H
