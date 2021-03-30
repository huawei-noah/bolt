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
#include "sys.h"
#include "tensor.hpp"
#include "algorithm_map.h"
#ifdef _USE_MALI
#include "gcl.h"
#include "gcl_engine.h"
#endif
#include "parameter_spec.h"

class Operator {
public:
    Operator()
    {
        this->dt = DT_F32;
        this->name = "";
        this->lenOfTemp = 0;
        this->archInfo.archPara = nullptr;
    }

    Operator(std::string name)
    {
        this->dt = DT_F32;
        this->name = name;
        this->lenOfTemp = 0;
        this->archInfo.archPara = nullptr;
    }

    virtual ~Operator()
    {
        if (this->archInfo.archPara != nullptr) {
            free(this->archInfo.archPara);
            this->archInfo.archPara = nullptr;
        }
    }

    virtual std::shared_ptr<Operator> clone() = 0;

    virtual EE infer_output_tensors_size(std::vector<Tensor *>, std::vector<Tensor *>) = 0;

    virtual U32 infer_tmp_memory_size()
    {
        this->lenOfTemp = 0;
        return 0;
    }

    virtual void set_tmp_memory(Tensor temp)
    {
        this->lenOfTemp = temp.bytes();
        this->temp = temp;
    }

    virtual void run() = 0;

    virtual void set_input_output_tensors(std::vector<Tensor> it, std::vector<Tensor> ot)
    {
        set_input_tensors(it);
        set_output_tensors(ot);
    }

    virtual void set_input_tensors(std::vector<Tensor> &it)
    {
        this->inputTensors = it;
    }

    virtual std::vector<Tensor> get_input_tensors()
    {
        return this->inputTensors;
    }

    virtual void set_output_tensors(std::vector<Tensor> &ot)
    {
        this->outputTensors = ot;
    }

    virtual std::vector<Tensor> get_output_tensors()
    {
        return this->outputTensors;
    }

    virtual bool can_input_output_the_same()
    {
        return false;
    }

    virtual bool is_weight()
    {
        return false;
    }

    virtual U32 get_len_of_temp()
    {
        return this->lenOfTemp;
    }

    virtual Tensor get_tmp()
    {
        return this->temp;
    }

    virtual void set_name(std::string opName)
    {
        this->name = opName;
    }

    std::string get_name()
    {
        return this->name;
    }

    virtual void set_schedule(Arch opSchedule)
    {
        this->archInfo.arch = opSchedule;
    }

    virtual void set_tensor_positions(std::vector<I32> tensorPos)
    {
        this->tensorPos = tensorPos;
    }

    virtual std::vector<I32> &get_tensor_positions()
    {
        return this->tensorPos;
    }

    virtual int get_next_operator_index()
    {
        return -1;
    }

    virtual void init_feature_scale(U32 num, QuantSpec *qs)
    {
#ifdef _USE_INT8
        if (1 == num && 0 == qs[0].scale[0]) {  // OP is labelled as no-quantization
            if (DT_F16_8Q == this->dt) {
                this->dt = DT_F16;
            }
            return;
        }
        featureScale.resize(num);
        for (U32 i = 0; i < num; i++) {
            featureScale[i].resize(qs[i].num_scale);
            memcpy(featureScale[i].data(), qs[i].scale, qs[i].num_scale * bytesOf(DT_F32));
        }
#endif
    }

#ifdef _USE_INT8
    virtual void set_feature_scale(std::vector<std::vector<F32>> fs)
    {
        this->featureScale = fs;
    }

    virtual bool is_dynamic_scale()
    {
        OperatorType ot = this->get_type();
        if (OT_Conv != ot) {
            return false;
        }

        U32 numScale = featureScale.size();
        U32 numQuant = (DT_F16_8Q == this->dt) ? inputTensors.size() : 0;

        if (0 != numScale && 0 == featureScale[0][0]) {  // OP is labelled as no-quantization
            return false;
        }

        if (0 != numScale && -2 == (featureScale.back())[0]) {  // OP is labelled as fp-output
            numScale = 0;
            numQuant += 1;
        }

        for (auto tensor : outputTensors) {
            if (DT_I8 == tensor.get_desc().dt) {
                numQuant++;
            }
        }
        if (0 == numQuant) {
            return false;
        }

        if (0 == numScale) {
            return true;
        }

        CHECK_REQUIREMENT(numQuant == numScale);
        return false;
    }
#endif

    virtual bool checkOperator()
    {
        for (U32 i = 0; i < inputTensors.size(); i++) {
            if (!tensorDescIsValid(inputTensors[i].get_desc())) {
                return false;
            }
        }
        for (U32 i = 0; i < outputTensors.size(); i++) {
            if (!tensorDescIsValid(outputTensors[i].get_desc())) {
                return false;
            }
        }
        return true;
    };

    virtual OperatorType get_type() = 0;

    virtual EE infer_forward_algorithm(std::shared_ptr<AlgorithmMap> algorithmMap)
    {
        UNUSED(algorithmMap);
        return SUCCESS;
    }

    virtual void set_algorithm_map(std::shared_ptr<AlgorithmMap> algorithmMap)
    {
        this->algorithmMap = algorithmMap;
    }

protected:
    ArchInfo archInfo;
    DataType dt;

    std::vector<Tensor> inputTensors;
    std::vector<Tensor> outputTensors;
    std::vector<I32> tensorPos;

    U32 lenOfTemp;
    Tensor temp;

    std::string name;
    std::vector<std::vector<F32>> featureScale;
    std::shared_ptr<AlgorithmMap> algorithmMap;
};

#endif  // _OPERATOR_H
