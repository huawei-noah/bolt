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
#include "tensor_computing.h"
#include "tensor.hpp"
#include "op_type.h"
#include <map>
#include "model_tools.h"
#define HashMap std::map

#ifdef _USE_MALI
#include "gcl.h"
#endif

class Operator {
public:
    virtual bool checkOperator() {
        for (U32 i = 0; i < inputTensors.size(); i++) {
            if (!tensorDescIsValid(inputTensors[i].get_desc()))
                return false;
        }
        for (U32 i = 0; i < outputTensors.size(); i++) {
            if (!tensorDescIsValid(outputTensors[i].get_desc()))
                return false;
        }
        return true;
    };

    virtual void run() = 0;

    /**
     * @param inputTensors
     * @param outputTensors
     */
    virtual void set_input_output_tensors(Vec<Tensor> it, Vec<Tensor> ot)
    {
        this->inputTensors = it;
        this->outputTensors = ot;
    }

    virtual Vec<Tensor> get_input_tensors()
    {
        return this->inputTensors;
    }

    virtual Vec<Tensor> get_output_tensors()
    {
        return this->outputTensors;
    }

    virtual void set_input_tensors(Vec<Tensor> it)
    {
        this->inputTensors = it;
    }

    virtual void set_output_tensors(Vec<Tensor> ot)
    {
        this->outputTensors = ot;
    }

    virtual bool can_input_output_the_same() { return false; }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>, Vec<TensorDesc>*) = 0;

    std::string get_name()
    {
       return this->name;
    }
    /**
     * @param name
     */
    explicit Operator(std::string name)
    {
        this->name = name;
    }

    Operator():name("") { }

    virtual bool is_weight()
    {
        return false;
    }

    virtual U32 infer_tmp_memory_size()
    {
        this->lenOfTemp = 0;
        this->temp = std::shared_ptr<Memory_>();
        return 0;
    }

    virtual void set_tmp_memory(U32 len, std::shared_ptr<Memory_> temp)
    {
        this->lenOfTemp = len;
        this->temp = temp;
    }
#ifdef _USE_MALI
    virtual EE set_mali_handle(std::shared_ptr<GCLHandle> handle){
        this->handle = handle;
        oclExtInfo.maliInfo.handle = handle.get();  
        runInfo.algorithm = 0;
        runInfo.best_w[0] = 1;
        runInfo.best_w[1] = 1;
        runInfo.best_c[0] = 1;
        runInfo.best_c[1] = 1;
        runInfo.best_k[0] = 1;
        runInfo.best_k[1] = 1;
        oclExtInfo.maliInfo.forwardRunInfo = &runInfo;
        return SUCCESS;
    }
    virtual EE infer_gclmem_desc(Vec<GCLMemDesc>*, Vec<GCLMemDesc>*){return NOT_SUPPORTED;}
#endif

    virtual U32 get_len_of_temp()
    {
        return this->lenOfTemp;
    }

    virtual std::shared_ptr<Memory_> get_tmp()
    {
        return this->temp;
    }

    virtual OperatorType get_op_type() = 0;

    virtual void set_op_name(std::string opName) {
        this->name = opName;
    }

    virtual void set_op_schedule(Arch opSchedule) {
        this->schedule = opSchedule;
    }

    virtual Vec<I32> get_tensor_positions()
    {
        return this->tensorPos;
    }

    virtual void set_tensor_positions(Vec<I32> tensorPos)
    {
        this->tensorPos = tensorPos;
    }

    virtual ~Operator(){ }

    virtual int get_next_operator_index()
    {
        return -1;
    }

    virtual void setAlgorithmInfoToMap(HashMap<std::string, std::string> &algorithmMap, std::string name, I32* algorithmArray, U32 ArrayNum) 
    {
        std::string algoInfo = "/";
        for(U32 i = 0; i < ArrayNum; i++) {
            algoInfo += std::to_string(algorithmArray[i]);
            algoInfo += "/";
        }
        algorithmMap[name] = algoInfo;
    }

    virtual void getAlgorithmInfoFromMap(HashMap<std::string, std::string> &algorithmMap, std::string name, I32* algorithmArray, U32 ArrayNum) 
    {
       std::string algoInfo = algorithmMap[name];
       U32 be = algoInfo.find_first_of("/"); 
       U32 end;
       for(U32 i = 0; i < ArrayNum; i++) {
           end = algoInfo.find("/", be + 1);
           algorithmArray[i] = std::stoi(algoInfo.substr(be + 1, end - be - 1));
           be  = end;
       }
    }

    virtual void init_feature_scale(U32 num, QuantSpec* qs)
    {
        UNUSED(num);
        UNUSED(qs);
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
    virtual void set_feature_scale(Vec<Vec<F32>> fs) {
        this->featureScale = fs;
    }

    virtual bool is_dynamic_scale()
    {
        OperatorType ot = this->get_op_type();
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
    std::string get_op_name()
    {
        return this->name;
    }

public:
    Arch schedule;
    DataType dt;

    Vec<Tensor> inputTensors;
    Vec<Tensor> outputTensors;
    Vec<I32> tensorPos;

    U32 lenOfTemp;
    std::shared_ptr<Memory_> temp;

#ifdef _USE_MALI
    std::shared_ptr<GCLHandle> handle;
    ExtInfo oclExtInfo;
    ForwardRunInfoMali runInfo;
#endif

    std::string name;
    Vec<Vec<F32>> featureScale;
};

#endif //_OPERATOR_H
