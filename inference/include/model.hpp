// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.


// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _MODEL_H
#define _MODEL_H

#include "operator.hpp"
#include "tensor_desc.h"
#ifdef _USE_MALI
#include "gcl.h"
#endif

class Model {
public:
    /**
     * @param name
     */
    Model(Arch schedule, DataType dt, std::string name) {
        this->schedule = schedule;
        this->dt   = dt;
        this->name = name;
#ifdef _USE_MALI
        if(schedule == MALI){
            GCLHandle_t handleInfo;
            CHECK_STATUS(gcl_create_handle(&handleInfo));
            std::shared_ptr<GCLHandle> handleSptr(handleInfo, gcl_destroy_handle);
            this->handle = handleSptr;
        }
#endif
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>) = 0;
    virtual void assign_output_tensor() = 0;
    virtual void infer_tmp_memory_size() = 0;
    virtual void assign_tmp_tensor() = 0;

    virtual void ready(Vec<TensorDesc> inDims) {
        infer_output_tensors_size(inDims);
        assign_output_tensor();

        infer_tmp_memory_size();
        assign_tmp_tensor();
    }

#ifdef _USE_MALI
    virtual void run_mali_prepare(){
#ifndef _DEBUG    
        for(auto op : ops){
            op->run();
        }
#endif        
    }
#endif 
    virtual void run() {
        
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] run()";
#endif

#ifdef  _USE_MALI
        if(this->schedule == MALI)  {
#ifndef _DEBUG    
            CHECK_STATUS(gcl_run_kernelVec(handle.get()));
#else            
            for (U32 opIndex = 0; opIndex < ops.size(); opIndex++) {
                    std::shared_ptr<Operator> op = this->ops[opIndex];
                    std::cout << funcStr << " op: " << op->get_name() << "/"<< OperatorTypeName()[op->get_op_type()] << std::endl;
                    op->run();
                    Tensor outputTensor = op->get_output_tensors()[0];
                    GCLMem_t output = outputTensor.get_val();
                    U32 s1 = output->desc.stride[0];
                    U32 s2 = output->desc.stride[1];
                    U32 s3 = output->desc.stride[2];
                    U32 ho = output->desc.offset[0];
                    U32 wo = output->desc.offset[1];
                    U32 bi = (s2 > 4) ? 4 : s2;
                    U32 bj = (s3 > 1) ? 2 : s3;
                    U32 size = 4 * s1 * s2 * bj * bytesOf(outputTensor.get_desc().dt);
                    U8* hostPtr = new U8[(size_t)size];
                    gcl_trans_memory(handle.get(), (void*)output, (void*)hostPtr, &size, DEVICE_BUF_TO_HOST, CL_TRUE);
                    F16* val = (F16*) hostPtr;
                    for(U32 i = 0; i < bi; i++){
                            for(U32 j = 0; j < bj; j++){
                                    for(U32 k = 0; k < 4; k++){
                                            std::cout << val[k + ho * 4 + j * s1 * 4 * s2 + (i + wo) * s1 * 4] << " ";
                                    }
                            }
                            std::cout << std::endl;
                    }
                    delete[] hostPtr;
            }
#endif
        } else {
#endif
            for (U32 opIndex = 0; opIndex < ops.size();) {
                std::shared_ptr<Operator> op = this->ops[opIndex];
#ifdef _DEBUG
                std::cout << funcStr << " op: " << op->get_name() << "/"<< OperatorTypeName()[op->get_op_type()] << std::endl;
#endif
                if (op->get_op_type() != OT_Repeat) {
                    op->run();
                    opIndex++;
                } else {
                    opIndex = op->get_next_operator_index();
                }
#ifdef _DEBUG
            // debug for nan
                Tensor outputTensor = op->get_output_tensors()[0];
                U32 elementNum = tensorNumElements(outputTensor.get_desc());
                for (U32 i = 0; i < elementNum; i++) {
                    F32 value = outputTensor.getElement(i);
                    if (i < 32) {
                        if (i % 8 == 0) {
                            if (i != 0)
                                std::cout << std::endl;
                            std::cout << "    ";
                        }
                        std::cout << value << " ";
                    }

                    if (UNI_ISINF(value)) {
                        std::cerr << "\n[ERROR] encounter inf" << std::endl;
                        exit(1);
                    }
                    if (UNI_ISNAN(value)) {
                        std::cerr << "\n[ERROR] encounter nan" << std::endl;
                        exit(1);
                    }
                }
                std::cout << std::endl;
#endif
            }
#ifdef _USE_MALI
        }
#endif
    }

    virtual bool checkOperator() {
        for(auto op : this->ops) {
            if (! op->checkOperator()) return false;
        }
        return true;
    }

    std::string get_name() {return this->name;}

    void loadAlgorithmMapFromText(std::string algorithmMapPath) {
        if (algorithmMapPath == std::string(""))
            return;
        FILE *file = fopen(algorithmMapPath.c_str(), "r");
        if (!file || feof(file))
            return;
        int num = 0;
        fscanf(file, "%d", &num);
        char operatorName[100];
        int algorithm;
        for (int i = 0; i < num; i++) {
            fscanf(file, "%s %d", operatorName, &algorithm);
            algorithmMap[operatorName] = algorithm;
        }
        fclose(file);
    }

    void saveAlgorithmMapToText(std::string algorithmMapPath) {
        if (algorithmMapPath == std::string("") || algorithmMap.size() == 0)
            return;
        FILE* fileProb = fopen(algorithmMapPath.c_str(), "r");
        if (fileProb) {
            fclose(fileProb);
            return;
        }

        FILE *file = fopen(algorithmMapPath.c_str(), "w");
        fprintf(file, "%lu\n", algorithmMap.size());
        for (auto iter: algorithmMap) {
            fprintf(file, "%s %d\n", iter.first.c_str(), iter.second);
        }
        fclose(file);
    }

protected:
    Vec<std::shared_ptr<Operator >> ops;
    Arch schedule;
    DataType dt;
#ifdef _USE_MALI
    std::shared_ptr<GCLHandle> handle;
#endif
    HashMap<std::string, int> algorithmMap;

private:
    std::string name;
};
#endif
