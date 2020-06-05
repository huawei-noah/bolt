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
#include "libkernelbin.h"
#endif

class Model {
public:
    /**
     * @param name
     */
    Model() {}
    Model(Arch schedule, DataType dt, std::string name) {
        this->schedule = schedule;
        this->dt   = dt;
        this->name = name;
        this->memoryAssigned = false;
        this->algorithmFileName = "algorithmInfo_";
        this->algorithmFileName += name;
        this->algorithmFileName += "_";
        this->algorithmFileName += std::to_string(schedule);
        this->algorithmFileName += "_";
        this->algorithmFileName += std::to_string(dt);
#ifdef _USE_MALI
        if(schedule == MALI){
            GCLHandle_t handleInfo;
            CHECK_STATUS(gcl_create_handle(&handleInfo));
            CHECK_STATUS(gcl_regist_binMap(handleInfo));
            this->handle = std::shared_ptr<GCLHandle>(handleInfo, gcl_destroy_handle);
        }
#endif
    }

    virtual EE infer_output_tensors_size(HashMap<std::string, TensorDesc>) = 0;
    virtual void assign_output_tensor() = 0;
    virtual void infer_tmp_memory_size() = 0;
    virtual void assign_tmp_tensor() = 0;

    virtual void ready(HashMap<std::string, TensorDesc> inputDescMap) {
        infer_output_tensors_size(inputDescMap);
        assign_output_tensor();

        infer_tmp_memory_size();
        assign_tmp_tensor();
    }

#ifdef _USE_MALI
    virtual EE infer_gclmem_descs(HashMap<std::string, TensorDesc>) {return NOT_SUPPORTED;}
    virtual void run_mali_prepare(bool reset) {
#ifndef _DEBUG    
        for(auto op : ops) op->run();
        if(reset) return;
        std::vector<U32> kernelIndex;
        U32 len = handle->kernelVec.size();
        for(U32 i = 0; i < len; i++) {
            auto kernelInfo = handle->kernelVec[i];
            U32  gs[3];
            U32  ls[3];
            bool findKernelThreadInfo = false;
            findKernelThreadInfo = getKernelThreadInfoFromMap(kernelThreadMap, kernelInfo.name, gs, ls);
            if(findKernelThreadInfo){
                handle->kernelVec[i].gs[0] = gs[0];
                handle->kernelVec[i].gs[1] = gs[1];
                handle->kernelVec[i].gs[2] = gs[2];
                handle->kernelVec[i].ls[0] = ls[0];
                handle->kernelVec[i].ls[1] = ls[1];
                handle->kernelVec[i].ls[2] = ls[2];
            } else {
                kernelIndex.push_back(i);
            }
        }
        CHECK_STATUS(gcl_run_kernelVec_select_ls(handle.get(), kernelIndex));
        for(U32 i = 0; i < len; i++) {
            auto kernelInfo = handle->kernelVec[i];
            setKernelThreadInfoToMap(kernelThreadMap, kernelInfo.name, kernelInfo.gs, kernelInfo.ls);
        }
#else
        UNUSED(reset);
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
                    if(output->desc.memFormat == DF_NCWHC4) {
                        U32 s1 = output->desc.stride[0];
                        U32 s2 = output->desc.stride[1];
                        U32 s3 = output->desc.stride[2];
                        U32 ho = output->desc.offset[0];
                        U32 wo = output->desc.offset[1];
                        U32 bi = (s2 > 4) ? 4 : s2;
                        U32 bj = (s3 > 1) ? 2 : s3;
                        if(bi == 1 && s1 == 1) bj = (s3 > 8) ? 8 : s3;
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
                /*    
                std::string name = op->get_name();
                for(auto outputTensor: op->get_output_tensors()) {
                    gcl_write_data_to_bin<F16>(handle.get(), outputTensor.get_desc(), outputTensor.get_val(), 0, name.c_str());
                    name = name + "x";
                }
                */
            }

            U32 len = handle->kernelVec.size();
            for(U32 i = 0; i < len; i++) {
                auto kernelInfo = handle->kernelVec[i];
                U32  gs[3];
                U32  ls[3];
                bool findKernelThreadInfo = getKernelThreadInfoFromMap(kernelThreadMap, kernelInfo.name, gs, ls);
                if(findKernelThreadInfo){
                    handle->kernelVec[i].gs[0] = gs[0];
                    handle->kernelVec[i].gs[1] = gs[1];
                    handle->kernelVec[i].gs[2] = gs[2];
                    handle->kernelVec[i].ls[0] = ls[0];
                    handle->kernelVec[i].ls[1] = ls[1];
                    handle->kernelVec[i].ls[2] = ls[2];
                }
            }
            CHECK_STATUS(gcl_run_kernelVec(handle.get()));
#endif
        } else {
#endif
            for (U32 opIndex = 0; opIndex < ops.size();) {
                std::shared_ptr<Operator> op = this->ops[opIndex];
#ifdef _DEBUG
                std::cout << funcStr << " op: " << op->get_name() << "/"<< OperatorTypeName()[op->get_op_type()] << std::endl;
#endif
                if (op->get_op_type() == OT_Repeat || op->get_op_type() == OT_Jump ) {
                    opIndex = op->get_next_operator_index();
                } else {
                    op->run();
                    opIndex++;
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
                        std::cerr << "\n[ERROR] encounter inf at " << i << std::endl;
                        exit(1);
                    }
                    if (UNI_ISNAN(value)) {
                        std::cerr << "\n[ERROR] encounter nan at " << i << std::endl;
                        exit(1);
                    }
                }
                /*
                std::string name = op->get_name();
                for(auto outputTensor: op->get_output_tensors()) {
                    auto desc = outputTensor.get_desc();
                    if(desc.nDims == 3) desc.df = DF_MTK;
                    gcl_write_data_to_bin<F16>(NULL, desc, outputTensor.get_val(), 1, name.c_str());
                    name = name + "x";
                }*/
                std::cout << std::endl;
#endif
            }
#ifdef _USE_MALI
        }
#endif
    }

#ifdef _USE_INT8
    virtual U32 find_next_dynamic_scale_op(Vec<U32> calibratedOpIdx, U32 startIdx)
    {
        CHECK_REQUIREMENT(startIdx < this->ops.size())
        for (U32 i = startIdx; i < this->ops.size(); ) {
            auto op = this->ops[i];
            if (op->is_dynamic_scale()) {
                bool calibrated = false;
                for (auto idx : calibratedOpIdx) {
                    if (i == idx) {
                        calibrated = true;
                        break;
                    }
                }
                if (!calibrated) {
                    return i;
                }
            }
            
            if (op->get_op_type() == OT_Repeat || op->get_op_type() == OT_Jump ) {
                i = op->get_next_operator_index();
            } else {
                i++;
            }
        }

        return 0;  // The first layer should never be quantized
    }

    virtual std::shared_ptr<Operator> get_op_by_index(U32 i)
    {
        return ops[i];
    }

    virtual void run_till_breakpoint(U32 opIdx)
    {
        CHECK_REQUIREMENT(MALI != this->schedule);
        for (U32 i = 0; i < this->ops.size(); ) {
            auto op = this->ops[i];
            if (op->get_op_type() == OT_Repeat || op->get_op_type() == OT_Jump ) {
                if (opIdx == i) {
                    break;
                }
                i = op->get_next_operator_index();
            } else {
                op->run();
                if (opIdx == i) {
                    break;
                }
                i++;
            }
        }
    }
#endif

    virtual bool checkOperator()
    {
        for (auto op : this->ops) {
            if (! op->checkOperator()) return false;
        }
        return true;
    }

    std::string get_name() {return this->name;}

    void loadAlgorithmMapFromText(std::string algorithmMapPath) {
        if (algorithmMapPath == std::string("")) return;    
        FILE *file = fopen(algorithmFileName.c_str(), "r");
        if (!file || feof(file))
            return;
        int num = 0;
        fscanf(file, "%d", &num);
        char operatorName[100];
        char algorithm[100];
        for (int i = 0; i < num; i++) {
            fscanf(file, "%s %s", operatorName, algorithm);
            algorithmMap[operatorName] = algorithm;
        }
#ifdef _USE_MALI
        if(this->schedule == MALI)  {
            fscanf(file, "%d", &num);
            char kernelName[100];
            char kernelThreadInfo[100];
            for (int i = 0; i < num; i++) {
                fscanf(file, "%s %s", kernelName, kernelThreadInfo);
                kernelThreadMap[kernelName] = kernelThreadInfo;
            }
        }
#endif        
        fclose(file);
    }

    void saveAlgorithmMapToText(std::string algorithmMapPath) {
        if (algorithmMapPath == std::string("")) return;
        FILE* fileProb = fopen(algorithmFileName.c_str(), "r");
        if (fileProb) {
            fclose(fileProb);
            return;
        }
        
        FILE *file = fopen(algorithmFileName.c_str(), "w");
        fprintf(file, "%ld\n", (I64)algorithmMap.size());
        for (auto iter: algorithmMap) {
            fprintf(file, "%s %s\n", iter.first.c_str(), iter.second.c_str());
        }
#ifdef _USE_MALI        
        if(this->schedule == MALI)  {
            fprintf(file, "%ld\n", (I64)kernelThreadMap.size());
            for (auto iter: kernelThreadMap) {
                fprintf(file, "%s %s\n", iter.first.c_str(), iter.second.c_str());
            }
        }
#endif        
        fclose(file);
    }
#ifdef _USE_MALI
    void setKernelThreadInfoToMap(HashMap<std::string, std::string> &kernelThreadMap, std::string name, U32 gs[3], U32 ls[3]) {
        std::string kernelThreadInfo = "/";
        for(U32 i = 0; i < 3; i++) {
            kernelThreadInfo += std::to_string(gs[i]);
            kernelThreadInfo += "/";
        }
        for(U32 i = 0; i < 3; i++) {
            kernelThreadInfo += std::to_string(ls[i]);
            kernelThreadInfo += "/";
        }
        kernelThreadMap[name] = kernelThreadInfo;
    }

    bool getKernelThreadInfoFromMap(HashMap<std::string, std::string> &kernelThreadMap, std::string name, U32* gs, U32* ls) {
        bool findKernelInfo = kernelThreadMap.count(name);
        if(!findKernelInfo) return findKernelInfo;
        std::string kernelThreadInfo = kernelThreadMap[name];
        U32 be = kernelThreadInfo.find_first_of("/");
        U32 end;
        for(U32 i = 0; i < 3; i++) {
            end = kernelThreadInfo.find("/", be + 1);
            gs[i] = std::stoi(kernelThreadInfo.substr(be + 1, end - be - 1));
            be = end;
        }
        for(U32 i = 0; i < 3; i++) {
            end = kernelThreadInfo.find("/", be + 1);
            ls[i] = std::stoi(kernelThreadInfo.substr(be + 1, end - be - 1));
            be = end;
        }
        return findKernelInfo;
    }
#endif

protected:
    Vec<std::shared_ptr<Operator>> ops;
    Arch schedule;
    DataType dt;
#ifdef _USE_MALI
    std::shared_ptr<GCLHandle> handle;
    HashMap<std::string, std::string> kernelThreadMap;
#endif
    HashMap<std::string, std::string> algorithmMap;
    bool memoryAssigned;

private:
    std::string name;
    std::string algorithmFileName;
};
#endif
