// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "tensor_computing.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif

inline EE priorbox_infer_output_size_cpu(std::vector<TensorDesc> inputDesc, PriorBoxDesc priorboxDesc, TensorDesc* outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc[0], &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_REQUIREMENT(!priorboxDesc.aspect_ratios.empty());
    U32 num_priorboxs = priorboxDesc.aspect_ratios.size();
    if(priorboxDesc.flip){
        num_priorboxs = num_priorboxs * 2;
    }
    CHECK_REQUIREMENT(!priorboxDesc.min_sizes.empty());
    U32 num_minsize = priorboxDesc.min_sizes.size();
    num_priorboxs = num_priorboxs * num_minsize + num_minsize;
    if(!priorboxDesc.max_sizes.empty()){
        U32 num_maxsize = priorboxDesc.max_sizes.size();
        CHECK_REQUIREMENT(num_minsize == num_maxsize);
        num_priorboxs = num_priorboxs + num_maxsize;
    }
    DEBUG_info(" Number of priorboxs per pixel : " << num_priorboxs);
    //on = 1, oc = 2, ol= 4*num_priorboxs*ih*iw
    *outputDesc =  tensor3d(idt, 1, 2, 4*num_priorboxs*ih*iw);
    return SUCCESS;
}

EE priorbox_infer_output_size(std::vector<TensorDesc> inputDesc, PriorBoxDesc priorboxDesc, TensorDesc* outputDesc, Arch arch, ExtInfo_t extInfo)
{
    UNUSED(arch);
    UNUSED(extInfo);
    CHECK_STATUS(priorbox_infer_output_size_cpu(inputDesc, priorboxDesc, outputDesc));
    return SUCCESS;
}

EE priorbox(std::vector<TensorDesc> inputDesc, PriorBoxDesc priorboxDesc, TensorDesc outputDesc, void* output, Arch arch, ExtInfo_t extInfo)
{
    UNUSED(extInfo);    
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = priorbox_general(inputDesc, priorboxDesc, outputDesc, output);
#endif
#ifdef _USE_GENERAL
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = priorbox_arm(inputDesc, priorboxDesc, outputDesc, output);
#endif
    }
    return ret;
}
