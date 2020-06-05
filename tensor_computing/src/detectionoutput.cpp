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

inline EE detectionoutput_infer_output_size_cpu(std::vector<TensorDesc> inputDesc, DetectionOutputDesc detectionoutputDesc, TensorDesc* outputDesc)
{
    if (inputDesc.size() != 3) {
        CHECK_STATUS(NOT_MATCH);
    }    
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt0, idt2;
    DataFormat idf0, idf2;
    U32 ih0, iw0;
    U32 in2, ic2, ilens2;
    //loc
    CHECK_STATUS(tensor2dfGet(inputDesc[0], &idt0, &idf0, &ih0, &iw0));
    //priorbox
    CHECK_STATUS(tensor3dGet(inputDesc[2], &idt2, &idf2, &in2, &ic2, &ilens2));
    CHECK_REQUIREMENT(iw0 == ilens2);
    //output size
    U32 oh, ow;
    //oh = the first box for saving the number of available boxes(1) + the maximum number of dectected boxes(keep_top_k)
    U32 num_detected_max = detectionoutputDesc.keep_top_k;
    oh = 1 + num_detected_max;
    //Each width is a 6 dimension vector, which stores [label, confidence, xmin, ymin, xmax, ymax] -> 6
    //The first box is [ number of available boxes, 0, 0, 0, 0, 0 ]
    ow = 6;
    *outputDesc = tensor2df(idt0, idf2, oh, ow);
    return SUCCESS;
}

EE detectionoutput_infer_output_size(std::vector<TensorDesc> inputDesc, DetectionOutputDesc detectionoutputDesc, TensorDesc* outputDesc, Arch arch, ExtInfo_t extInfo)
{
    UNUSED(arch);
    UNUSED(extInfo);
    UNUSED(detectionoutputDesc);
    CHECK_STATUS(detectionoutput_infer_output_size_cpu(inputDesc, detectionoutputDesc, outputDesc));
    return SUCCESS;
}

EE detectionoutput(std::vector<TensorDesc> inputDesc, std::vector<void*> input, DetectionOutputDesc detectionoutputDesc, TensorDesc outputDesc, void* output, Arch arch, ExtInfo_t extInfo)
{
    UNUSED(extInfo);
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = detectionoutput_general(inputDesc, input, detectionoutputDesc, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = detectionoutput_arm(inputDesc, input, detectionoutputDesc, outputDesc, output);
#endif
    }
    return ret;
}
