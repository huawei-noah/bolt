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
#include "ut_util.h"
#include <typeinfo>

int priorboxTest(int argc, char **argv, DataType dt){
    CHECK_REQUIREMENT(argc == 18 || argc == 19 || argc == 20 || argc == 21);
    // in0 feature map
    U32 in0 = atoi(argv[1]);
    U32 ic0 = atoi(argv[2]);
    U32 ih0 = atoi(argv[3]);
    U32 iw0 = atoi(argv[4]);
    // in1 data
    U32 in1 = atoi(argv[5]);
    U32 ic1 = atoi(argv[6]);
    U32 ih1 = atoi(argv[7]);
    U32 iw1 = atoi(argv[8]);
    // param priorbox
    F32 min_size = (F32)atof(argv[9]);
    F32 max_size = (F32)atof(argv[10]);
    U32 flip = atoi(argv[11]);
    U32 clip = atoi(argv[12]);
    F32 step = (F32)atof(argv[13]);
    // output
    U32 on = atoi(argv[14]);
    U32 oc = atoi(argv[15]);
    U32 olens = atoi(argv[16]);
    // multi param priorbox
    F32 ar1 = (F32)atof(argv[17]);
    F32 ar2;
    F32 min_size1;
    F32 max_size1;
    if(argc == 19 || argc == 21){
        ar2 = (F32)atof(argv[18]);
        if(argc == 21){
            min_size1 = (F32)atof(argv[19]);
            max_size1 = (F32)atof(argv[20]);
        }
    }
    if(argc == 20){
        min_size1 = (F32)atof(argv[18]);
        max_size1 = (F32)atof(argv[19]);
    }
    
    CHECK_REQUIREMENT(in0 == 1 && in1 == 1 && on == 1 && oc == 2);

    PriorBoxDesc priorbox_desc;
    priorbox_desc.min_sizes.push_back(min_size);
    priorbox_desc.max_sizes.push_back(max_size);
    priorbox_desc.aspect_ratios.push_back(ar1);
    if(argc == 19 || argc == 21){
        priorbox_desc.aspect_ratios.push_back(ar2);
        if(argc == 21){
            priorbox_desc.min_sizes.push_back(min_size1);
            priorbox_desc.max_sizes.push_back(max_size1);
        }
    }
    if(argc == 20){
        priorbox_desc.min_sizes.push_back(min_size1);
        priorbox_desc.max_sizes.push_back(max_size1);
    }
    priorbox_desc.flip = flip;
    priorbox_desc.clip = clip;
    priorbox_desc.image_h = ih1;
    priorbox_desc.image_w = iw1;
    priorbox_desc.step_h = step;
    priorbox_desc.step_w = step;
    priorbox_desc.variances[0] = 0.10000000149;
    priorbox_desc.variances[1] = 0.10000000149;
    priorbox_desc.variances[2] = 0.20000000298;
    priorbox_desc.variances[3] = 0.20000000298;
    priorbox_desc.offset = 0.5;

    std::vector<TensorDesc> input_descs;
    TensorDesc output_desc;
    TensorDesc input_desc_fm = tensor4df(dt, DF_NCHWC8, in0, ic0, ih0, iw0);
    TensorDesc input_desc_data = tensor4df(dt, DF_NCHWC8, in1, ic1, ih1, iw1);
    input_descs.push_back(input_desc_fm);
    input_descs.push_back(input_desc_data);
    CHECK_STATUS(priorbox_infer_output_size(input_descs, priorbox_desc, &output_desc, UT_ARCH));
    U32 input_len_fm = tensorNumElements(input_descs[0]);
    U32 input_len_data = tensorNumElements(input_descs[1]);
    U32 output_len = tensorNumElements(output_desc);
    CHECK_REQUIREMENT(input_len_fm == in0*ic0*ih0*iw0 && input_len_data == in1*ic1*ih1*iw1 && output_len == on*oc*olens);

    U8* output = ut_input_v(output_len, dt, UT_INIT_ZERO);
    U8* output_ref = ut_input_v(output_len, dt, UT_INIT_ZERO);
    
    if (UT_CHECK) {
        CHECK_STATUS(priorbox(input_descs,
                            priorbox_desc,
                            output_desc, output,
                            UT_ARCH));

        CHECK_STATUS(priorbox(input_descs,
                             priorbox_desc,
                             output_desc, output_ref,
                             CPU_GENERAL));
        // check
        ut_check_v(output, output_ref, output_len, dt, 0.05, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for(int iter = 0; iter < UT_LOOPS; iter++){
        CHECK_STATUS(priorbox(input_descs, priorbox_desc, output_desc, output, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    U32 num_priorboxs = priorbox_desc.aspect_ratios.size();
    if(priorbox_desc.flip){
        num_priorboxs = num_priorboxs * 2;
    }
    U32 num_minsize = priorbox_desc.min_sizes.size();
    num_priorboxs = (num_priorboxs + 1) * num_minsize;
    if(!priorbox_desc.max_sizes.empty()){
        U32 num_maxsize = priorbox_desc.max_sizes.size();
        num_priorboxs = num_priorboxs + num_maxsize;
    }
    U32 ochannel = 2;
    U32 numperbox = 4;
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) * (%u %u %u) = (%u %u %u)", in0, ic0, ih0, iw0, ochannel, numperbox, num_priorboxs, on, oc, olens);
    sprintf(buffer, "%20s, %80s", "Priorbox", params);
    double ops = 1.0 * output_len;
    ut_log(dt, buffer, ops, time);

    free(output);
    free(output_ref);
    return 0;
}


int main(int argc, char** argv){
#ifdef _USE_FP16
    priorboxTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    priorboxTest(argc, argv, DT_F32);
#endif
    return 0;
}