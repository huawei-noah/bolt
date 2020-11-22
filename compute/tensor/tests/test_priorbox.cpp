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

int priorboxTest(int argc, char **argv, DataType dt)
{
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
    F32 ar2 = 0;
    F32 min_size1 = 0;
    F32 max_size1 = 0;
    if (argc == 19 || argc == 21) {
        ar2 = (F32)atof(argv[18]);
        if (argc == 21) {
            min_size1 = (F32)atof(argv[19]);
            max_size1 = (F32)atof(argv[20]);
        }
    }
    if (argc == 20) {
        min_size1 = (F32)atof(argv[18]);
        max_size1 = (F32)atof(argv[19]);
    }
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    CHECK_REQUIREMENT(in0 == 1 && in1 == 1 && on == 1 && oc == 2);

    PriorBoxParamSpec priorbox_desc;
    int min_sizes_len = 1;
    int max_sizes_len = 1;
    int aspect_ratios_len = 1;
    priorbox_desc.min_sizes[0] = min_size;
    priorbox_desc.max_sizes[0] = max_size;
    priorbox_desc.aspect_ratios[0] = ar1;
    priorbox_desc.min_sizes[1] = min_size1;
    priorbox_desc.max_sizes[1] = max_size1;
    priorbox_desc.aspect_ratios[1] = ar2;
    if (argc == 19 || argc == 21) {
        aspect_ratios_len++;
        if (argc == 21) {
            min_sizes_len++;
            max_sizes_len++;
        }
    }
    if (argc == 20) {
        min_sizes_len++;
        max_sizes_len++;
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

    std::vector<Tensor> inputTensors(2);
    std::vector<Tensor *> inputTensorPtrs(2);
    Tensor inputTensor_fm, inputTensor_data;
    TensorDesc inputDesc_fm = tensor4df(dt, DF_NCHWC8, in0, ic0, ih0, iw0);
    TensorDesc inputDesc_data = tensor4df(dt, DF_NCHWC8, in1, ic1, ih1, iw1);
    inputTensor_fm.resize(inputDesc_fm);
    inputTensor_data.resize(inputDesc_data);
    U32 input_len_fm = tensorNumElements(inputDesc_fm);
    U32 input_len_data = tensorNumElements(inputDesc_data);
    inputTensors[0] = inputTensor_fm;
    inputTensors[1] = inputTensor_data;
    inputTensorPtrs[0] = &inputTensors[0];
    inputTensorPtrs[1] = &inputTensors[1];
    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(
        priorbox_infer_output_size(inputTensorPtrs, priorbox_desc, &outputTensor, &archInfo));
    outputTensor.alloc();
    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();
    U32 output_len = outputTensor.length();
    CHECK_REQUIREMENT(input_len_fm == in0 * ic0 * ih0 * iw0 &&
        input_len_data == in1 * ic1 * ih1 * iw1 && output_len == on * oc * olens);

    if (UT_CHECK) {
        CHECK_STATUS(priorbox(inputTensors, priorbox_desc, outputTensor, &archInfo));

        CHECK_STATUS(priorbox(inputTensors, priorbox_desc, outputTensorRef, &archInfo_org));
        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), output_len, dt, 0.05, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(priorbox(inputTensors, priorbox_desc, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    U32 num_priorboxs = aspect_ratios_len;
    if (priorbox_desc.flip) {
        num_priorboxs = num_priorboxs * 2;
    }
    U32 num_minsize = min_sizes_len;
    num_priorboxs = (num_priorboxs + 1) * num_minsize;
    if (max_sizes_len != 0) {
        U32 num_maxsize = max_sizes_len;
        num_priorboxs = num_priorboxs + num_maxsize;
    }
    U32 ochannel = 2;
    U32 numperbox = 4;
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) * (%u %u %u) = (%u %u %u)", in0, ic0, ih0, iw0, ochannel,
        numperbox, num_priorboxs, on, oc, olens);
    sprintf(buffer, "%20s, %80s", "Priorbox", params);
    double ops = 1.0 * output_len;
    ut_log(dt, buffer, ops, time);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    priorboxTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    priorboxTest(argc, argv, DT_F32);
#endif
    return 0;
}
