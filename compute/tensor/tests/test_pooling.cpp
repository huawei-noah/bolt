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

int poolingTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 18);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 it = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);

    PoolingParamSpec p;
    p.mode = POOLING_MAX;
    p.rm = CEIL;
    p.kernel_t = atoi(argv[6]);
    p.kernel_h = atoi(argv[7]);
    p.kernel_w = atoi(argv[8]);
    p.stride_t = atoi(argv[9]);
    p.stride_h = atoi(argv[10]);
    p.stride_w = atoi(argv[11]);
    p.padding_before = atoi(argv[12]);
    p.padding_after = atoi(argv[13]);
    p.padding_top = atoi(argv[14]);
    p.padding_bottom = atoi(argv[15]);
    p.padding_left = atoi(argv[16]);
    p.padding_right = atoi(argv[17]);

    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    TensorDesc inputDesc;
    if (it == 1) {
        inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
    } else {
        inputDesc = tensor5df(dt, DF_NCHWC8, in, ic, it, ih, iw);
    }
    Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inputDesc);
    U32 input_len = inputTensor.length();
    U8 *input = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, inputTensor.bytes());

    // set output
    Tensor outputTensor;
    CHECK_STATUS(pooling_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    outputTensor.alloc();
    TensorDesc outputDesc = outputTensor.get_desc();
    Tensor outputTensorRef = Tensor::alloc_sized<CPUMem>(outputDesc);
    U32 output_len = outputTensor.length();
    Tensor tmpTensor;
    if (UT_CHECK) {
        CHECK_STATUS(pooling(inputTensor, p, tmpTensor, outputTensor, &archInfo));

        CHECK_STATUS(pooling(inputTensor, p, tmpTensor, outputTensorRef, &archInfo_org));
        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), output_len, dt, 0.05, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(pooling(inputTensor, p, tmpTensor, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    DataType odt;
    DataFormat odf;
    U32 on = 0, oc = 0, ot = 0, oh = 0, ow = 0;
    if (tensorIs4d(outputDesc)) {
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else if (tensorIs5d(outputDesc)) {
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    }
    sprintf(params, "(%u %u %u %u %u)/(%u %u %u)=(%u %u %u %u %u)", in, ic, it, ih, iw, p.kernel_t,
        p.kernel_h, p.kernel_w, on, oc, ot, oh, ow);
    sprintf(buffer, "%20s, %80s", "Pooling", params);
    double ops = 1.0 * output_len * p.kernel_t * p.kernel_h * p.kernel_w;
    ut_log(dt, buffer, ops, time);

    free(input);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    poolingTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    poolingTest(argc, argv, DT_F32);
#endif
    return 0;
}
