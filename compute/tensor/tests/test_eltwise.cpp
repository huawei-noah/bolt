// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>

#include "tensor_computing.h"
#include "ut_util.h"

int eltwiseTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 6);
    U32 num = atoi(argv[1]);
    U32 in = atoi(argv[2]);
    U32 ic = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);
    DataFormat idf = DF_NCHW;
    if (ic % 8 == 0) {
        idf = DF_NCHWC8;
    }

    U32 len = in * ic * ih * iw;
    EltwiseMode eltwiseMode = ELTWISE_MAX;
    EltwiseParamSpec eltwiseDesc;
    eltwiseDesc.mode = eltwiseMode;
    eltwiseDesc.activation_type = ACTIVATION_NULL;

    std::vector<void *> input(num);
    std::vector<Tensor> inTensors(num);
    std::vector<Tensor *> inTensorPtr(num);
    TensorDesc inDesc = tensor4df(dt, idf, in, ic, ih, iw);
    Tensor outTensor;
    for (U32 i = 0; i < num; i++) {
        input[i] = (void *)ut_input_v(len, dt, UT_INIT_RANDOM);
        inTensors[i].resize(inDesc);
        inTensors[i].alloc();
        UNI_MEMCPY(get_ptr_from_tensor(inTensors[i], CPU_GENERAL), input[i], tensorNumBytes(inDesc));
        inTensorPtr[i] = &inTensors[i];
    }

    CHECK_STATUS(eltwise_infer_output_size(inTensorPtr, eltwiseDesc, &outTensor, &UT_CPU_ARCHINFO));
    CHECK_REQUIREMENT(len == outTensor.length());
    outTensor.alloc();
    Tensor outTensorRef;
    outTensorRef.resize(outTensor.get_desc());
    outTensorRef.alloc();

    U32 tmpBytes;
    CHECK_STATUS(eltwise_infer_forward_tmp_bytes(inTensors, outTensor, &tmpBytes, &UT_CPU_ARCHINFO));
    Tensor tmpTensor;
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();

    if (UT_CHECK) {
        CHECK_STATUS(eltwise(inTensors, eltwiseDesc, tmpTensor, outTensor, &UT_CPU_ARCHINFO));

        CHECK_STATUS(eltwise(inTensors, eltwiseDesc, tmpTensor, outTensorRef, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(get_ptr_from_tensor(outTensor, CPU_GENERAL),
            get_ptr_from_tensor(outTensorRef, CPU_GENERAL), len, dt, 1);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(eltwise(inTensors, eltwiseDesc, tmpTensor, outTensor, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "%u (%u %u %u %u)=(%u %u %u %u)", num, in, ic, ih, iw, in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "Eltwise", params);
    double ops = 1.0 * num * in * ic * ih * iw;
    ut_log(dt, buffer, ops, time);

    for (U32 i = 0; i < num; i++) {
        free(input[i]);
    }

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    eltwiseTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    eltwiseTest(argc, argv, DT_F32);
#endif
    return 0;
}
