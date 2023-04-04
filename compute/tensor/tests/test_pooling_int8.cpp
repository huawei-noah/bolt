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

#ifdef _USE_INT8
int int8PoolingTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 18);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 it = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);

    PoolingParamSpec p;
    p.mode = POOLING_MEAN;
    p.round_mode = ROUND_CEIL;
    p.kernel_t = atoi(argv[6]);
    p.kernel_h = atoi(argv[7]);
    p.kernel_w = atoi(argv[8]);
    p.stride_t = atoi(argv[9]);
    p.stride_h = atoi(argv[10]);
    p.stride_w = atoi(argv[11]);
    p.pad_before = atoi(argv[12]);
    p.pad_after = atoi(argv[13]);
    p.pad_top = atoi(argv[14]);
    p.pad_bottom = atoi(argv[15]);
    p.pad_left = atoi(argv[16]);
    p.pad_right = atoi(argv[17]);

    TensorDesc inputDesc = tensor4df(DT_I8, DF_NCHWC8, in, ic, ih, iw);
    TensorDesc inputDescRef = inputDesc;
    inputDescRef.dt = dt;
    Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inputDesc);
    Tensor inputTensorRef = Tensor::alloc_sized<CPUMem>(inputDescRef);
    ut_init_v((U8 *)get_ptr_from_tensor(inputTensorRef, CPU_GENERAL), inputTensorRef.length(), dt,
        UT_INIT_RANDOM);

    Tensor outputTensor;
    CHECK_STATUS(pooling_infer_output_size(&inputTensor, p, &outputTensor, &UT_CPU_ARCHINFO));
    outputTensor.alloc();
    TensorDesc outputDesc = outputTensor.get_desc();
    outputDesc.dt = dt;
    Tensor outputTensorRef = Tensor::alloc_sized<CPUMem>(outputDesc);

    U32 output_len = outputTensor.length();

    F32 inputScale = -1;
    CHECK_STATUS(quantize(inputTensorRef, &inputTensor, &inputScale, &UT_CPU_ARCHINFO));
    inputTensor.set_scale(inputScale);

    std::vector<Tensor> outputTensors = {outputTensor};
    std::vector<Tensor> outputTensorsRef = {outputTensorRef};

    Tensor tmpTensor;
    if (UT_CHECK) {
        CHECK_STATUS(pooling(inputTensor, p, tmpTensor, outputTensors, &UT_CPU_ARCHINFO));
        F32 outputScale = outputTensor.get_scale();
        INT8 *output = (INT8 *)get_ptr_from_tensor(outputTensor, CPU_GENERAL);
        U8 *out_d = ut_input_v(output_len, dt, UT_INIT_ZERO);
        for (U32 i = 0; i < output_len; i++) {
            switch (dt) {
#ifdef _USE_FP32
                case DT_F32:
                    ((F32 *)out_d)[i] = output[i] / outputScale;
                    break;
#endif
#ifdef _USE_FP16
                case DT_F16:
                    ((F16 *)out_d)[i] = output[i] / outputScale;
                    break;
#endif
                default:
                    break;
            }
        }

        CHECK_STATUS(pooling(inputTensorRef, p, tmpTensor, outputTensorsRef, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(out_d, get_ptr_from_tensor(outputTensorRef, CPU_GENERAL), output_len, dt, 0.05);
        free(out_d);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(pooling(inputTensor, p, tmpTensor, outputTensors, &UT_CPU_ARCHINFO));
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
    ut_log(DT_I8, buffer, ops, time);

    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_INT8
#ifdef _USE_FP16
    int8PoolingTest(argc, argv, DT_F16);
#endif
#endif
    return 0;
}
