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

#ifdef _USE_INT8
int int8ConcatTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc > 2);
    ConcatParamSpec p;
    int num = atoi(argv[1]);
    p.axis = atoi(argv[2]);
    CHECK_REQUIREMENT(p.axis == 0 || p.axis == 1);
    CHECK_REQUIREMENT(argc == 1 + 2 + (num + 1) * 4);

    std::vector<Tensor> inTensors(num);
    std::vector<Tensor> inTensorsRef(num);
    std::vector<Tensor *> inTensorPtr(num);
    Tensor outTensor;

    U32 in_len = 0;
    for (int i = 0; i < num; i++) {
        U32 n = atoi(argv[3 + i * 4]);
        U32 c = atoi(argv[3 + i * 4 + 1]);
        U32 h = atoi(argv[3 + i * 4 + 2]);
        U32 w = atoi(argv[3 + i * 4 + 3]);
        TensorDesc inDesc;
        if (c % 8 == 0) {
            inDesc = tensor4df(DT_I8, DF_NCHWC8, n, c, h, w);
        } else {
            inDesc = tensor4df(DT_I8, DF_NCHW, n, c, h, w);
        }
        inTensors[i] = Tensor::alloc_sized<CPUMem>(inDesc);
        inDesc.dt = dt;
        inTensorsRef[i] = Tensor::alloc_sized<CPUMem>(inDesc);
        ut_init_v((U8 *)get_ptr_from_tensor(inTensorsRef[i], CPU_GENERAL), inTensorsRef[i].length(),
            dt, UT_INIT_RANDOM);

        inTensorPtr[i] = &inTensors[i];

        F32 scale = -1;
        CHECK_STATUS(quantize(inTensorsRef[i], &inTensors[i], &scale, &UT_CPU_ARCHINFO));
        inTensors[i].set_scale(scale);

        in_len += inTensors[i].length();
    }
    U32 on = atoi(argv[3 + num * 4]);
    U32 oc = atoi(argv[3 + num * 4 + 1]);
    U32 oh = atoi(argv[3 + num * 4 + 2]);
    U32 ow = atoi(argv[3 + num * 4 + 3]);

    CHECK_STATUS(concat_infer_output_size(inTensorPtr, p, &outTensor, &UT_CPU_ARCHINFO));
    U32 out_len = outTensor.length();
    CHECK_REQUIREMENT(in_len == out_len && out_len == on * oc * oh * ow);
    outTensor.alloc();

    Tensor tmpTensor;
    if (UT_CHECK) {
        CHECK_STATUS(concat(inTensors, p, tmpTensor, outTensor, &UT_CPU_ARCHINFO));
        F32 scale_o = outTensor.get_scale();
        INT8 *output = (INT8 *)get_ptr_from_tensor(outTensor, CPU_GENERAL);
        U8 *tmp = (U8 *)ut_input_v(in_len, dt, UT_INIT_ZERO);
        U8 *out_d = (U8 *)ut_input_v(in_len, dt, UT_INIT_ZERO);
        for (int i = 0, index = 0; i < num; i++) {
            UNI_MEMCPY(tmp + index, get_ptr_from_tensor(inTensorsRef[i], CPU_GENERAL),
                inTensorsRef[i].bytes());
            index += inTensorsRef[i].bytes();
        }

        for (U32 i = 0; i < out_len; i++) {
            switch (dt) {
#ifdef _USE_FP16
                case DT_F16:
                    ((F16 *)out_d)[i] = output[i] / scale_o;
                    break;
#endif
#ifdef _USE_FP32
                case DT_F32:
                    ((F32 *)out_d)[i] = output[i] / scale_o;
                    break;
#endif
                default:
                    break;
            }
        }

        // check
        ut_check_v(out_d, tmp, in_len, dt, 0.05);
        free(tmp);
        free(out_d);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(concat(inTensors, p, tmpTensor, outTensor, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "%d (*)/%u=(%u %u %u %u)", num, p.axis, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Concat", params);
    double ops = 1.0 * out_len;
    ut_log(DT_I8, buffer, ops, time);

    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_INT8
#ifdef _USE_FP16
    int8ConcatTest(argc, argv, DT_F16);
#else
    int8ConcatTest(argc, argv, DT_F32);
#endif
#endif
    return 0;
}
