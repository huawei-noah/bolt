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
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;

    std::vector<Tensor> inTensors(num);
    std::vector<Tensor> inTensorsRef(num);
    std::vector<Tensor *> inTensorPtr(num);
    Tensor outTensor;

    for (int i = 0; i < num; i++) {
        std::vector<U32> in_dim(4);
        in_dim[0] = atoi(argv[3 + i * 4]);
        in_dim[1] = atoi(argv[3 + i * 4 + 1]);
        in_dim[2] = atoi(argv[3 + i * 4 + 2]);
        in_dim[3] = atoi(argv[3 + i * 4 + 3]);
        TensorDesc inDesc;
        if (in_dim[1] % 8 == 0) {
            inDesc = tensor4df(DT_I8, DF_NCHWC8, in_dim[0], in_dim[1], in_dim[2], in_dim[3]);
        } else {
            inDesc = tensor4df(DT_I8, DF_NCHW, in_dim[0], in_dim[1], in_dim[2], in_dim[3]);
        }
        inTensors[i].resize(inDesc);
        inDesc.dt = dt;
        inTensorsRef[i].resize(inDesc);
        inTensorPtr[i] = &inTensors[i];
    }
    U32 on = atoi(argv[3 + num * 4]);
    U32 oc = atoi(argv[3 + num * 4 + 1]);
    U32 oh = atoi(argv[3 + num * 4 + 2]);
    U32 ow = atoi(argv[3 + num * 4 + 3]);

    CHECK_STATUS(concat_infer_output_size(inTensorPtr, p, &outTensor, &archInfo));

    U32 in_len = 0;
    for (int i = 0; i < num; i++) {
        in_len += inTensors[i].length();
    }
    U32 out_len = outTensor.length();
    CHECK_REQUIREMENT(in_len == out_len && out_len == on * oc * oh * ow);

    U8 *tmp = ut_input_v(in_len, dt, UT_INIT_RANDOM);
    //INT8 *quant = (INT8 *)ut_input_v(in_len, DT_I8, UT_INIT_ZERO);

    U32 count = 0;
    for (int i = 0; i < num; i++) {
        //input_ref[i] = (void *)(tmp + count * bytesOf(dt));
        inTensorsRef[i].alloc();
        U32 floatBytes = inTensorsRef[i].bytes();
        memcpy(get_ptr_from_tensor(inTensorsRef[i], UT_ARCH), tmp + count, floatBytes);

        inTensors[i].alloc();
        TensorDesc dummy;
        F16 scale = -1;
        quantize_tensor(inTensorsRef[i].get_desc(), tmp + count, &dummy,
            get_ptr_from_tensor(inTensors[i], UT_ARCH), &scale);
        inTensors[i].set_scale(scale);
        count += floatBytes;
    }

    outTensor.alloc();
    U8 *out_d = ut_input_v(out_len, dt, UT_INIT_ZERO);

    Tensor tmpTensor;

    if (UT_CHECK) {
        CHECK_STATUS(concat(inTensors, p, tmpTensor, outTensor, &archInfo));
        F32 scale_o = outTensor.get_scale();
        INT8 *output = (INT8 *)get_ptr_from_tensor(outTensor, UT_ARCH);

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
        ut_check_v(out_d, tmp, in_len, dt, 0.05, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(concat(inTensors, p, tmpTensor, outTensor, &archInfo));
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

    free(tmp);
    free(out_d);
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_INT8
    int8ConcatTest(argc, argv, DT_F16);
#endif
    return 0;
}
