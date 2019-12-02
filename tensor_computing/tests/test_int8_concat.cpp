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
#include "utils.h"


int main(int argc, char** argv){
    CHECK_REQUIREMENT(argc > 2);
    int num = atoi(argv[1]);
    U32 axis = atoi(argv[2]);
    CHECK_REQUIREMENT(axis == 0 || axis == 1);
    CHECK_REQUIREMENT(argc == 1 + 2 + (num+1)*4);

    std::vector<TensorDesc> in_desc(num);
    std::vector<TensorDesc> in_desc_ref(num);
    TensorDesc out_desc;
    std::vector<std::vector<U32>> in_dims(num);
    for (int i = 0; i < num; i++){
        std::vector<U32> in_dim(4);
        in_dim[0] = atoi(argv[3 + i * 4]);
        in_dim[1] = atoi(argv[3 + i * 4 + 1]);
        in_dim[2] = atoi(argv[3 + i * 4 + 2]);
        in_dim[3] = atoi(argv[3 + i * 4 + 3]);
        in_dims[i] = in_dim;
        in_desc[i] = tensor4df(DT_I8, DF_NCHWC8, in_dim[0], in_dim[1], in_dim[2], in_dim[3]);
        in_desc_ref[i] = in_desc[i];
        in_desc_ref[i].dt = DT_F16;
    }
    U32 on = atoi(argv[3 + num * 4]);
    U32 oc = atoi(argv[3 + num * 4 + 1]);
    U32 oh = atoi(argv[3 + num * 4 + 2]);
    U32 ow = atoi(argv[3 + num * 4 + 3]);

    CHECK_STATUS(concat_infer_output_size(in_desc, &out_desc, axis));

    U32 in_len = 0;
    for (int i = 0; i < num; i++){
        in_len += tensorNumElements(in_desc[i]);
    }
    U32 out_len = tensorNumElements(out_desc);
    CHECK_REQUIREMENT(in_len == out_len && out_len == on * oc * oh * ow);

    std::vector<void*> input_ref(num);
    std::vector<void*> input(num);

    F16 *tmp = ut_input_v<F16>(in_len, UT_INIT_RANDOM);
    INT8 *quant = ut_input_v<INT8>(in_len, UT_INIT_RANDOM);

    U32 count = 0;
    std::vector<F16> scale_i(num);

    for (int i = 0; i < num; i++){
        input_ref[i] = (void *)(tmp + count);
        input[i] = (void *)(quant + count);
        quantize_tensor(in_desc_ref[i], input_ref[i], &(in_desc[i]), input[i], &(scale_i[i]));
        count += tensorNumElements(in_desc[i]);
    }

    INT8 *output = ut_input_v<INT8>(out_len, UT_INIT_ZERO);
    F16 *out_d = ut_input_v<F16>(out_len, UT_INIT_ZERO);
    F16 scale_o;

    if (UT_CHECK) {
        CHECK_STATUS(concat(in_desc, input, scale_i, out_desc, output, &scale_o, axis, UT_ARCH));

        for (U32 i = 0; i < out_len; i++) {
            out_d[i] = output[i] / scale_o;
        }

        // check
        ut_check_v<F16>(out_d, tmp, in_len, F16(0.05), __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for(int iter = 0; iter < UT_LOOPS; iter++){
        CHECK_STATUS(concat(in_desc, input, scale_i, out_desc, output, &scale_o, axis, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "%d (*)/%u=(%u %u %u %u)",
                    num, axis, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Concat", params);
    double ops = 1.0 * out_len;
    ut_log<INT8>(buffer, ops, time);

    free(tmp);
    free(output);
    free(out_d);
    return 0;
}
