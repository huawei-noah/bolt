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

int concatTest(int argc, char** argv, DataType dt) {
    CHECK_REQUIREMENT(argc > 2);
    int num = atoi(argv[1]);
    U32 axis = atoi(argv[2]);
    CHECK_REQUIREMENT(axis == 0 || axis == 1);
    CHECK_REQUIREMENT(argc == 1 + 2 + (num+1)*4);

    std::vector<TensorDesc> in_desc(num);
    TensorDesc out_desc;
    std::vector<std::vector<U32>> in_dims(num);
    for(int i=0; i<num; i++){
        std::vector<U32> in_dim(4);
        in_dim[0] = atoi(argv[3 + i * 4]);
        in_dim[1] = atoi(argv[3 + i * 4 + 1]);
        in_dim[2] = atoi(argv[3 + i * 4 + 2]);
        in_dim[3] = atoi(argv[3 + i * 4 + 3]);
        in_dims[i] = in_dim;
        in_desc[i] = tensor4df(dt, DF_NCHWC8, in_dim[0], in_dim[1], in_dim[2], in_dim[3]);
    }
    U32 on = atoi(argv[3 + num * 4]);
    U32 oc = atoi(argv[3 + num * 4 + 1]);
    U32 oh = atoi(argv[3 + num * 4 + 2]);
    U32 ow = atoi(argv[3 + num * 4 + 3]);

    CHECK_STATUS(concat_infer_output_size(in_desc, &out_desc, axis, UT_ARCH));

    U32 in_len = 0;
    for(int i=0; i<num; i++){
        in_len += tensorNumElements(in_desc[i]);
    }
    U32 out_len = tensorNumElements(out_desc);
    CHECK_REQUIREMENT(in_len == out_len && out_len == on * oc * oh * ow);

    std::vector<void*> input(num);
    U8 *tmp = ut_input_v(in_len, dt, UT_INIT_RANDOM);
    U32 count = 0;
    for(int i=0; i<num; i++){
        input[i] = (void *)(tmp + count * bytesOf(dt));
        count += tensorNumElements(in_desc[i]);
    }
    U8 *output = ut_input_v(out_len, dt, UT_INIT_ZERO);

    if(UT_CHECK){
        CHECK_STATUS(concat(in_desc, input, nullptr, out_desc, output, nullptr, axis, UT_ARCH));

        // check
        ut_check_v(output, tmp, in_len, dt, 0, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for(int iter=0; iter<UT_LOOPS; iter++){
        CHECK_STATUS(concat(in_desc, input, nullptr, out_desc, output, nullptr, axis, UT_ARCH));
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
    ut_log(dt, buffer, ops, time);

    free(tmp);
    free(output);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    concatTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    concatTest(argc, argv, DT_F32);
#endif
    return 0;
}
