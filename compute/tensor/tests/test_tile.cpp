// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

#include "tensor_computing.h"
#include "ut_util.h"

int tileTest(int argc, char **argv, DataType dt)
{
    // input dim
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    //input axis and tiles
    TileParamSpec tileParamSpec;
    tileParamSpec.axis = atoi(argv[5]);
    tileParamSpec.dimsSize = 0;
    tileParamSpec.repeatsInfo[0] = atoi(argv[6]);

    //set input
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    DataFormat df = DF_NCHW;
    TensorDesc inDesc = tensor4df(dt, df, in, ic, ih, iw);
    U32 len = tensorNumElements(inDesc);
    U8 *input = ut_input_v(len, dt, UT_INIT_RANDOM);
    Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inDesc);
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, inputTensor.bytes());

    //set output
    Tensor outputTensor;
    CHECK_STATUS(tile_infer_output_size(&inputTensor, tileParamSpec, &outputTensor, &archInfo));
    outputTensor.alloc();
    if (UT_CHECK) {
        CHECK_STATUS(tile(inputTensor, tileParamSpec, outputTensor, &archInfo));

        CHECK_REQUIREMENT(outputTensor.length() == (len * tileParamSpec.repeatsInfo[0]));
    }

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    tileTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    tileTest(argc, argv, DT_F32);
#endif
    return 0;
}
