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

int whereTest(int argc, char **argv)
{
    /*
    float replaceF = 1.1;
    ArchInfo archInfo;
    archInfo.arch = CPU_GENERAL;
    TensorDesc inDesc = tensor4df(DT_F32, DF_NCHW, 1, 2, 3, 3);
    U32 len = tensorNumElements(inDesc);
    U8 *input = ut_input_v(len, DT_F32, UT_INIT_RANDOM);
    Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inDesc);
    memcpy(get_ptr_from_tensor(inputTensor, archInfo.arch), input, inputTensor.bytes());
    TensorDesc conditionDesc = tensor4df(DT_F32, DF_NCHW, 1, 1, 3, 3);
    U32 len_condition = tensorNumElements(conditionDesc);
    U8 *condition = ut_input_v(len_condition, DT_F32, UT_INIT_ZERO);
    F32* conditionF32 = (F32*)condition;
    conditionF32[0] = 1;
    conditionF32[2] = 1;
    Tensor conditionTensor = Tensor::alloc_sized<CPUMem>(conditionDesc);
    memcpy(get_ptr_from_tensor(conditionTensor, archInfo.arch), condition, conditionTensor.bytes());
    Tensor outputTensor;
    CHECK_STATUS(where_infer_output_size(&inputTensor, &outputTensor, &archInfo));
    outputTensor.alloc();
    CHECK_STATUS(where(inputTensor, conditionTensor, outputTensor, replaceF, &archInfo));
    */
    return 0;
}

int whereTest2(int argc, char **argv)
{
    /*
    float replaceF = 1.1;
    ArchInfo archInfo;
    archInfo.arch = CPU_GENERAL;
    TensorDesc inDesc = tensor4df(DT_F32, DF_NCHW, 1, 2, 2, 2);
    U32 len = tensorNumElements(inDesc);
    U8 *input = ut_input_v(len, DT_F32, UT_INIT_RANDOM);
    Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inDesc);
    memcpy(get_ptr_from_tensor(inputTensor, archInfo.arch), input, inputTensor.bytes());
    TensorDesc conditionDesc = tensor4df(DT_F32, DF_NCHW, 1, 2, 2, 2);
    U32 len_condition = tensorNumElements(conditionDesc);
    U8 *condition = ut_input_v(len_condition, DT_F32, UT_INIT_ZERO);
    F32* conditionF32 = (F32*)condition;
    conditionF32[0] = 1;
    conditionF32[3] = 1;
    Tensor conditionTensor = Tensor::alloc_sized<CPUMem>(conditionDesc);
    memcpy(get_ptr_from_tensor(conditionTensor, archInfo.arch), condition, conditionTensor.bytes());
    Tensor outputTensor;
    CHECK_STATUS(where_infer_output_size(&inputTensor, &outputTensor, &archInfo));
    outputTensor.alloc();
    CHECK_STATUS(where(inputTensor, conditionTensor, outputTensor, replaceF, &archInfo));
    */
    return 0;
}

int main(int argc, char **argv)
{
    whereTest(argc, argv);
    whereTest2(argc, argv);
    return 0;
}
