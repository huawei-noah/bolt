// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ut_util.h"
#include "image_processing.hpp"
#include "tensor_auxiliary.h"

int main()
{
    TensorDesc rgbDesc = tensor4df(DT_U8, DF_RGB, 1, 3, 1280, 960);
    U8 *rgb = ut_input_v(tensorNumElements(rgbDesc), DT_U8, UT_INIT_POS);
    Tensor rgbTensor = Tensor::alloc_sized<CPUMem>(rgbDesc);
    UNI_MEMCPY(get_ptr_from_tensor(rgbTensor, ARM_A76), rgb, tensorNumBytes(rgbDesc));

    TensorDesc imageDesc = tensor4df(DT_F32, DF_NCHW, 1, 3, 224, 224);
    load_resize_image(rgbTensor, imageDesc, RGB, 0.017);
    free(rgb);
    return 0;
}
