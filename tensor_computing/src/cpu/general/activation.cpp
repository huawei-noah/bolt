// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/general/tensor_computing_general.h"
#include "cpu/general/common_general.h"

EE activation_general(TensorDesc inputDesc, void* data, ActivationMode activationMode)
{
    if (nullptr == data)
        CHECK_STATUS(NULL_POINTER);
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    U32 len = tensorNumElements(inputDesc);
    for (U32 i = 0; i < len; i++) {
        switch (idt) {
#ifdef _USE_FP16
            case DT_F16: {
                F16* dataPtr = (F16 *)data;
                CHECK_STATUS(activation<F16>(activationMode, dataPtr[i], &dataPtr[i]));
                break;
            }
#endif
#ifdef _USE_FP32
            case DT_F32: {
                F32* dataPtr = (F32 *)data;
                CHECK_STATUS(activation<F32>(activationMode, dataPtr[i], &dataPtr[i]));
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    }
    return ret;
}
