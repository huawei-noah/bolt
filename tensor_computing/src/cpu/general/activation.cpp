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
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
        case DT_F16: {
            U32 len = tensorNumElements(inputDesc);
            F16* dataPtr = (F16 *)data;
            for (U32 i = 0; i < len; i++) {
                CHECK_STATUS_WITH_RETURN(activation<F16>(activationMode, dataPtr[i], &dataPtr[i]));
            }
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
