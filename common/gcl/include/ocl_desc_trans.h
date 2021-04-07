// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _OCL_DESC_TRANS
#define _OCL_DESC_TRANS

#include "tensor.hpp"
#include "memory_ocl.hpp"
#include "gcl_common.h"
inline void ocl_set_desc(Tensor *tensor, GCLMemDesc desc)
{
    OclMemory *mem = (OclMemory *)tensor->get_memory();
    mem->padding(desc);
}

inline void ocl_set_descs(std::vector<Tensor *> tensors, std::vector<GCLMemDesc> descs)
{
    for (U32 i = 0; i < tensors.size(); i++) {
        ocl_set_desc(tensors[i], descs[i]);
    }
}

inline GCLMemDesc ocl_get_desc(Tensor tensor)
{
    OclMemory *mem = (OclMemory *)tensor.get_memory();
    return mem->get_desc();
}

inline std::vector<GCLMemDesc> ocl_get_descs_ptr(std::vector<Tensor *> tensors)
{
    std::vector<GCLMemDesc> descs;
    for (U32 i = 0; i < tensors.size(); i++) {
        descs.push_back(ocl_get_desc(*(tensors[i])));
    }
    return descs;
}

inline std::vector<GCLMemDesc> ocl_get_descs(std::vector<Tensor> tensors)
{
    std::vector<GCLMemDesc> descs;
    for (U32 i = 0; i < tensors.size(); i++) {
        descs.push_back(ocl_get_desc(tensors[i]));
    }
    return descs;
}
#endif
