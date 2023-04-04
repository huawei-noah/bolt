// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_AUXILIARY
#define _H_TENSOR_AUXILIARY

#include <vector>
#include <set>
#include <map>

#include "sys.h"
#include "tensor.hpp"
#include "tensor_transpose.h"

// deprecated API, this will be remove
inline void *get_ptr_from_tensor(Tensor tensor, Arch arch)
{
    void *ptr = nullptr;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        ptr = ((OclMemory *)(tensor.get_memory()))->get_ptr();
#endif
    } else {
        ptr = ((CpuMemory *)(tensor.get_memory()))->get_ptr();
    }
    return ptr;
}

inline std::vector<TensorDesc> get_desc_from_tensors(std::vector<Tensor> tensors)
{
    int size = tensors.size();
    std::vector<TensorDesc> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = tensors[i].get_desc();
    }
    return result;
}

inline std::vector<TensorDesc> get_desc_from_tensor_ptrs(std::vector<Tensor *> tensors)
{
    int size = tensors.size();
    std::vector<TensorDesc> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = tensors[i]->get_desc();
    }
    return result;
}

inline std::vector<F32> get_scale_from_tensors(std::vector<Tensor> tensors)
{
    int size = tensors.size();
    std::vector<F32> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = tensors[i].get_scale();
    }
    return result;
}

template <typename T>
inline std::vector<T> get_data_from_tensors(std::vector<Tensor> tensors, Arch arch)
{
    int size = tensors.size();
    std::vector<T> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = (T)get_ptr_from_tensor(tensors[i], arch);
    }
    return result;
}

template <typename T>
inline std::vector<T> get_data_from_tensor_ptrs(std::vector<Tensor *> tensors, Arch arch)
{
    int size = tensors.size();
    std::vector<T> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = (T)get_ptr_from_tensor(*tensors[i], arch);
    }
    return result;
}

inline void update_desc_from_tensor(Tensor *tensor)
{
    TensorDesc desc = tensor->get_desc();
    if (tensorIsShape(desc) && tensor->get_mem_type() == CPUMem) {
        I32 *p = (I32 *)get_ptr_from_tensor(*tensor, CPU_GENERAL);
        for (U32 i = 0; i < tensorNumElements(desc); i++) {
            desc.dims[desc.nDims + i] = p[i];
        }
    }
    tensor->resize(desc);
}
#endif
