// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MODEL_COMMON
#define _H_MODEL_COMMON

#include <string>
#include "model_spec.h"
#include "memory_cpu.h"

inline void *mt_malloc(U32 size)
{
    return UNI_OPERATOR_NEW(size);
}

template <typename T>
inline void mt_free(T *&p)
{
    UNI_OPERATOR_DELETE(p);
    p = nullptr;
}

// only WeightSpec's weight and vec varialbles free by using this.
// because this will use mmap memory.
template <typename T>
inline void mt_free(T *&p, ModelSpec *spec)
{
    if (spec == nullptr || spec->file == nullptr || (uintptr_t(p) < uintptr_t(spec->file->content)) ||
        (uintptr_t(p) >= uintptr_t(spec->file->content + spec->file->length))) {
        UNI_OPERATOR_DELETE(p);
    }
    p = nullptr;
}

inline EE str_copy(char *dst, const char *src, I32 src_len, I32 dst_len = NAME_LEN)
{
    UNI_MEMSET(dst, 0, dst_len);
    I32 length = NAME_LEN - 1;
    if (length > src_len) {
        length = src_len;
    }
    UNI_MEMCPY(dst, src, length * sizeof(I8));
    return SUCCESS;
}

OperatorSpec mt_create_operator(
    const char *name, OperatorType type, U32 num_inputs, U32 num_outputs);

EE mt_insert_operator(ModelSpec *ms, int index, OperatorSpec newOperator);

EE mt_insert_weight(ModelSpec *ms, WeightSpec *newWeight, int num);

WeightSpec mt_create_weight(
    const char *name, DataType dataType, U32 bytesOfWeight, U32 bytesOfVec, U32 numQuantScale);

bool isDeprecatedOp(OperatorType opType);

bool isDeprecatedOpWeight(const ModelSpec *spec, int index);

void modify_ms_inputs_and_outputs(
    ModelSpec *ms, std::string modifiedInputs, std::string modifiedOutputs);
#endif
