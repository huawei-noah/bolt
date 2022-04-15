// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef UTILS_H
#define UTILS_H

#include <training/base/common/MemoryManager.h>
#include <training/system/Name.h>
#include <training/base/common/NetworkParameters.h>

namespace raul::utils
{

void printTensorNames(std::ostream& stream, const raul::Name& name, const Names& a, const MemoryManager& memoryManager);
void printTensorNames(std::ostream& stream, const raul::Name& name, const Names& a, const MemoryManagerFP16& memoryManager);
[[maybe_unused]] void printList(std::ostream& stream, const raul::Name& name, const Names& a);

void traceTensor(const raul::Name& tensorName, const raul::NetworkParameters& networkParameters, const raul::Name& prefix = "");

#define traceWithPrefix(tensor, prefix) utils::traceTensor(tensor, mNetworkParams, (prefix) + mName / __func__ + ".")
#define trace(tensor) traceWithPrefix(tensor, "")

} // namespace raul::utils

#endif // UTILS_H
