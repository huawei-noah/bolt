// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <filesystem>

#include <training/common/Common.h>
#include <training/common/MemoryManager.h>
#include <training/common/MemoryManagerGPU.h>
#include <training/common/Tensor.h>

namespace raul
{
class DataLoader
{
  public:
    template<typename T>
    static int readArrayFromTextFile(const std::filesystem::path& path, T& array, size_t width, size_t height, size_t depth = 1, size_t batch = 1);
    static int readArrayFromTextFile(const std::filesystem::path& path, MemoryManagerGPU& m, const Name& tensor, size_t width, size_t height, size_t depth = 1, size_t batch = 1);

    template<typename T>
    static void loadData(const std::filesystem::path& path, T& output, size_t width, size_t height, size_t depth = 1, size_t batch = 1);

    template<typename T>
    static void loadData(const std::filesystem::path& path, T& output);

    static void loadData(const std::filesystem::path& path, MemoryManagerGPU& m, const Name& tensor, size_t width, size_t height, size_t depth = 1, size_t batch = 1);
    static void loadData(const std::filesystem::path& path, MemoryManagerGPU& m, const Name& tensor);

    Tensor::dt_range loadData(const std::filesystem::path& path, size_t width, size_t height, size_t depth = 1, size_t batch = 1);
    Tensor::dt_range loadFilters(const std::string& pathPrefix, size_t fileIndexOffset, const std::string& pathPostfix, size_t width, size_t height, size_t depth, size_t filtersAmount);

    const Tensor& buildOneHotVector(const TensorU8& labels, size_t labelsCount);

    Tensor& createTensor(size_t size);

    const MemoryManager& getMemoryManager() const { return mMemoryManager; }

  private:
    MemoryManager mMemoryManager;
};
} // raul namespace

#endif