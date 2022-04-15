// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MemoryManagerGPU.h"
#include <training/opencl/OpenCLKernelManager.h>

#define PADDING_BYTES(a, b) (((b) - ((a) % (b))) % (b))

namespace raul
{
MemoryManagerGPU::MemoryManagerGPU()
    : mKernelManager(nullptr)
    , mAlignment(1u)
{
    mTensorNameGenerator.setPrefix("TensorGPU_");
}

void MemoryManagerGPU::setGpuAttribs(OpenCLKernelManager* manager, size_t alignment)
{
    mKernelManager = manager;
    mAlignment = alignment;

    if (mAlignment < 1u)
    {
        THROW_NONAME("MemoryManagerGPU", "wrong alignment");
    }
}

size_t MemoryManagerGPU::size() const
{
    return mTensors.size();
}

bool MemoryManagerGPU::tensorExists(const raul::Name& name) const
{
    return checkTensorExists(name);
}

bool MemoryManagerGPU::checkTensorExists(const raul::Name& name) const
{
    return mTensors.find(name) != mTensors.end();
}

void MemoryManagerGPU::checkName(const Name& name, const char* caller, bool shouldExist) const
{
    using namespace std::string_literals;
    if (name.empty())
    {
        THROW_NONAME("MemoryManagerGPU", std::string("empty name (requested by ") + caller + ")");
    }
    if (tensorExists(name) != shouldExist)
    {
        THROW_NONAME("MemoryManagerGPU", "tensor '" + name + (shouldExist ? "' doesn`t exist" : "' already exists"));
    }
}

TensorGPU* MemoryManagerGPU::createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width)
{
    checkName(name, __func__, false);

    if (!mKernelManager)
    {
        THROW_NONAME("MemoryManagerGPU", "no kernel manager");
    }

    mTensors[name] = std::make_shared<TensorGPU>(mKernelManager, name, batchSize, depth, height, width, AllocationMode::STANDARD);

    return mTensors[name].get();
}

TensorGPU* MemoryManagerGPU::createTensor(const Name& name, raul::shape inShape)
{
    checkName(name, __func__, false);

    if (!mKernelManager)
    {
        THROW_NONAME("MemoryManagerGPU", "no kernel manager");
    }

    mTensors[name] = std::make_shared<TensorGPU>(mKernelManager, name, inShape[0], inShape[1], inShape[2], inShape[3]);

    return mTensors[name].get();
}

TensorGPU* MemoryManagerGPU::createTensorAligned(const Name& name, raul::shape inShape)
{
    checkName(name, __func__, false);

    if (!mKernelManager)
    {
        THROW_NONAME("MemoryManagerGPU", "no kernel manager");
    }

    size_t sizeInBytes = inShape.total_size() * sizeof(dtype);
    sizeInBytes += PADDING_BYTES(sizeInBytes, mAlignment);

    mTensors[name] = std::make_shared<TensorGPU>(mKernelManager, name, inShape[0], inShape[1], inShape[2], inShape[3], sizeInBytes / sizeof(dtype));

    return mTensors[name].get();
}

TensorGPU* MemoryManagerGPU::createTensor(size_t batchSize, size_t depth, size_t height, size_t width)
{
    return createTensor(mTensorNameGenerator.generate(), batchSize, depth, height, width);
}

TensorGPU* MemoryManagerGPU::createTensor(shape inShape)
{
    return createTensor(mTensorNameGenerator.generate(), inShape);
}

TensorGPU* MemoryManagerGPU::createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, const dtype filler)
{
    checkName(name, __func__, false);

    if (!mKernelManager)
    {
        THROW_NONAME("MemoryManagerGPU", "no kernel manager");
    }

    mTensors[name] = std::make_shared<TensorGPU>(mKernelManager, name, batchSize, depth, height, width, filler);

    return mTensors[name].get();
}

TensorGPU* MemoryManagerGPU::createTensor(const Name& name, shape inShape, const dtype filler)
{
    checkName(name, __func__, false);

    if (!mKernelManager)
    {
        THROW_NONAME("MemoryManagerGPU", "no kernel manager");
    }

    mTensors[name] = std::make_shared<TensorGPU>(mKernelManager, name, inShape[0], inShape[1], inShape[2], inShape[3], filler);

    return mTensors[name].get();
}

TensorGPU* MemoryManagerGPU::createTensor(size_t batchSize, size_t depth, size_t height, size_t width, const dtype filler)
{
    return createTensor(mTensorNameGenerator.generate(), batchSize, depth, height, width, filler);
}

TensorGPU* MemoryManagerGPU::createShape(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, AllocationMode allocMode)
{
    checkName(name, __func__, false);

    if (!mKernelManager)
    {
        THROW_NONAME("MemoryManagerGPU", "no kernel manager");
    }

    mTensors[name] = std::make_shared<TensorGPU>(mKernelManager, name, batchSize, depth, height, width, allocMode, false);

    return mTensors[name].get();
}

TensorGPU* MemoryManagerGPU::createShape(const Name& name, shape inShape, AllocationMode allocMode)
{
    return createShape(name, inShape[0], inShape[1], inShape[2], inShape[3], allocMode);
}

TensorGPU* MemoryManagerGPU::createShape(shape inShape)
{
    return createShape(mTensorNameGenerator.generate(), inShape, AllocationMode::STANDARD);
}

TensorGPU& MemoryManagerGPU::getTensor(const raul::Name& name)
{
    checkName(name, __func__);

    std::string tensorName = name;

    return *mTensors.find(tensorName)->second;
}

const TensorGPU& MemoryManagerGPU::getTensor(const raul::Name& name) const
{
    checkName(name, __func__);

    std::string tensorName = name;

    return *mTensors.find(tensorName)->second;
}

const TensorGPU& MemoryManagerGPU::operator()(const raul::Name& name) const
{
    return getTensor(name);
}

TensorGPU& MemoryManagerGPU::operator()(const raul::Name& name)
{
    return getTensor(name);
}

TensorGPUHelper MemoryManagerGPU::operator[](const raul::Name& name)
{
    return TensorGPUHelper(getTensor(name), mKernelManager);
}

void MemoryManagerGPU::deleteTensor(const raul::Name& name)
{
    checkName(name, __func__);

    std::string tensorName = name;

    mTensors.erase(tensorName);
}

void MemoryManagerGPU::clear()
{
    mTensors.clear();
}

void MemoryManagerGPU::copy(const Tensor& tensor, const raul::Name& nameGPU)
{
    getTensor(nameGPU).copyFrom(tensor);
}

void MemoryManagerGPU::copy(const raul::Name& nameGPU, Tensor& tensor) const
{
    getTensor(nameGPU).copyTo(tensor);
}

} // namespace raul