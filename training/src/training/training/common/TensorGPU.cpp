// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TensorGPU.h"
#include "Tensor.h"

#include <training/opencl/OpenCLKernelManager.h>

namespace raul
{
TensorGPU::TensorGPU(OpenCLKernelManager* manager, const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, AllocationMode allocMode, bool isAllocate)
    : mName(name)
    , mAllocationMode(allocMode)
    , mBufferSize(0)
    , mManager(manager)
    , mShape(batchSize, depth, height, width)
{
    if (isAllocate)
    {
        if (mAllocationMode == AllocationMode::STANDARD)
        {
            auto caller = "TensorGPU[" + mName + "::ctor";
            mBufferSize = mShape.total_size();
            mMem = mManager->createBuffer(mBufferSize * sizeof(dtype), caller);
        }
    }
}

TensorGPU::TensorGPU(OpenCLKernelManager* manager, const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, size_t bufferSize)
    : mName(name)
    , mAllocationMode(AllocationMode::STANDARD)
    , mBufferSize(bufferSize)
    , mManager(manager)
    , mShape(batchSize, depth, height, width)
{
    auto caller = "TensorGPU[" + mName + "::ctor]";
    mMem = mManager->createBuffer(mBufferSize * sizeof(dtype), caller);
}

TensorGPU::TensorGPU(OpenCLKernelManager* manager, const Tensor& tensor)
    : mName(tensor.getName())
    , mAllocationMode(AllocationMode::STANDARD)
    , mBufferSize(tensor.size())
    , mManager(manager)
    , mShape(tensor.getShape())
{
    mBufferSize = mShape.total_size();
    auto caller = "TensorGPU[" + mName + "::ctor]";
    mMem = mManager->createBuffer(mBufferSize * sizeof(dtype), caller);
    mManager->writeBuffer(mMem, mBufferSize * sizeof(dtype), &tensor[0], caller);
}

TensorGPU::TensorGPU(OpenCLKernelManager* manager, const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, dtype filler)
    : mName(name)
    , mAllocationMode(AllocationMode::STANDARD)
    , mManager(manager)
    , mShape(batchSize, depth, height, width)
{
    mBufferSize = mShape.total_size();
    auto caller = "TensorGPU[" + mName + "::ctor]";
    mMem = mManager->createBuffer(mBufferSize * sizeof(dtype), caller);
    mManager->fillBuffer(mMem, filler, caller);
}

std::ostream& operator<<(std::ostream& out, const TensorGPU& instance)
{
    out << "TensorGPU";
    const auto name = instance.getName();
    if (!name.empty())
    {
        out << " '" << name << "'";
    }
    out << " " << seq2str(instance.getShape());

    if (io::tensor::isSetFlag(out, io::tensor::TensorView::size))
    {
        out << ", size: " << instance.size() * sizeof(TensorGPU::type);
    }
    if (io::tensor::isSetFlag(out, io::tensor::TensorView::scale))
    {
        out << ", scale: no";
    }

    if (io::tensor::isSetFlag(out, io::tensor::TensorView::content))
    {
        out << std::endl << "[data not available]";
    }
    return out;
}

void TensorGPU::copy(const TensorGPU& tensor)
{
    auto caller = "TensorGPU[" + mName + "::copy]";
    if (tensor.size() != size())
    {
        THROW("TensorGPU", mName, " size mismatch");
    }
    mManager->copyBuffer(tensor.mMem, mMem, caller);
}

void TensorGPU::copyFrom(const Tensor& tensor)
{
    auto caller = "TensorGPU[" + mName + "::copyFrom]";
    if (tensor.size() > size() || tensor.empty() != empty()) // due to alignment on GPU if pool (subbufer) used
    {
        THROW("TensorGPU", mName, "size mismatch");
    }
    mManager->writeBuffer(mMem, tensor.size() * sizeof(dtype), &tensor[0], caller);
}

void TensorGPU::copyTo(Tensor& tensor) const
{
    auto caller = "TensorGPU[" + mName + "::copyTo]";
    if (tensor.size() > size() || tensor.empty() != empty()) // due to alignment on GPU if pool (subbufer) used
    {
        THROW("TensorGPU", mName, "size mismatch");
    }
    mManager->readBuffer(mMem, tensor.size() * sizeof(dtype), &tensor[0], caller);
}

void TensorGPU::memClear()
{
    mMem = cl::Buffer();

    mBufferSize = 0;
}

void TensorGPU::memAllocate(const cl::Buffer& buffer)
{
    cl_int status;
    mBufferSize = buffer.getInfo<CL_MEM_SIZE>(&status) / sizeof(dtype);
    Common::checkOpenCLStatus(status, "TensorGPU[" + mName + "::memAllocate", "getInfo error");
    mMem = buffer;
}

const std::string TensorGPU::getDescription() const
{
    return mName + " " + seq2str(getShape());
}

TensorGPUHelper& TensorGPUHelper::operator=(const TensorGPUHelper& t)
{
    // Assumed CommandQueue should be same for both objects (no check performed)

    mTensor.copy(t.mTensor);
    return *this;
}

TensorGPUHelper& TensorGPUHelper::operator=(const Tensor& t)
{
    mTensor.copyFrom(t);
    return *this;
}

TensorGPUHelper& TensorGPUHelper::operator=(dtype val)
{
    mManager->fillBuffer(mTensor.getBuffer(), val, "TensorGPUHelper[operator=]");
    return *this;
}

void TensorGPUHelper::copyTo(Tensor& tensor) const
{
    mTensor.copyTo(tensor);
}

cl::CommandQueue& TensorGPUHelper::getQueue()
{
    return mManager->getCommandQueue();
}

const cl::CommandQueue& TensorGPUHelper::getQueue() const
{
    return mManager->getCommandQueue();
}

} // namespace raul
