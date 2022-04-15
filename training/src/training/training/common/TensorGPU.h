// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TENSOR_GPU_H
#define TENSOR_GPU_H

#include <training/common/OpenCLInclude.h>

#include <training/common/Common.h>

namespace raul
{
template<typename Type>
class TensorImpl;
class OpenCLKernelManager;
typedef TensorImpl<dtype> Tensor;

/**
 * @brief Manages GPU memory. Underlying buffer (cl::Buffer) and shape.total_size() might have different sizes
 */
class TensorGPU
{
  public:
    typedef dtype type;

    explicit TensorGPU() = delete;
    explicit TensorGPU(const TensorGPU&) = delete;
    explicit TensorGPU(TensorGPU&&) = delete;

    TensorGPU(OpenCLKernelManager* manager, const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, AllocationMode allocMode = AllocationMode::STANDARD, bool isAllocate = true);

    // for alignment
    TensorGPU(OpenCLKernelManager* manager, const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, size_t bufferSize);

    TensorGPU(OpenCLKernelManager* manager, const Tensor& tensor);

    // filler
    TensorGPU(OpenCLKernelManager* manager, const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, dtype filler);

    /**
     * set data with size check (copy)
     */
    void copy(const TensorGPU& tensor);

    void copyFrom(const Tensor& tensor);
    void copyTo(Tensor& tensor) const;

    size_t size() const { return mBufferSize; }
    bool empty() const { return mBufferSize == 0; }

    void memAllocate(const cl::Buffer& buffer); // d.polubotko: do not use directly in layers, should be used by actions only
    void memClear();                            // d.polubotko: do not use directly in layers, should be used by actions only

    const std::string& getName() const { return mName; }
    const std::string getDescription() const;

    size_t getBatchSize() const { return mShape[0]; }
    size_t getDepth() const { return mShape[1]; }
    size_t getHeight() const { return mShape[2]; }
    size_t getWidth() const { return mShape[3]; }

    shape getShape() const { return mShape; }

    bool isBroadcastableTo(const shape& to_shape) const noexcept { return Common::shapeIsBroadcastable(getShape(), to_shape); }

    AllocationMode getAllocationMode() const { return mAllocationMode; }

    cl::Buffer& getBuffer() { return mMem; }
    const cl::Buffer& getBuffer() const { return mMem; }

    friend std::ostream& operator<<(std::ostream& out, const TensorGPU& instance);

  private:
    std::string mName;

    AllocationMode mAllocationMode;

    cl::Buffer mMem;
    size_t mBufferSize; // elements, size of underlying buffer and shape total_size might be different
    OpenCLKernelManager* mManager;
    shape mShape;
};

/**
 * @brief Class to simplify memory copy from/to CPU/GPU
 */
class TensorGPUHelper
{
  public:
    TensorGPUHelper(TensorGPU& tensor, OpenCLKernelManager* manager)
        : mTensor(tensor)
        , mManager(manager)
    {
    }

    /**
     * @brief Copy from one buffer to another. Assumed CommandQueue should be same for both objects (no check performed)
     */
    TensorGPUHelper& operator=(const TensorGPUHelper& tensor);

    TensorGPUHelper& operator=(const Tensor& tensor);

    TensorGPUHelper& operator=(dtype value);

    void copyTo(Tensor& tensor) const;

    TensorGPU& getTensor() { return mTensor; }
    const TensorGPU& getTensor() const { return mTensor; }

    cl::CommandQueue& getQueue();
    const cl::CommandQueue& getQueue() const;
    OpenCLKernelManager* getKernelManager() const { return mManager; }

    void finish() const { getQueue().finish(); }

  private:
    TensorGPU& mTensor;
    OpenCLKernelManager* mManager;
};
} // raul namespace

#endif
