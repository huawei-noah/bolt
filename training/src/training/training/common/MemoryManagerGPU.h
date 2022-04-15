// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef MEMORY_MANAGER_GPU_H
#define MEMORY_MANAGER_GPU_H

#include <unordered_map>

#include <training/common/TensorGPU.h>
#include <training/tools/NameGenerator.h>

namespace raul
{

class OpenCLKernelManager;

/**
 * @brief Memory managers GPU own GPU tensors and control their life time.
 */
class MemoryManagerGPU
{
  public:
    typedef TensorGPU tensor;
    typedef typename TensorGPU::type type;

    /**
     * @brief Construct a new Memory Manager
     *
     *  Note: The memory manager is uncopyable.
     *
     */
    MemoryManagerGPU();

    /**
     * @brief Set specific gpu attributes to process tensors on specific device
     * @param alignment Alignment of gpu memory (for sub buffers usually) in bytes
     */
    void setGpuAttribs(OpenCLKernelManager* manager, size_t alignment = 1u);

    /**
     * @brief Create a TensorGPU object of a specific shape
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @return TensorGPU* Pointer to an allocated tensor object
     *
     * @see TensorGPU
     */
    TensorGPU* createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width);

    /**
     * @brief Create a TensorGPU object of a specific shape
     *
     * @param name Name of the tensor
     * @param inShape Shape of tensor
     * @return TensorGPU* Pointer to an allocated tensor object
     *
     * @see TensorGPU
     */
    TensorGPU* createTensor(const Name& name, shape inShape);

    /**
     * @brief Create a TensorGPU object of a specific shape aligned to necessary sizes
     *
     * @param name Name of the tensor
     * @param inShape Shape of tensor
     * @return TensorGPU* Pointer to an allocated tensor object
     *
     * @see TensorGPU
     */
    TensorGPU* createTensorAligned(const Name& name, shape inShape);

    /**
     * @brief Create a TensorGPU object of a specific shape and randomly generated name
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @return TensorGPU* Pointer to an allocated tensor object
     *
     * @see TensorGPU
     */
    TensorGPU* createTensor(size_t batchSize, size_t depth, size_t height, size_t width);

    /**
     * @brief Create a TensorGPU object of a specific shape and randomly generated name
     *
     * @param name Name of the tensor
     * @param inShape Shape of tensor
     * @return TensorGPU* Pointer to an allocated tensor object
     *
     * @see TensorGPU
     */
    TensorGPU* createTensor(shape inShape);

    /**
     * @brief Create a TensorGPU object of a specific shape and inital value
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @param filler Initial value of the tensor (value must be dtype)
     * @return TensorGPU* Pointer to an allocated tensor object
     */
    TensorGPU* createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, const dtype filler);

    /**
     * @brief Create a TensorGPU object of a specific shape and inital value
     *
     * @param name Name of the tensor
     * @param shape TensorGPU shape
     * @param filler Initial value of the tensor (value must be dtype)
     * @return TensorGPU* Pointer to an allocated tensor object
     */
    TensorGPU* createTensor(const Name& name, shape inShape, const dtype filler);

    /**
     * @brief Create a TensorGPU object of a specific shape, inital value and randomly generated name
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @param filler Initial value of the tensor (value must be dtype)
     * @return TensorGPU* Pointer to an allocated tensor object
     */
    TensorGPU* createTensor(size_t batchSize, size_t depth, size_t height, size_t width, const dtype filler);

    /**
     * @brief Create a TensorGPU object of a specific shape without allocating memory
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @return TensorGPU* Pointer to an allocated tensor object
     *
     * @see TensorGPU
     */
    TensorGPU* createShape(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, AllocationMode allocMode);

    /**
     * @brief Create a TensorGPU object of a specific shape without allocating memory
     *
     * @param name Name of the tensor
     * @param shape TensorGPU shape
     *
     * @see TensorGPU
     */
    TensorGPU* createShape(const Name& name, shape inShape, AllocationMode allocMode);

    /**
     * @brief Create a TensorGPU object of a specific shape without allocating memory and randomly generated name
     *
     * @param sh TensorGPU shape
     * @return TensorGPU* Pointer to an allocated tensor object
     *
     * @see TensorGPU
     */
    TensorGPU* createShape(shape inShape);

    /**
     * @brief Get the tensor from memory by name
     *
     * @param name Name or alias of the tensor
     * @return TensorGPU& Reference to the tensor object
     */
    TensorGPU& getTensor(const raul::Name& name);

    /**
     * @brief Get the read-only tensor from memory by name
     *
     * @param name Name or alias of the tensor
     * @return const TensorGPU& Reference to the tensor object
     */
    const TensorGPU& getTensor(const raul::Name& name) const;

    /**
     * @brief Get the read-only tensor from memory by name
     *
     * @param name Name or alias of the tensor
     * @return const TensorGPU& Reference to the tensor object
     */
    const TensorGPU& operator()(const raul::Name& name) const;

    /**
     * @brief Get the tensor from memory by name
     *
     * @param name Name or alias of the tensor
     * @return TensorGPU& Reference to the tensor object
     */
    TensorGPU& operator()(const raul::Name& name);

    /**
     * @brief Get the tensor helper from memory by name
     *
     * @param name Name or alias of the tensor
     * @return TensorGPUHelper value to the helper tensorGPU object
     */
    TensorGPUHelper operator[](const raul::Name& name);

    /**
     * @brief Delete the tensor from memory by name
     *
     * @param name  Name or alias of the tensor
     */
    void deleteTensor(const raul::Name& name);

    /**
     * @brief Delete all tensor in the current namespace
     *
     */
    void clear();

    /**
     * @brief Get an amount of tensor in the current namespace
     *
     * @return size_t Amount of tensors
     */
    size_t size() const;

    /**
     * @brief Check if tensor exists in the current namespace
     *
     * @param name Name or alias of the tensor
     * @return true Exists
     * @return false Not exists
     */
    bool tensorExists(const raul::Name& name) const;

    /**
     * @brief Copy from Tensor to TensorGPU
     */
    void copy(const Tensor& tensor, const raul::Name& nameGPU);

    /**
     * @brief Copy from TensorGPU to Tensor
     */
    void copy(const raul::Name& nameGPU, Tensor& tensor) const;

    OpenCLKernelManager* getKernelManager() const { return mKernelManager; }

    /**
     * @brief Get GPU mem alignment (used for sub buffers) in bytes
     */
    size_t getAlignment() const { return mAlignment; }

  private:
    /**
     * @brief Internal utility method to check if tensor with provided name exists
     *
     *  Methods throws a runtime exception if name collision occurs.
     *
     * @param name Name or alias of the tensor
     * @param caller The function which requested the tensor
     * @param shouldExist Flag which specifies expectation of existence
     */
    void checkName(const Name& name, const char* caller, bool shouldExist = true) const;

    /**
     * @brief Check if tensor (aliases not included) exists in the current namespace
     *
     * @param name Name of the tensor
     * @return true Exists
     * @return false Not exists
     */
    bool checkTensorExists(const raul::Name& name) const;

    MemoryManagerGPU(const MemoryManagerGPU&) = delete;
    MemoryManagerGPU& operator=(const MemoryManagerGPU&) = delete;
    MemoryManagerGPU* operator&() = delete;

    OpenCLKernelManager* mKernelManager;

    using MemoryDict = std::unordered_map<raul::Name, std::shared_ptr<TensorGPU>>;
    MemoryDict mTensors;
    NameGenerator mTensorNameGenerator;

    size_t mAlignment;
};
} // raul namespace

#endif