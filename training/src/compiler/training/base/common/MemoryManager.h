// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <map>
#include <string>
#include <unordered_map>

#include <training/base/common/Tensor.h>
#include <training/system/NameGenerator.h>

namespace raul
{

template<typename T>
struct TensorsNamespace
{
    using MemoryDict = std::unordered_map<raul::Name, std::shared_ptr<T>>;
    using AliasesDict = std::unordered_map<raul::Name, raul::Name>;    // alias to tensor name mapping
    using TensorToAliasesDict = std::unordered_map<raul::Name, Names>; // tensor to alias mapping

    MemoryDict tensors;
    AliasesDict aliasToTensor;
    TensorToAliasesDict tensorToAliases;

    void clear()
    {
        tensors.clear();
        aliasToTensor.clear();
        tensorToAliases.clear();
    }

    size_t size() const { return tensors.size(); }
};

/**
 * @brief Memory managers own tensors and control their life time.
 *
 * MemoryManagerImpl creates tensor, shape or alias and owns it.
 * Tensor - object with NCHW description and allocated plain memory (size() = N*C*H*W).
 * Shape - Tensor without memory allocation (size() == 0).
 * Alias - alias to original tensor.
 */
template<typename T>
class MemoryManagerImpl
{
  public:
    typedef T tensor;
    typedef typename T::type type;

    /**
     * @brief Construct a new Memory Manager
     *
     *  Note: The memory manager is uncopyable.
     *
     */
    MemoryManagerImpl();

    /**
     * @brief Create a Tensor object of a specific shape
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width);

    /**
     * @brief Create a Tensor object of a specific shape
     *
     * @param name Name of the tensor
     * @param inShape Shape of tensor
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(const Name& name, shape inShape);

    /**
     * @brief Create a Tensor object of a specific shape with specific data
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @param list Data for tensor initialization
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, std::initializer_list<typename T::type> list);

    /**
     * @brief Create a Tensor object of a specific shape with specific data
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @param beginEnd Data range for tensor initialization
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, typename T::dt_range beginEnd);

    /**
     * @brief Create a Tensor object of a specific shape with specific data
     *
     * @param name Name of the tensor
     * @param inShape Shape of tensor
     * @param beginEnd Data range for tensor initialization
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(const Name& name, shape inShape, typename T::dt_range beginEnd);

    /**
     * @brief Create a Tensor object of a specific shape with specific data
     *
     * @param name Name of the tensor
     * @param inShape Shape of tensor
     * @param list Data for tensor initialization
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(const Name& name, shape inShape, std::initializer_list<typename T::type> list);

    /**
     * @brief Create a Tensor object of a specific shape and randomly generated name
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(size_t batchSize, size_t depth, size_t height, size_t width);

    /**
     * @brief Create a Tensor object of a specific shape and randomly generated name
     *
     * @param name Name of the tensor
     * @param inShape Shape of tensor
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(shape inShape);

    /**
     * @brief Create a Tensor object of a specific shape with specific data and randomly generated name
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @param list Data for tensor initialization
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(size_t batchSize, size_t depth, size_t height, size_t width, std::initializer_list<typename T::type> list);

    /**
     * @brief Create a Tensor object of a specific shape with specific data and randomly generated name
     *
     * @param name Name of the tensor
     * @param inShape Shape of tensor
     * @param list Data for tensor initialization
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createTensor(shape inShape, std::initializer_list<typename T::type> list);

    /**
     * @brief Create a Tensor object of a specific shape and inital value
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @param filler Initial value of the tensor (value must be dtype)
     * @return Tensor* Pointer to an allocated tensor object
     */
    T* createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, const typename T::type filler);

    /**
     * @brief Create a Tensor object of a specific shape and inital value
     *
     * @param name Name of the tensor
     * @param shape Tensor shape
     * @param filler Initial value of the tensor (value must be dtype)
     * @return Tensor* Pointer to an allocated tensor object
     */
    T* createTensor(const Name& name, shape inShape, const typename T::type filler);

    /**
     * @brief Create a Tensor object of a specific shape, inital value and randomly generated name
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @param filler Initial value of the tensor (value must be dtype)
     * @return Tensor* Pointer to an allocated tensor object
     */
    T* createTensor(size_t batchSize, size_t depth, size_t height, size_t width, const typename T::type filler);

    /**
     * @brief Create a Tensor object of a specific shape without allocating memory
     *
     * @param name Name of the tensor
     * @param batchSize Size of batches
     * @param depth Number of channels
     * @param height Height of tensor
     * @param width Width of tensor
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createShape(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, AllocationMode allocMode);

    /**
     * @brief Create a Tensor object of a specific shape without allocating memory
     *
     * @param name Name of the tensor
     * @param tensor Tensor to copy shape from
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createShape(const Name& name, const T& tensor);

    /**
     * @brief Create a Tensor object of a specific shape without allocating memory
     *
     * @param name Name of the tensor
     * @param shape Tensor shape
     *
     * @see Tensor
     */
    T* createShape(const Name& name, shape inShape, AllocationMode allocMode);

    /**
     * @brief Create a Tensor object of a specific shape without allocating memory and randomly generated name
     *
     * @param tensor Tensor to copy shape from
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createShape(const T& tensor);

    /**
     * @brief Create a Tensor object of a specific shape without allocating memory and randomly generated name
     *
     * @param sh Tensor shape
     * @return Tensor* Pointer to an allocated tensor object
     *
     * @see Tensor
     */
    T* createShape(shape inShape);

    /**
     * @brief Create an alias for existing Tensor object
     *
     * @param name Name of existing tensor
     * @param aliasName Alias for Name
     *
     * @see Tensor
     */
    void createAlias(const Name& name, const std::string& aliasName);

    /**
     * @brief Get the tensor from memory by name
     *
     * @param name Name or alias of the tensor
     * @return Tensor& Reference to the tensor object
     */
    T& getTensor(const raul::Name& name);

    /**
     * @brief Get the read-only tensor from memory by name
     *
     * @param name Name or alias of the tensor
     * @return const Tensor& Reference to the tensor object
     */
    const T& getTensor(const raul::Name& name) const;

    /**
     * @brief Get the read-only tensor from memory by name
     *
     * @param name Name or alias of the tensor
     * @return const Tensor& Reference to the tensor object
     */
    const T& operator[](const raul::Name& name) const;

    /**
     * @brief Get the tensor from memory by name
     *
     * @param name Name or alias of the tensor
     * @return Tensor& Reference to the tensor object
     */
    T& operator[](const raul::Name& name);

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
     * @brief Get full size of allocated memory for all namespaces
     *
     * @return size_t Size in bytes
     */
    size_t getTotalMemory() const;

    const TensorsNamespace<T>& getTensorCollection() const;

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

    /**
     * @brief Check if alias exists in the current namespace
     *
     * @param name Alias of the tensor
     * @return true Exists
     * @return false Not exists
     */
    bool checkAliasExists(const raul::Name& name) const;

    MemoryManagerImpl(const MemoryManagerImpl&) = delete;
    MemoryManagerImpl& operator=(const MemoryManagerImpl&) = delete;
    MemoryManagerImpl* operator&() = delete;

  private:
    TensorsNamespace<T> mTensors;
    NameGenerator mTensorNameGenerator;
};

typedef MemoryManagerImpl<Tensor> MemoryManager;
typedef MemoryManagerImpl<TensorFP16> MemoryManagerFP16;

} // raul namespace

#endif
