// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MemoryManager.h"

namespace raul
{

using namespace std::string_literals;

template<typename T>
MemoryManagerImpl<T>::MemoryManagerImpl()
{
    mTensorNameGenerator.setPrefix("Tensor_");
}

template<typename T>
size_t MemoryManagerImpl<T>::size() const
{
    return mTensors.size();
}

template<typename T>
const TensorsNamespace<T>& MemoryManagerImpl<T>::getTensorCollection() const
{
    return mTensors;
}

template<typename T>
bool MemoryManagerImpl<T>::tensorExists(const raul::Name& name) const
{
    return checkTensorExists(name) || checkAliasExists(name);
}

template<typename T>
bool MemoryManagerImpl<T>::checkTensorExists(const raul::Name& name) const
{
    return mTensors.tensors.find(name) != mTensors.tensors.end();
}

template<typename T>
bool MemoryManagerImpl<T>::checkAliasExists(const raul::Name& name) const
{
    return mTensors.aliasToTensor.find(name) != mTensors.aliasToTensor.end();
}

template<typename T>
void MemoryManagerImpl<T>::checkName(const Name& name, const char* caller, bool shouldExist) const
{
    if (name.empty())
    {
        THROW_NONAME("MemoryManagerImpl<type="s + BASE_TYPE_NAME(type) + ">", "empty name (requested by "s + caller + ")");
    }
    if (tensorExists(name) != shouldExist)
    {
        THROW_NONAME("MemoryManagerImpl<type="s + BASE_TYPE_NAME(type) + ">", "tensor '" + name + (shouldExist ? "' doesn't exist" : "' already exists") + " (requested by "s + caller + ")");
    }
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, batchSize, depth, height, width, AllocationMode::STANDARD);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(const Name& name, raul::shape inShape)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, inShape);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, std::initializer_list<typename T::type> list)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, batchSize, depth, height, width, list);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, typename T::dt_range beginEnd)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, batchSize, depth, height, width, beginEnd);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(const Name& name, shape inShape, typename T::dt_range beginEnd)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, inShape, beginEnd);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(const Name& name, shape inShape, std::initializer_list<typename T::type> list)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, inShape, list);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(size_t batchSize, size_t depth, size_t height, size_t width)
{
    return createTensor(mTensorNameGenerator.generate(), batchSize, depth, height, width);
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(shape inShape)
{
    return createTensor(mTensorNameGenerator.generate(), inShape);
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(size_t batchSize, size_t depth, size_t height, size_t width, std::initializer_list<typename T::type> list)
{
    return createTensor(mTensorNameGenerator.generate(), batchSize, depth, height, width, list);
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(shape inShape, std::initializer_list<typename T::type> list)
{
    return createTensor(mTensorNameGenerator.generate(), inShape, list);
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, const typename T::type filler)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, batchSize, depth, height, width, filler);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(const Name& name, shape inShape, const typename T::type filler)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, inShape, filler);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createTensor(size_t batchSize, size_t depth, size_t height, size_t width, const typename T::type filler)
{
    return createTensor(mTensorNameGenerator.generate(), batchSize, depth, height, width, filler);
}

template<typename T>
T* MemoryManagerImpl<T>::createShape(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, AllocationMode allocMode)
{
    checkName(name, __func__, false);

    mTensors.tensors[name] = std::make_shared<T>(name, batchSize, depth, height, width, allocMode, false);

    return mTensors.tensors[name].get();
}

template<typename T>
T* MemoryManagerImpl<T>::createShape(const Name& name, const T& tensor)
{
    return createShape(name, tensor.getBatchSize(), tensor.getDepth(), tensor.getHeight(), tensor.getWidth(), AllocationMode::STANDARD);
}

template<typename T>
T* MemoryManagerImpl<T>::createShape(const Name& name, shape inShape, AllocationMode allocMode)
{
    return createShape(name, inShape[0], inShape[1], inShape[2], inShape[3], allocMode);
}

template<typename T>
T* MemoryManagerImpl<T>::createShape(const T& tensor)
{
    return createShape(mTensorNameGenerator.generate(), tensor.getBatchSize(), tensor.getDepth(), tensor.getHeight(), tensor.getWidth(), AllocationMode::STANDARD);
}

template<typename T>
T* MemoryManagerImpl<T>::createShape(shape inShape)
{
    return createShape(mTensorNameGenerator.generate(), inShape);
}

template<typename T>
void MemoryManagerImpl<T>::createAlias(const Name& name, const std::string& aliasName)
{
    if (name.empty())
    {
        THROW_NONAME("MemoryManagerImpl<type="s + BASE_TYPE_NAME(type) + ">", "empty name");
    }
    if (aliasName.empty())
    {
        THROW_NONAME("MemoryManagerImpl<type="s + BASE_TYPE_NAME(type) + ">", "empty alias");
    }

    bool isNameAlias = !checkTensorExists(name) && checkAliasExists(name);

    if (!isNameAlias)
    {
        if (!checkTensorExists(name))
        {
            THROW_NONAME("MemoryManagerImpl<type="s + BASE_TYPE_NAME(type) + ">", "tensor '" + name + "' doesn`t exist");
        }
        if (checkAliasExists(name))
        {
            THROW_NONAME("MemoryManagerImpl<type="s + BASE_TYPE_NAME(type) + ">", "alias '" + name + "' already exists");
        }
    }

    if (checkTensorExists(aliasName))
    {
        THROW_NONAME("MemoryManagerImpl<type="s + BASE_TYPE_NAME(type) + ">", "tensor '" + name + "' already exists");
    }
    if (checkAliasExists(aliasName))
    {
        THROW_NONAME("MemoryManagerImpl<type="s + BASE_TYPE_NAME(type) + ">", "alias '" + name + "' already exists");
    }

    if (isNameAlias)
    {
        std::string tensorName = mTensors.aliasToTensor[name];

        mTensors.aliasToTensor[aliasName] = tensorName;
        mTensors.tensorToAliases[tensorName].push_back(aliasName);
    }
    else
    {
        mTensors.aliasToTensor[aliasName] = name;
        mTensors.tensorToAliases[name].push_back(aliasName);
    }
}

template<typename T>
T& MemoryManagerImpl<T>::getTensor(const raul::Name& name)
{
    checkName(name, __func__);

    std::string tensorName = name;

    bool isNameAlias = !checkTensorExists(name) && checkAliasExists(name);
    if (isNameAlias)
    {
        tensorName = mTensors.aliasToTensor[name];
    }

    return *mTensors.tensors.find(tensorName)->second;
}

template<typename T>
const T& MemoryManagerImpl<T>::getTensor(const raul::Name& name) const
{
    checkName(name, __func__);

    std::string tensorName = name;

    bool isNameAlias = !checkTensorExists(name) && checkAliasExists(name);
    if (isNameAlias)
    {
        tensorName = mTensors.aliasToTensor.find(name)->second;
    }

    return *mTensors.tensors.find(tensorName)->second;
}

template<typename T>
const T& MemoryManagerImpl<T>::operator[](const raul::Name& name) const
{
    return getTensor(name);
}

template<typename T>
T& MemoryManagerImpl<T>::operator[](const raul::Name& name)
{
    return getTensor(name);
}

template<typename T>
void MemoryManagerImpl<T>::deleteTensor(const raul::Name& name)
{
    checkName(name, __func__);

    std::string tensorName = name;

    bool isNameAlias = !checkTensorExists(name) && checkAliasExists(name);
    if (isNameAlias)
    {
        tensorName = mTensors.aliasToTensor[name];
    }

    auto it = mTensors.tensorToAliases.find(tensorName);

    if (it != mTensors.tensorToAliases.end())
    {
        for (const auto& alias : it->second)
        {
            mTensors.aliasToTensor.erase(alias);
        }

        mTensors.tensorToAliases.erase(tensorName);
    }

    mTensors.tensors.erase(tensorName);
}

template<typename T>
void MemoryManagerImpl<T>::clear()
{
    mTensors.clear();
}

template<typename T>
size_t MemoryManagerImpl<T>::getTotalMemory() const
{
    size_t ret = 0;

    for (const auto& it : mTensors.tensors)
    {
        ret += it.second->size();
    }

    ret *= sizeof(dtype);

    return ret;
}

template class MemoryManagerImpl<Tensor>;
template class MemoryManagerImpl<TensorFP16>;

} // namespace raul
