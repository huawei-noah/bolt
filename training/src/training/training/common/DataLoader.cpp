// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DataLoader.h"

#include <training/common/Conversions.h>

#include <fstream>
#include <iostream>

namespace raul
{

template<typename T>
int DataLoader::readArrayFromTextFile(const std::filesystem::path& path, T& array, size_t width, size_t height, size_t depth, size_t batch)
{
    std::ifstream file;
    file.open(path.string());
    if (file.is_open())
    {
        std::string line;
        size_t currRow = 0, currCol = 0, currDepth = 0, currBatch = 0;

        while (std::getline(file, line))
        {
            if (line.length() > 0 && line[0] == '#') continue;

            const char* begin = line.c_str();
            char* end = NULL;

            do
            {
                if (end != NULL)
                {
                    begin = end;
                }
                end = NULL;
                float v = strtof(begin, &end);
                if (end != begin)
                {
                    array[currCol + width * (currRow + height * (currDepth + currBatch * depth))] = static_cast<typename T::type>(v);

                    if (++currCol == width)
                    {
                        ++currRow;
                        currCol = 0;
                    }

                    if (currRow == height)
                    {
                        ++currDepth;
                        currRow = 0;
                    }

                    if (currDepth == depth)
                    {
                        ++currBatch;
                        currDepth = 0;
                    }
                }
            } while (end != begin);

            if (currBatch == batch) break;
        }
        // m[tensor] = TORANGE(array);
        file.close();
    }
    else
    {
        std::cerr << "Unable to open the file " << path.string() << std::endl;
        return 1;
    }
    return 0;
}

int DataLoader::readArrayFromTextFile(const std::filesystem::path& path, MemoryManagerGPU& m, const Name& tensor, size_t width, size_t height, size_t depth, size_t batch)
{
    std::ifstream file;
    file.open(path.string());
    if (file.is_open())
    {
        std::string line;
        size_t currRow = 0, currCol = 0, currDepth = 0, currBatch = 0;
        Tensor array(m(tensor).getShape());

        while (std::getline(file, line))
        {
            if (line.length() > 0 && line[0] == '#') continue;

            const char* begin = line.c_str();
            char* end = NULL;

            do
            {
                if (end != NULL)
                {
                    begin = end;
                }
                end = NULL;
                float v = strtof(begin, &end);
                if (end != begin)
                {
                    array[currCol + width * (currRow + height * (currDepth + currBatch * depth))] = v;

                    if (++currCol == width)
                    {
                        ++currRow;
                        currCol = 0;
                    }

                    if (currRow == height)
                    {
                        ++currDepth;
                        currRow = 0;
                    }

                    if (currDepth == depth)
                    {
                        ++currBatch;
                        currDepth = 0;
                    }
                }
            } while (end != begin);

            if (currBatch == batch) break;
        }
        m[tensor] = TORANGE(array);
        file.close();
    }
    else
    {
        std::cerr << "Unable to open the file " << path.string() << std::endl;
        return 1;
    }
    return 0;
}

void DataLoader::loadData(const std::filesystem::path& path, MemoryManagerGPU& m, const Name& tensor, size_t width, size_t height, size_t depth, size_t batch)
{
    if (m(tensor).getShape().total_size() != (batch * depth * height * width))
    {
        THROW_NONAME("DataLoader", "output wrong size [" + Conversions::toString(m(tensor).getShape().total_size()) + "], should be [" + Conversions::toString(batch * depth * height * width) + "]");
    }
    int err = readArrayFromTextFile(path, m, tensor, width, height, depth, batch);
    if (err != 0)
    {
        THROW_NONAME("DataLoader", "cannot load file: " + path.string());
    }
}

void DataLoader::loadData(const std::filesystem::path& path, MemoryManagerGPU& m, const Name& tensor)
{
    loadData(path, m, tensor, m(tensor).getShape().total_size(), 1U, 1U, 1U);
}

template<typename T>
void DataLoader::loadData(const std::filesystem::path& path, T& output, size_t width, size_t height, size_t depth, size_t batch)
{
    if (output.size() != (batch * depth * height * width))
    {
        THROW_NONAME("DataLoader", "output wrong size [" + Conversions::toString(output.size()) + "], should be [" + Conversions::toString(batch * depth * height * width) + "]");
    }
    int err = readArrayFromTextFile<T>(path, output, width, height, depth, batch);
    if (err != 0)
    {
        THROW_NONAME("DataLoader", "cannot load file: " + path.string());
    }
}

template void DataLoader::loadData<Tensor>(const std::filesystem::path& path, Tensor& output, size_t width, size_t height, size_t depth, size_t batch);
template void DataLoader::loadData<TensorFP16>(const std::filesystem::path& path, TensorFP16& output, size_t width, size_t height, size_t depth, size_t batch);

template<typename T>
void DataLoader::loadData(const std::filesystem::path& path, T& output)
{
    loadData<T>(path, output, output.getWidth() * output.getHeight() * output.getDepth() * output.getBatchSize(), 1U, 1U, 1U);
}

template void DataLoader::loadData<Tensor>(const std::filesystem::path& path, Tensor& output);
template void DataLoader::loadData<TensorFP16>(const std::filesystem::path& path, TensorFP16& output);

Tensor::dt_range DataLoader::loadData(const std::filesystem::path& path, size_t width, size_t height, size_t depth, size_t batch)
{
    Tensor* ret = mMemoryManager.createTensor(batch, depth, height, width);
    loadData(path, *ret, width, height, depth, batch);
    return *ret;
}

/// @todo(ck): use filesystem instead of string as pathPrefix type
Tensor::dt_range DataLoader::loadFilters(const std::string& pathPrefix, size_t fileIndexOffset, const std::string& pathPostfix, size_t width, size_t height, size_t depth, size_t filtersAmount)
{
    Tensor* ret = mMemoryManager.createTensor(1, 1, 1, filtersAmount * depth * height * width);

    Tensor* filter = mMemoryManager.createTensor(1, 1, 1, depth * height * width);

    for (size_t q = 0; q < filtersAmount; ++q)
    {
        DataLoader::loadData(pathPrefix + Conversions::toString(fileIndexOffset + q) + pathPostfix, *filter, width, height, depth);
        std::copy(filter->cbegin(), filter->cend(), ret->begin() + depth * height * width * q);
    }

    mMemoryManager.deleteTensor(filter->getName());

    return *ret;
}

const Tensor& DataLoader::buildOneHotVector(const TensorU8& labels, size_t labelsCount)
{
    Tensor* ret = mMemoryManager.createTensor(1, 1, 1, labels.size() * labelsCount, 0.0_dt);

    for (size_t i = 0; i < labels.size(); i++)
    {
        (*ret)[labels[i] + labelsCount * i] = 1.0_dt;
    }

    return *ret;
}

Tensor& DataLoader::createTensor(size_t size)
{
    Tensor* ret = mMemoryManager.createTensor(1, 1, 1, size);
    return *ret;
}

} // raul namespace
