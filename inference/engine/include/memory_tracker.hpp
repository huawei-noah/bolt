// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MEMORY_TRACKER_H
#define _MEMORY_TRACKER_H

#include "tensor_desc.h"
#ifdef _USE_GPU
#include "image_manager.hpp"
#endif

class MemoryTracker {
public:
    MemoryTracker()
    {
        this->storageSize.clear();
        this->tensorStoragePosition.clear();
        this->memoryNeedAssign = true;
    }
    ~MemoryTracker(){}
    void trackOpTensorSizes(std::shared_ptr<Operator> op, std::vector<std::string> tensorNames)
    {
        I32 *pos = op->get_tensor_positions().data();
        auto inputTensors = op->get_input_tensors();
        auto outputTensors = op->get_output_tensors();
        size_t numInput = inputTensors.size();
        size_t numOutput = outputTensors.size();
        for (size_t i = 0; i < numInput; i++) {
            I32 slot = pos[i];
            this->tensorStoragePosition[tensorNames[i]] = slot;
            if (slot < 0) {
                this->trackSingleTensor(inputTensors[i]);
                continue;
            }
            this->trackSlotSize(slot, inputTensors[i]);
        }
        for (size_t i = 0; i < numOutput; i++) {
            I32 slot = pos[numInput + i];
            this->tensorStoragePosition[tensorNames[numInput + i]] = slot;
            if (slot < 0) {
                this->trackSingleTensor(outputTensors[i]);
                continue;
            }
            this->trackSlotSize(slot, outputTensors[i]);
        }
    }

    I32 getSlotByTensorName(std::string name)
    {
        return tensorStoragePosition[name];
    }

    U32 getNumSlots()
    {
        U32 size = storageSize.size();
#ifdef _USE_MALI
        size += imageManager.getNumImageVecs();
#endif
        return size;
    }

    U32 getSizeSum()
    {
        U32 sum = 0;
        for (U32 size : this->storageSize) {
            sum += size;
        }
#ifdef _USE_GPU
        sum += imageManager.getImageVecsSizeSum();
#endif
        return sum;
    }

    std::vector<U32> getStorageSize()
    {
        return this->storageSize;
    }

#ifdef _USE_GPU
    std::map<I32, std::vector<std::vector<U32>>> getOclImageSize()
    {
        return imageManager.getImageVecs();
    }

    I32 getOclImageSubSlotId(U32 slot, U32 *str)
    {
        I32 subSlot = imageManager.getImageVecsId(slot, str[0], str[1], str[2]);
        if (subSlot < 0) {
            UNI_ERROR_LOG("gpu image buffer reuse parameter is wrong.\n");
        }
        return subSlot;
    }
#endif

    void setMemoryAssigned()
    {
        this->memoryNeedAssign = false;
    }

    bool getMemoryNeedAssign()
    {
        return this->memoryNeedAssign;
    }

     
protected:
    void trackSingleTensor(Tensor tensor)
    {
        if (!memoryNeedAssign) {
            Memory *mem = tensor.get_memory();
            if (mem->get_mem_type() == OCLMem || mem->get_mem_type() == CPUMem) {
                checkSingleMem(mem);
            } else {
#ifdef _USE_GPU
                checkSingleImg((OclMemoryImg *)mem);
#endif
            }
        }
    }

    void trackSlotSize(I32 slot, Tensor tensor)
    {
        Memory *mem = tensor.get_memory();
        if (mem->get_mem_type() == OCLMem || mem->get_mem_type() == CPUMem) {
            U32 size = mem->bytes();
            buildStorageSize(slot, size);
        } else {
#ifdef _USE_GPU
            U32 str[3];
            OclMemoryImg *img = (OclMemoryImg *)mem;
            img->stride(str);
            buildOclImageSize(slot, str);
#endif
        }
    }

    void checkSingleMem(Memory *mem)
    {
        U32 size = mem->bytes();
        U32 capacity;
        mem->capacity(&capacity);
        if (size > capacity) {
            this->memoryNeedAssign = true;
        }
    }

    void buildStorageSize(I32 slot, U32 size)
    {
        if (slot >= (I32)this->storageSize.size()) {
            this->storageSize.resize(slot + 1, 0);
        }
        if (size > this->storageSize[slot]) {
            this->storageSize[slot] = size;
            this->memoryNeedAssign = true;
        }
    }

#ifdef _USE_GPU
    void checkSingleImg(OclMemoryImg *img)
    {
        U32 str[3];
        U32 capacity[3];
        img->stride(str);
        img->capacity(capacity);
        if (str[0] > capacity[0] || str[1] > capacity[1] || str[2] > capacity[2]) {
            this->memoryNeedAssign = true;
        }
    }

    void buildOclImageSize(I32 slot, U32 *size)
    {
        bool needExpand = imageManager.buildImageVecs(slot, size[0], size[1], size[2]);
        if (!this->memoryNeedAssign) {
            this->memoryNeedAssign = needExpand;
        }
    }
#endif

    std::vector<U32> storageSize;
    std::map<std::string, I32> tensorStoragePosition;
    bool memoryNeedAssign;
#ifdef _USE_GPU
    ImageManager imageManager;
#endif
};
#endif
