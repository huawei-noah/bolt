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

class MemoryTracker {
public:
    MemoryTracker()
    {
        this->storageSize.clear();
        this->tensorStoragePosition.clear();
        this->memoryNeedAssign = true;
    }

    void trackOpTensorSizes(std::shared_ptr<Operator> op, std::vector<std::string> tensorNames)
    {
        I32 *pos = op->get_tensor_positions().data();
        auto inputTensors = op->get_input_tensors();
        auto outputTensors = op->get_output_tensors();
        size_t numInput = inputTensors.size();
        size_t numOutput = outputTensors.size();
        for (size_t i = 0; i < numInput; i++) {
            U32 size = inputTensors[i].bytes();
            I32 slot = pos[i];
            this->tensorStoragePosition[tensorNames[i]] = slot;
            if (-1 == slot) {
                if (!memoryNeedAssign) {
                    if (size > inputTensors[i].capacity()) {
                        this->memoryNeedAssign = true;
                    }
                }
                continue;
            }
            this->trackSlotSize(slot, size);
        }
        for (size_t i = 0; i < numOutput; i++) {
            U32 size = outputTensors[i].bytes();
            I32 slot = pos[numInput + i];
            this->tensorStoragePosition[tensorNames[numInput + i]] = slot;
            if (-1 == slot) {
                if (!memoryNeedAssign) {
                    if (size > outputTensors[i].capacity()) {
                        this->memoryNeedAssign = true;
                    }
                }
                continue;
            }
            this->trackSlotSize(slot, size);
        }
    }

    I32 getSlotByTensorName(std::string name)
    {
        return tensorStoragePosition[name];
    }

    U32 getNumSlots()
    {
        return this->storageSize.size();
    }

    U32 getSizeSum()
    {
        U32 sum = 0;
        for (U32 size : this->storageSize) {
            sum += size;
        }
        return sum;
    }

    std::vector<U32> getStorageSize()
    {
        return this->storageSize;
    }

    void setMemoryAssigned()
    {
        this->memoryNeedAssign = false;
    }

    bool getMemoryNeedAssign()
    {
        return this->memoryNeedAssign;
    }

protected:
    void trackSlotSize(I32 slot, U32 size)
    {
        if (slot >= (I32)this->storageSize.size()) {
            this->storageSize.resize(slot + 1, 0);
        }
        if (size > this->storageSize[slot]) {
            this->storageSize[slot] = size;
            this->memoryNeedAssign = true;
        }
    }

    std::vector<U32> storageSize;
    std::map<std::string, I32> tensorStoragePosition;
    bool memoryNeedAssign;
};
#endif
