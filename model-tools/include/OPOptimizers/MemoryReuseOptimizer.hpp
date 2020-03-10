// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_MEMORYREUSEOPTIMIZER
#define _H_MEMORYREUSEOPTIMIZER

#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class MemoryReuseOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        bool hasOptimized = false;
        std::map<std::string, int> endOfLife;
        int i;
        for (i = 0; i < spec->num_operator_specs; i++) {
            if (OT_None != spec->ops[i].type) {
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    std::string inputName = spec->ops[i].input_tensors_name[j];
                    auto it = endOfLife.find(inputName);
                    if (it == endOfLife.end()) {
                        endOfLife.insert(std::make_pair(inputName, i));
                    } else {
                        it->second = i;
                    }
                }
            }
        }

        // model outputs should also be recorded
        for (int j = 0; j < spec->num_outputs; j++) {
            std::string outputName = spec->output_names[j];
            auto it = endOfLife.find(outputName);
            if (it == endOfLife.end()) {
                endOfLife.insert(std::make_pair(outputName, i));
            } else {
                it->second = i;
            }
        }

        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (OT_None != spec->ops[i].type) {
                U32 numInputs = spec->ops[i].num_inputs;
                U32 numOutputs = spec->ops[i].num_outputs;
                spec->ops[i].tensor_positions = (I32*)mt_new_storage((numInputs + numOutputs) * bytesOf(DT_I32));

                std::vector<std::tuple<std::string, int, U32>> layerTensors;

                for (U32 j = 0; j < numInputs; j++) {
                    std::string inputName = spec->ops[i].input_tensors_name[j];
                    auto it = aliveTensors.find(inputName);
                    if (it != aliveTensors.end()) {
                        // This tensor has already been allocated
                        spec->ops[i].tensor_positions[j] = it->second;
                        continue;
                    }

                    int lastId = endOfLife[inputName];
                    layerTensors.push_back(std::make_tuple(inputName, lastId, j));
                }

                for (U32 j = 0; j < numOutputs; j++) {
                    std::string outputName = spec->ops[i].output_tensors_name[j];
                    auto it = aliveTensors.find(outputName);
                    if (it != aliveTensors.end()) {
                        // This tensor has already been allocated
                        spec->ops[i].tensor_positions[numInputs + j] = it->second;
                        continue;
                    }

                    if (isOPtoBypass(spec->ops[i].type)) {
                        // Use -1 to label this tensor as standalone
                        aliveTensors.insert(std::make_pair(outputName, -1));
                        spec->ops[i].tensor_positions[numInputs + j] = -1;
                        continue;
                    }

                    int lastId = endOfLife[outputName];
                    layerTensors.push_back(std::make_tuple(outputName, lastId, numInputs + j));
                }

                // Sort the unallocated tensors according to their death time
                sort(layerTensors.begin(), layerTensors.end(), [=](auto a, auto b)
                {
                    return std::get<1>(a) > std::get<1>(b);
                });

                for (auto tuple : layerTensors) {
                    std::string tensorName = std::get<0>(tuple);
                    int deathTime = std::get<1>(tuple);
                    U32 tensorID = std::get<2>(tuple);
                    spec->ops[i].tensor_positions[tensorID] = allocate(tensorName, deathTime, i);
                }

#ifdef _DEBUG
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    std::string inputName = spec->ops[i].input_tensors_name[j];
                    std::cout << "Input Tensor " << inputName << " at " << spec->ops[i].tensor_positions[j] << std::endl;
                }

                for (U32 j = 0; j < spec->ops[i].num_outputs; j++) {
                    std::string outputName = spec->ops[i].output_tensors_name[j];
                    std::cout << "Output Tensor " << outputName << " at " << spec->ops[i].tensor_positions[numInputs+j] << std::endl;
                }
                std::cout << std::endl;
#endif
            }
        }
        return hasOptimized;
    }

public:

private:
    std::vector<std::pair<std::string, int>> storages;

    std::map<std::string, int> aliveTensors;

    I32 allocate(std::string tensorName, int deathTime, int curID)
    {
        I32 pos;
        for (pos = storages.size() - 1; pos >= 0; pos--) {
            if (curID > storages[pos].second) {
                storages[pos] = std::make_pair(tensorName, deathTime);
                break;
            }
        }
        if (-1 == pos) {
            pos = storages.size();
            storages.push_back(std::make_pair(tensorName, deathTime));
        }
        aliveTensors.insert(std::make_pair(tensorName, pos));
        return pos;
    }

    bool isOPtoBypass(OperatorType ot) {
        switch (ot) {
            case OT_None: {
                return true;
            }
            case OT_PreAllocatedMemory: {
                return true;
            }
            case OT_SharedWeight: {
                return true;
            }
            case OT_Check: {
                return true;
            }
            case OT_Repeat: {
                return true;
            }
            default: {
                return false;
            }
        }
    }
};
#endif
