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

#include <map>
#include "OPOptimizer.hpp"

class MemoryReuseOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        std::map<std::string, int> endOfLife;
        int i;
        for (i = 0; i < spec->num_operator_specs; i++) {
            if (OT_None != spec->ops[i].type) {
                if (OT_Repeat == spec->ops[i].type) {
                    std::string startName = spec->ops[i].input_tensors_name[0];
                    int loopStart = searchOperatorIndexByName(spec, startName);
                    CHECK_REQUIREMENT(loopStart >= 0);
                    loops.push_back(std::make_pair(loopStart, i));
                    continue;
                }

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

        // model inputs should not overwrite each other
        // if model input is used within a loop, the space should not be reused
        for (int j = 0; j < spec->num_inputs; j++) {
            std::string inputName = spec->input_names[j];
            int lastId = endOfLife[inputName];

            bool loopExternal = false;
            for (auto loop : loops) {
                int loopStart = std::get<0>(loop);
                int loopEnd = std::get<1>(loop);
                if (lastId >= loopStart && lastId <= loopEnd) {
                    loopExternal = true;
                }
            }
            if (loopExternal) {
                aliveTensors.insert(std::make_pair(inputName, -1));
            } else {
                allocate(inputName, endOfLife[inputName], 0);
            }
        }

        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (OT_None != spec->ops[i].type) {
                U32 numInputs = spec->ops[i].num_inputs;
                U32 numOutputs = spec->ops[i].num_outputs;
                spec->ops[i].tensor_positions =
                    (I32 *)mt_new_storage((numInputs + numOutputs) * bytesOf(DT_I32));

                std::vector<std::tuple<std::string, int, U32>> layerTensors;

                for (U32 j = 0; j < numInputs; j++) {
                    std::string inputName = spec->ops[i].input_tensors_name[j];
                    auto it = aliveTensors.find(inputName);
                    if (it != aliveTensors.end()) {
                        // This tensor has already been allocated
                        spec->ops[i].tensor_positions[j] = it->second;
                        continue;
                    }

                    if (isOPtoBypass(spec->ops[i].type)) {
                        // Use -1 to label this tensor as standalone
                        aliveTensors.insert(std::make_pair(inputName, -1));
                        spec->ops[i].tensor_positions[j] = -1;
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

                    int lastId = endOfLife[outputName];

                    bool loopExternal = false;
                    for (auto loop : loops) {
                        int loopStart = std::get<0>(loop);
                        int loopEnd = std::get<1>(loop);
                        if (lastId >= loopStart && lastId <= loopEnd &&
                            (i < loopStart || i > loopEnd)) {
                            loopExternal = true;
                        }
                    }

                    if (loopExternal || isOPtoBypass(spec->ops[i].type)) {
                        // Use -1 to label this tensor as standalone
                        aliveTensors.insert(std::make_pair(outputName, -1));
                        spec->ops[i].tensor_positions[numInputs + j] = -1;
                        continue;
                    }

                    layerTensors.push_back(std::make_tuple(outputName, lastId, numInputs + j));
                }

                // Sort the unallocated tensors according to their death time
                sort(layerTensors.begin(), layerTensors.end(),
                    [=](std::tuple<std::string, int, U32> a, std::tuple<std::string, int, U32> b) {
                        return std::get<1>(a) > std::get<1>(b);
                    });

                for (auto tuple : layerTensors) {
                    std::string tensorName = std::get<0>(tuple);
                    int deathTime = std::get<1>(tuple);
                    U32 tensorID = std::get<2>(tuple);
                    spec->ops[i].tensor_positions[tensorID] = allocate(tensorName, deathTime, i);
                }

#if 0  //def _DEBUG
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    std::string inputName = spec->ops[i].input_tensors_name[j];
                    UNI_DEBUG_LOG("Input Tensor %s at %d\n", inputName.c_str(),
                        spec->ops[i].tensor_positions[j]);
                }

                for (U32 j = 0; j < spec->ops[i].num_outputs; j++) {
                    std::string outputName = spec->ops[i].output_tensors_name[j];
                    UNI_DEBUG_LOG("Output Tensor %s at %d\n", outputName.c_str(),
                        spec->ops[i].tensor_positions[numInputs + j]);
                }
#endif
            }
        }
        return hasOptimized;
    }

public:
private:
    std::vector<std::pair<std::string, int>> storages;

    std::map<std::string, int> aliveTensors;

    std::vector<std::pair<int, int>>
        loops;  // If a tensor used in a loop is produced outside, it should not be overwritten

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

    bool isOPtoBypass(OperatorType ot)
    {
        char *environmentSetting = getenv("BOLT_MEMORY_REUSE_OPTIMIZATION");
        bool memoryReuse =
            (environmentSetting != NULL && std::string(environmentSetting) == std::string("OFF"))
            ? false
            : true;
        if (!memoryReuse) {
            return true;
        }
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
            case OT_Jump: {
                return true;
            }
            case OT_LayerNorm: {
                return true;
            }
            default: {
                return false;
            }
        }
    }
};
#endif
