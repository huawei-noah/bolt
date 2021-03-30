// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TINYBERT_TEST
#define _H_TINYBERT_TEST

#ifdef _USE_OPENMP
#include <omp.h>
#endif

#include "inference.hpp"
#include "data_loader.hpp"
#include "profiling.h"
#include "parse_command.h"

static std::string tinybertTestKernel(U32 sequenceIndex,
    std::vector<Tensor> sequence,
    std::shared_ptr<CNN> pipeline,
    std::vector<std::vector<Tensor>> intents,
    std::vector<std::vector<Tensor>> slots,
    int *falseIntent,
    int *falseSlot,
    const char **inputNames,
    const char **outputNames,
    bool useGPU)
{
    std::map<std::string, TensorDesc> inputDescMap;
    inputDescMap[inputNames[0]] = sequence[0].get_desc();
    inputDescMap[inputNames[1]] = sequence[1].get_desc();
    inputDescMap[inputNames[2]] = sequence[2].get_desc();
    pipeline->reready(inputDescMap);

    std::map<std::string, std::shared_ptr<U8>> inputs;
    inputs[inputNames[0]] = ((CpuMemory *)sequence[0].get_memory())->get_shared_ptr();
    inputs[inputNames[1]] = ((CpuMemory *)sequence[1].get_memory())->get_shared_ptr();
    inputs[inputNames[2]] = ((CpuMemory *)sequence[2].get_memory())->get_shared_ptr();

    pipeline->set_input_by_assign(inputs);

    pipeline->run();

    Tensor intentSoftmax = pipeline->get_tensor_by_name(outputNames[0]);

    U32 intentNum = intentSoftmax.length();
    U32 intentMaxIndex = 0;
#ifdef _USE_MALI
    if (useGPU) {
        auto mem = (OclMemory *)intentSoftmax.get_memory();
        mem->get_mapped_ptr();
    }
#endif
    for (U32 index = 1; index < intentNum; index++) {
        if (intentSoftmax.element(index) > intentSoftmax.element(intentMaxIndex)) {
            intentMaxIndex = index;
        }
    }
    std::string log = std::string(" intent: ") + std::to_string(intentMaxIndex) + std::string(" ") +
        std::to_string(intentSoftmax.element(intentMaxIndex));
    if (intents.size() > 0) {
        F32 *intentResult =
            (F32 *)((CpuMemory *)(intents[sequenceIndex][0].get_memory()))->get_ptr();
        if (intentMaxIndex != intentResult[0] ||
            abs(intentSoftmax.element(intentMaxIndex) - intentResult[1]) > 0.1) {
            (*falseIntent)++;
        }
    }
    Tensor slotSoftmax = pipeline->get_tensor_by_name(outputNames[1]);
    auto slotDesc = slotSoftmax.get_desc();
    U32 slotNum = slotDesc.dims[1];
    U32 slotRange = slotDesc.dims[0];
#ifdef _USE_MALI
    if (useGPU) {
        auto mem = (OclMemory *)slotSoftmax.get_memory();
        mem->get_mapped_ptr();
    }
#endif
    std::vector<U32> slotSoftmaxResult;
    log += std::string(" slot: ");
    for (U32 i = 0; i < slotNum; i++) {
        U32 slotMaxIndex = 0;
        for (U32 index = 1; index < slotRange; index++) {
            if (slotSoftmax.element(i * slotRange + index) >
                slotSoftmax.element(i * slotRange + slotMaxIndex)) {
                slotMaxIndex = index;
            }
        }
        slotSoftmaxResult.push_back(slotMaxIndex);
        log += std::to_string(slotMaxIndex) + std::string(" ");
    }
    if (slots.size() > sequenceIndex) {
        U32 *slotResult = (U32 *)((CpuMemory *)(slots[sequenceIndex][0].get_memory()))->get_ptr();
        for (U32 i = 0; i < slotSoftmaxResult.size(); i++) {
            if (slotSoftmaxResult.size() != slots[sequenceIndex][0].get_desc().dims[0] ||
                slotResult[i] != slotSoftmaxResult[i]) {
                (*falseSlot)++;
                break;
            }
        }
    }
    return log;
}

inline void tinybertTest(int argc,
    char **argv,
    const char **inputNames,
    const char **outputNames,
    F32 *intentRate,
    F32 *slotRate)
{
    UNI_TIME_INIT
    ParseRes parse_res;
    parseCommandLine(argc, argv, &parse_res, "examples");

    char *modelPath = (char *)"";
    char *sequenceDirectory = (char *)"";
    char *affinityPolicyName = (char *)"";
    char *algorithmMapPath = (char *)"";
    int loopTime = 1;

    if (!parse_res.model.second) {
        exit(-1);
    }
    if (parse_res.model.second) {
        modelPath = parse_res.model.first;
    }
    if (parse_res.inputPath.second) {
        sequenceDirectory = parse_res.inputPath.first;
    }
    if (parse_res.archInfo.second) {
        affinityPolicyName = parse_res.archInfo.first;
    }
    if (parse_res.algoPath.second) {
        algorithmMapPath = parse_res.algoPath.first;
    }
    if (parse_res.loopTime.second) {
        loopTime = parse_res.loopTime.first;
    }

    bool useGPU = (strcmp(affinityPolicyName, "GPU") == 0) ? true : false;
    std::shared_ptr<CNN> pipelineBase;
    UNI_PROFILE(pipelineBase = createPipeline(affinityPolicyName, modelPath, algorithmMapPath),
        std::string("bolt::prepare"), std::string("prepare"));

    // load sequences
    std::map<std::string, std::shared_ptr<Tensor>> inMap = pipelineBase->get_input();
    std::vector<TensorDesc> sequenceDescs;
    TensorDesc wordInputDesc = (*(inMap[inputNames[0]])).get_desc();
    wordInputDesc.dt = DT_U32;
    sequenceDescs.push_back(wordInputDesc);
    TensorDesc positionInputDesc = (*(inMap[inputNames[1]])).get_desc();
    positionInputDesc.dt = DT_U32;
    sequenceDescs.push_back(positionInputDesc);
    TensorDesc tokenTypeInputDesc = (*(inMap[inputNames[2]])).get_desc();
    tokenTypeInputDesc.dt = DT_U32;
    sequenceDescs.push_back(tokenTypeInputDesc);
    std::vector<std::vector<Tensor>> sequences, intents, slots;
    std::vector<std::string> sequencePaths =
        load_data(sequenceDirectory + std::string("/input"), sequenceDescs, &sequences);

    // load result
    std::vector<TensorDesc> intentDescs;
    TensorDesc intentDesc = tensor1d(DT_F32, 2);
    intentDescs.push_back(intentDesc);
    std::vector<std::string> intentPaths =
        load_data(sequenceDirectory + std::string("/intent"), intentDescs, &intents);
    std::vector<TensorDesc> slotDescs;
    slotDescs.push_back(wordInputDesc);
    std::vector<std::string> slotPaths =
        load_data(sequenceDirectory + std::string("/slot"), slotDescs, &slots);

    int falseIntent = 0;
    int falseSlot = 0;
#ifdef _USE_OPENMP
    double timeBegin = ut_time_ms();
#pragma omp parallel num_threads(OMP_NUM_THREADS)
    {
        if (useGPU) {
            UNI_ERROR_LOG("GPU mode has not support OpenMP for tinybert\n");
            exit(1);
        }
        std::shared_ptr<CNN> pipeline = std::shared_ptr<CNN>(new CNN());
        int threadId = omp_get_thread_num();
        UNI_PROFILE(*pipeline = pipelineBase->clone(),
            std::string("bolt::clone-") + std::to_string(threadId), std::string("clone"));
        pipeline->set_runtime_device(threadId, threadId);
#pragma omp for
        for (U32 sequenceIndex = 0; sequenceIndex < sequences.size(); sequenceIndex++) {
            std::string log = sequencePaths[sequenceIndex] + ":" +
                tinybertTestKernel(sequenceIndex, sequences[sequenceIndex], pipeline, intents,
                    slots, &falseIntent, &falseSlot, inputNames, outputNames, useGPU);
            UNI_INFO_LOG("%s\n", log.c_str());
        }
    }
#else
#ifdef _USE_MALI
    /*warp up*/
    if (useGPU) {
        pipelineBase->run();
    }
#endif
    double timeBegin = ut_time_ms();
    for (int i = 0; i < loopTime; ++i) {
        for (U32 sequenceIndex = 0; sequenceIndex < sequences.size(); sequenceIndex++) {
            std::string log = sequencePaths[sequenceIndex] + ":" +
                tinybertTestKernel(sequenceIndex, sequences[sequenceIndex], pipelineBase, intents,
                    slots, &falseIntent, &falseSlot, inputNames, outputNames, useGPU);
            UNI_INFO_LOG("%s\n", log.c_str());
        }
    }
#endif
    double timeEnd = ut_time_ms();
    pipelineBase->saveAlgorithmMapToFile(algorithmMapPath);
    double totalTime = (timeEnd - timeBegin) / loopTime;
    UNI_TIME_STATISTICS
    U32 validSequence = UNI_MAX(1, sequences.size());
    *intentRate = 100.0 * (validSequence - falseIntent) / validSequence;
    *slotRate = 100.0 * (validSequence - falseSlot) / validSequence;
    UNI_CI_LOG("intent correct rate: %f %%\n", *intentRate);
    UNI_CI_LOG("slot   correct rate: %f %%\n", *slotRate);
    UNI_CI_LOG("avg_time:%fms/sequence\n", 1.0 * totalTime / validSequence);
}
#endif  // _H_TINYBERT_TEST
