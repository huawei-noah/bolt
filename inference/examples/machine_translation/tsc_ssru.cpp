// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TSC_SSRU_TEST
#define _H_TSC_SSRU_TEST

#include "inference.hpp"
#include "data_loader.hpp"
#include "profiling.h"
#include "parse_command.h"

int main(int argc, char *argv[])
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
        return 1;
    }
    if (parse_res.model.second) {
        modelPath = parse_res.model.first;
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
    std::shared_ptr<CNN> pipelineBase;
    UNI_PROFILE(pipelineBase = createPipeline(affinityPolicyName, modelPath, algorithmMapPath),
        std::string("bolt::prepare"), std::string("prepare"));

    U32 batch = 4;
    U32 inputLen = 55;
    U32 seqLen = batch * inputLen;
    U32 shortlistLen = 25100;
    U32 input_ids[] = {
        2583,
        16370,
        422,
        175,
        11445,
        38,
        156,
        16718,
        13,
        345,
        1485,
        3677,
        2,
        2905,
        845,
        17379,
        7408,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2583,
        16370,
        43,
        5,
        2905,
        845,
        17379,
        109,
        16740,
        4,
        3339,
        12550,
        19144,
        55,
        257,
        7,
        156,
        18,
        1961,
        22348,
        1609,
        30,
        4,
        22068,
        12143,
        7,
        18,
        1394,
        609,
        172,
        4,
        1634,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5999,
        1567,
        55,
        1588,
        2,
        331,
        15,
        1311,
        16969,
        8,
        6134,
        7,
        15,
        3770,
        7120,
        823,
        5,
        75,
        55,
        679,
        4508,
        2,
        5036,
        6753,
        47,
        16370,
        14288,
        4,
        3540,
        4862,
        6112,
        623,
        156,
        1124,
        82,
        278,
        1981,
        150,
        122,
        18183,
        55,
        13,
        42,
        15,
        33,
        4759,
        569,
        85,
        62,
        6,
        4,
        910,
        3873,
        3,
        0,
        1056,
        345,
        1485,
        3677,
        2,
        122,
        278,
        7088,
        107,
        1089,
        21486,
        9584,
        5,
        8,
        1329,
        11445,
        38,
        156,
        16718,
        13,
        15880,
        10997,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };
    F32 *masks = (F32 *)malloc(seqLen * sizeof(F32));
    U32 h_sequence_length[] = {18, 34, 55, 24};
    for (U32 i = 0; i < batch; ++i) {
        for (U32 j = 0; j < h_sequence_length[i]; ++j) {
            masks[i * inputLen + j] = 1.0f;
        }
        for (U32 j = h_sequence_length[i]; j < inputLen; ++j) {
            masks[i * inputLen + j] = 0.0f;
        }
    }

    U32 *positions = (U32 *)malloc(seqLen * sizeof(U32));
    for (U32 i = 0; i < seqLen; ++i) {
        positions[i] = i % inputLen;
    }

    U32 *shortlist = (U32 *)malloc(shortlistLen * sizeof(U32));
    for (U32 i = 0; i < shortlistLen; ++i) {
        shortlist[i] = i;
    }

    I32 trueRes[] = {2583, 16370, 14386, 14745, 2584, 37, 12, 14143, 2, 72, 3219, 2479, 19, 23,
        3268, 2, 13166, 12, 8506, 7585, 17379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 2583, 16370, 5, 8506, 61, 7585, 17379, 132, 13166, 2, 12, 24232, 17, 13955,
        813, 523, 468, 1406, 725, 2027, 725, 2027, 725, 27, 12, 6596, 14, 13039, 2, 10, 2471, 5104,
        61, 6584, 1048, 19, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5999, 1567, 55, 1588, 2, 615, 863, 6452, 9,
        9257, 93, 3821, 4579, 5369, 300, 2151, 8386, 2, 1195, 6753, 2, 86, 16370, 13071, 2, 86, 10,
        1075, 5, 153, 6112, 72, 272, 1232, 35, 9869, 1134, 2, 115, 97, 35, 9869, 1134, 2, 115, 97,
        35, 9869, 1134, 2, 44, 33, 12604, 569, 1080, 62, 6, 12, 7185, 23, 171, 3, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1760, 3219, 2479, 19, 23, 3268, 2, 1134, 97, 3771, 14, 23, 4980, 23969,
        15532, 14, 37, 5903, 9, 14745, 2584, 35, 10, 14143, 9300, 2, 72, 17832, 23, 24330, 3, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // load sequences
    const char *inputNames[4] = {"encoder_positions", "encoder_words", "nmt_mask", "shortlist"};
    const char *outputNames[1] = {"decoder_output"};

    std::map<std::string, TensorDesc> inputDescMap;
    inputDescMap[inputNames[0]] = tensor2d(DT_U32, batch, inputLen);
    inputDescMap[inputNames[1]] = tensor2d(DT_U32, batch, inputLen);
    inputDescMap[inputNames[2]] = tensor2d(DT_F32, batch, inputLen);
    inputDescMap[inputNames[3]] = tensor2d(DT_U32, 1, shortlistLen);
    pipelineBase->reready(inputDescMap);

    std::map<std::string, std::shared_ptr<U8>> inputs;
    inputs[inputNames[0]] = std::shared_ptr<U8>((U8 *)positions);
    inputs[inputNames[1]] = std::shared_ptr<U8>((U8 *)input_ids, [](U8 *) {});
    inputs[inputNames[2]] = std::shared_ptr<U8>((U8 *)masks);
    inputs[inputNames[3]] = std::shared_ptr<U8>((U8 *)shortlist);

    pipelineBase->set_input_by_assign(inputs);
    double timeBegin = ut_time_ms();
    for (int i = 0; i < loopTime; ++i) {
        pipelineBase->run();
    }
    double timeEnd = ut_time_ms();
    double totalTime = (timeEnd - timeBegin);

    Tensor decoder_output = pipelineBase->get_tensor_by_name(outputNames[0]);
    U32 outputNum = decoder_output.length();
    for (U32 i = 0; i < outputNum; ++i) {
        if (decoder_output.element(i) != trueRes[i]) {
            UNI_ERROR_LOG("ERROR: Get Wrong Result!\n");
        }
    }
    UNI_CI_LOG("avg_time: %fms/sequence\n", 1.0 * totalTime / loopTime);
    return 0;
}

#endif
