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
    UNI_TIME_STOP
    ParseRes parse_res;
    parseCommandLine(argc, argv, &parse_res, "examples");

    set_cpu_num_threads(2);

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

    U32 inputLen = 17;
    U32 shortlistLen = 997;
    U32 input_ids[] = {
        2698, 15, 6009, 2, 264, 5434, 213, 9753, 404, 4, 4589, 3467, 43, 5, 22556, 3, 0};
    U32 positions[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    U32 shortlist[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 110, 112, 116, 118, 120, 121, 123,
        124, 126, 128, 129, 130, 132, 138, 141, 142, 143, 144, 149, 153, 154, 155, 158, 171, 175,
        182, 183, 188, 198, 199, 202, 210, 213, 215, 218, 222, 224, 238, 247, 251, 261, 264, 273,
        275, 279, 283, 286, 294, 298, 299, 301, 302, 317, 318, 319, 323, 329, 336, 339, 343, 351,
        354, 362, 365, 370, 372, 376, 377, 379, 381, 382, 384, 388, 391, 394, 398, 400, 401, 402,
        404, 405, 416, 419, 422, 424, 426, 428, 431, 456, 457, 473, 478, 482, 485, 486, 491, 499,
        504, 507, 510, 513, 528, 529, 530, 546, 563, 570, 579, 586, 592, 594, 601, 605, 606, 607,
        619, 620, 621, 628, 634, 642, 660, 677, 685, 695, 696, 715, 716, 723, 725, 742, 760, 762,
        768, 774, 784, 787, 804, 813, 831, 839, 841, 847, 851, 852, 863, 886, 887, 895, 907, 912,
        913, 924, 927, 934, 940, 943, 965, 970, 972, 974, 1005, 1009, 1020, 1035, 1036, 1040, 1051,
        1054, 1060, 1065, 1066, 1071, 1084, 1090, 1097, 1117, 1132, 1134, 1170, 1180, 1185, 1199,
        1227, 1246, 1253, 1258, 1284, 1291, 1306, 1322, 1333, 1373, 1375, 1379, 1397, 1411, 1419,
        1421, 1439, 1441, 1448, 1470, 1497, 1521, 1530, 1538, 1547, 1569, 1627, 1631, 1641, 1645,
        1655, 1659, 1688, 1709, 1716, 1717, 1726, 1738, 1745, 1749, 1759, 1761, 1765, 1773, 1774,
        1787, 1792, 1796, 1811, 1830, 1846, 1858, 1878, 1889, 1891, 1892, 1919, 1978, 2009, 2018,
        2028, 2038, 2055, 2065, 2093, 2097, 2101, 2127, 2149, 2155, 2161, 2169, 2170, 2200, 2204,
        2241, 2274, 2277, 2308, 2309, 2333, 2336, 2344, 2347, 2370, 2393, 2420, 2464, 2466, 2475,
        2478, 2495, 2497, 2517, 2526, 2533, 2542, 2567, 2578, 2579, 2606, 2608, 2622, 2677, 2680,
        2685, 2687, 2698, 2716, 2751, 2763, 2784, 2798, 2808, 2856, 2892, 2901, 2924, 2982, 2991,
        3019, 3044, 3069, 3115, 3212, 3235, 3239, 3252, 3269, 3326, 3329, 3338, 3374, 3406, 3429,
        3438, 3440, 3467, 3477, 3488, 3546, 3616, 3627, 3644, 3673, 3696, 3697, 3701, 3707, 3733,
        3745, 3759, 3769, 3771, 3799, 3803, 3823, 3826, 3828, 3841, 3858, 3861, 3863, 3865, 3898,
        3926, 3939, 3946, 3948, 3966, 4026, 4028, 4037, 4047, 4072, 4077, 4081, 4091, 4103, 4149,
        4162, 4165, 4183, 4187, 4255, 4303, 4345, 4370, 4436, 4444, 4462, 4464, 4510, 4518, 4534,
        4545, 4554, 4568, 4581, 4582, 4589, 4590, 4601, 4605, 4618, 4627, 4645, 4685, 4686, 4699,
        4718, 4741, 4770, 4776, 4797, 4807, 4834, 4841, 4880, 4883, 4954, 4980, 5029, 5048, 5064,
        5094, 5102, 5129, 5160, 5164, 5169, 5198, 5212, 5220, 5234, 5236, 5259, 5273, 5298, 5362,
        5375, 5387, 5392, 5405, 5434, 5440, 5464, 5471, 5478, 5521, 5523, 5524, 5526, 5546, 5585,
        5606, 5609, 5652, 5668, 5684, 5686, 5716, 5747, 5749, 5750, 5774, 5791, 5801, 5815, 5832,
        5839, 5849, 5882, 5907, 5912, 5927, 5928, 5955, 6005, 6009, 6017, 6018, 6047, 6108, 6130,
        6139, 6166, 6194, 6195, 6205, 6244, 6255, 6257, 6263, 6270, 6282, 6393, 6431, 6442, 6453,
        6495, 6505, 6510, 6519, 6567, 6594, 6598, 6610, 6627, 6668, 6680, 6744, 6788, 6808, 6859,
        6862, 6865, 6876, 6883, 6888, 6926, 6933, 6943, 6956, 6974, 6986, 6995, 7014, 7049, 7164,
        7182, 7291, 7297, 7310, 7326, 7360, 7382, 7401, 7414, 7424, 7442, 7450, 7466, 7539, 7545,
        7548, 7567, 7572, 7594, 7604, 7626, 7644, 7654, 7663, 7671, 7687, 7745, 7748, 7798, 7802,
        7809, 7812, 7845, 7865, 7878, 7894, 7901, 7910, 7931, 7978, 8109, 8161, 8213, 8224, 8236,
        8264, 8323, 8353, 8393, 8415, 8447, 8484, 8503, 8532, 8569, 8605, 8640, 8649, 8663, 8676,
        8771, 9024, 9041, 9073, 9136, 9166, 9206, 9279, 9320, 9440, 9446, 9467, 9537, 9550, 9563,
        9579, 9589, 9670, 9676, 9739, 9741, 9753, 9807, 9813, 9846, 9869, 9901, 9906, 9917, 9923,
        9924, 9954, 9971, 9999, 10011, 10022, 10049, 10053, 10069, 10147, 10200, 10212, 10255,
        10296, 10303, 10357, 10358, 10377, 10449, 10503, 10581, 10609, 10627, 10634, 10641, 10649,
        10655, 10661, 10678, 10682, 10727, 10750, 10782, 10803, 10829, 10836, 10872, 10942, 10991,
        11004, 11011, 11102, 11129, 11289, 11366, 11403, 11427, 11509, 11553, 11566, 11574, 11643,
        11648, 11676, 11681, 11684, 11685, 11692, 11700, 11707, 11828, 11887, 11906, 11944, 12017,
        12027, 12053, 12196, 12232, 12304, 12328, 12329, 12376, 12517, 12607, 12628, 12672, 12750,
        12768, 12833, 12855, 12856, 12881, 12895, 12947, 12961, 12971, 12990, 13011, 13032, 13050,
        13079, 13105, 13117, 13155, 13183, 13184, 13321, 13472, 13505, 13572, 13651, 13719, 13722,
        13779, 13848, 13862, 13993, 14011, 14139, 14149, 14259, 14415, 14420, 14485, 14511, 14561,
        14578, 14583, 14658, 14689, 14728, 14739, 14746, 14810, 14858, 14915, 15004, 15007, 15122,
        15164, 15216, 15350, 15433, 15483, 15487, 15501, 15541, 15647, 15649, 15680, 15684, 15717,
        15783, 15793, 15804, 15824, 15901, 15972, 15980, 16102, 16130, 16166, 16187, 16251, 16262,
        16349, 16385, 16407, 16413, 16423, 16434, 16474, 16550, 16565, 16663, 16687, 16693, 16775,
        16807, 16809, 16850, 16881, 16907, 16933, 16937, 16970, 17007, 17014, 17036, 17117, 17140,
        17275, 17333, 17356, 17364, 17375, 17391, 17472, 17704, 17730, 17759, 17761, 17773, 17802,
        17823, 17946, 18197, 18224, 18440, 18470, 18533, 18557, 18562, 18633, 18733, 18818, 18896,
        18901, 19020, 19044, 19058, 19064, 19119, 19155, 19238, 19239, 19247, 19282, 19289, 19361,
        19566, 19740, 19775, 19837, 19998, 20013, 20039, 20078, 20120, 20172, 20200, 20251, 20356,
        20567, 20574, 20654, 20674, 20693, 20787, 20902, 20918, 21173, 21186, 21383, 21414, 21502,
        21621, 21843, 21878, 21943, 22002, 22013, 22049, 22076, 22104, 22134, 22196, 22286, 22302,
        22547, 22556, 22591, 22689, 22704, 22791, 22817, 22837, 22936, 23022, 23108, 23383, 23427,
        23497, 23514, 23544, 23645, 23668, 23713, 23775, 23871, 23900, 23935, 23947, 24103, 24442,
        24638, 24740, 24813, 24829};

    U32 nonZeroNum = 15;
    I32 trueRes[] = {2698, 15, 6009, 2, 264, 11, 1, 53, 213, 9753, 404, 4589, 3467, 43, 5};

    // load sequences
    const char *inputNames[3] = {"encoder_positions", "encoder_words", "shortlist"};
    const char *outputNames[1] = {"decoder_output"};

    std::map<std::string, TensorDesc> inputDescMap;
    inputDescMap[inputNames[0]] = tensor2d(DT_U32, 1, inputLen);
    inputDescMap[inputNames[1]] = tensor2d(DT_U32, 1, inputLen);
    inputDescMap[inputNames[2]] = tensor2d(DT_U32, 1, shortlistLen);
    pipelineBase->reready(inputDescMap);

    std::map<std::string, std::shared_ptr<U8>> inputs;
    inputs[inputNames[0]] = std::shared_ptr<U8>((U8 *)positions, [](U8 *) {});
    inputs[inputNames[1]] = std::shared_ptr<U8>((U8 *)input_ids, [](U8 *) {});
    inputs[inputNames[2]] = std::shared_ptr<U8>((U8 *)shortlist, [](U8 *) {});

    UNI_TIME_START
    pipelineBase->set_input_by_assign(inputs);
    double timeBegin = ut_time_ms();
    for (int i = 0; i < loopTime; ++i) {
        pipelineBase->run();
    }
    double timeEnd = ut_time_ms();
    double totalTime = (timeEnd - timeBegin);

    Tensor decoder_output = pipelineBase->get_tensor_by_name(outputNames[0]);
    U32 outputNum = decoder_output.length();
    U32 i = 0;
    bool state = true;
    for (; i < nonZeroNum; ++i) {
        if (decoder_output.element(i) != trueRes[i]) {
            UNI_ERROR_LOG("ERROR: Get Wrong Result!\n");
            state = false;
        }
    }
    for (; i < outputNum; ++i) {
        if (decoder_output.element(i) != 0) {
            UNI_ERROR_LOG("ERROR: Get Wrong Result!\n");
            state = false;
        }
    }
    if (state) {
        std::cout << "Verify en-cs Result Success!" << std::endl;
    } else {
        UNI_ERROR_LOG("Verify en-cs Result Fail!");
    }
    UNI_TIME_STATISTICS
    UNI_CI_LOG("avg_time: %fms/sequence\n", 1.0 * totalTime / loopTime);
}

#endif
