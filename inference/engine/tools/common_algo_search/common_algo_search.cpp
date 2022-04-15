// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ut_util.h"
#include "tensor_computing.h"
#include "algorithm_map.h"
#include "thread_affinity.h"
#include "parse_command.h"

int convolutionCPUFloatAlgorithmSearch(Arch arch, DataType dt, std::string path)
{
    TensorDesc inputDesc, filterDesc;
    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    ActivationParamSpec activationDesc;
    activationDesc.mode = ACTIVATION_RELU;
    activationDesc.value[0] = 0;
    ConvolutionParamSpec convParamSpec;
    convParamSpec.dilatedRate_h = 1;
    convParamSpec.dilatedRate_w = 1;
    U32 in = 1;
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    U32 ic_step, ihw_step, fn_step, ic_max, ihw_max, fn_max;
    std::set<U32> fwh;
    std::set<U32> stride;
    std::string modelName = "";
    std::string deviceName = "";
    AlgorithmMap *algoMap = new AlgorithmMap(arch, modelName, deviceName, dt);
    algoMap->getCommonAlgoMapPara(
        &ic_step, &ihw_step, &fn_step, &ic_max, &ihw_max, &fn_max, &fwh, &stride);
    for (auto sv : stride) {
        for (auto fv : fwh) {
            U32 pl = fv / 2;
            U32 pr = (fv - 1) / 2;
            U32 pt = fv / 2;
            U32 pb = (fv - 1) / 2;
            for (U32 fn = fn_step; fn <= fn_max; fn += fn_step) {
                for (U32 ic = ic_step; ic <= ic_max; ic += ic_step) {
                    for (U32 ih = ihw_step; ih <= ihw_max; ih += ihw_step) {
                        for (U32 iw = ihw_step; iw <= ihw_max; iw += ihw_step) {
                            if (ic % 8 != 0) {
                                inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, ih);
                            } else {
                                inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, ih);
                            }
                            convParamSpec.stride_h = sv;
                            convParamSpec.stride_w = sv;
                            convParamSpec.padding_left = pl;
                            convParamSpec.padding_right = pr;
                            convParamSpec.padding_top = pt;
                            convParamSpec.padding_bottom = pb;
                            filterDesc = tensor4df(dt, DF_NCHW, fn, ic, fv, fv);
                            Tensor inputTensor;
                            Tensor outputTensor;
                            Tensor filterTensor;
                            inputTensor.resize(inputDesc);
                            filterTensor.resize(filterDesc);
                            CHECK_STATUS(convolution_infer_output_size(&inputTensor, filterTensor,
                                convParamSpec, &outputTensor, dt, &archInfo));
                            ConvolutionForwardAlgorithm algorithm = CONVOLUTION_ALGORITHM_NULL;
                            CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor,
                                filterTensor, outputTensor, convParamSpec, policy, &algorithm, dt,
                                activationDesc, &archInfo));
                            algoMap->setCommonAlgoInfoToMap(OT_Conv, dt, ic, ih, iw, fn, fv, fv, sv,
                                sv, (I32 *)(&algorithm), 1);
                        }
                    }
                }
            }
        }
    }
    algoMap->saveAlgorithmMapToFile(path);
    delete algoMap;
    return 0;
}

int main(int argc, char *argv[])
{
    std::string affinityPolicyName = "CPU_AFFINITY_HIGH_PERFORMANCE";
    std::string algorithmMapPath = "./";
    ParseRes parse_res;
    parseCommandLine(argc, argv, &parse_res, "examples");
    if (parse_res.archInfo.second) {
        affinityPolicyName = parse_res.archInfo.first;
    }
    if (parse_res.algoPath.second) {
        algorithmMapPath = parse_res.algoPath.first;
    }
    AffinityPolicy affinityPolicy = thread_affinity_get_policy_by_name(affinityPolicyName.c_str());

    if (affinityPolicyName == "CPU_AFFINITY_HIGH_PERFORMANCE" ||
        affinityPolicyName == "CPU_AFFINITY_LOW_POWER") {
        Arch arch;
#ifndef __APPLE__
        DeviceInfo deviceInfo = get_cpu_info(affinityPolicy);
        set_cpu_dynamic(&deviceInfo, 0);
        arch = deviceInfo.schedule;
#else
        arch = ARM_A76;
#endif
#ifdef _USE_FP16
        convolutionCPUFloatAlgorithmSearch(arch, DT_F16, algorithmMapPath);

#endif
#ifdef _USE_FP32
        convolutionCPUFloatAlgorithmSearch(arch, DT_F32, algorithmMapPath);
#endif
    } else if (affinityPolicyName == "GPU") {
        UNI_ERROR_LOG("Unsupport GPU now\n");
        exit(-1);
    } else {
        UNI_ERROR_LOG("Unknow archInfo %s, please use "
                      "CPU_AFFINITY_HIGH_PERFORMANCE/CPU_AFFINITY_LOW_POWER/GPU\n",
            affinityPolicyName.c_str());
        exit(-1);
    }
    return 0;
}
