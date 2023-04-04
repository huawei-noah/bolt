// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CHANNELPADDINGOPTIMIZER
#define _H_CHANNELPADDINGOPTIMIZER

#include "OPOptimizer.hpp"

class ChannelPaddingOptimizer : public OPOptimizer {
    void insertChannelResizeOperator(ModelSpec *spec,
        int index,
        const char *name,
        char *input_name,
        const char *symmetric,
        int group,
        int numChannelsBefore,
        int numChannelAfter)
    {
        OperatorSpec channelResizeOperator = mt_create_operator(name, OT_ChannelResize, 1, 1);
        if (symmetric == nullptr || symmetric == NULL) {
            UNI_MEMSET(channelResizeOperator.ps.channel_resize_spec.symmetric, 0, NAME_LEN);
        } else {
            str_copy(channelResizeOperator.ps.channel_resize_spec.symmetric, symmetric,
                strlen(symmetric));
        }
        channelResizeOperator.ps.channel_resize_spec.group = group;
        channelResizeOperator.ps.channel_resize_spec.channel_before = numChannelsBefore;
        channelResizeOperator.ps.channel_resize_spec.channel_after = numChannelAfter;
        // channel cut
        if (numChannelAfter < numChannelsBefore) {
            str_copy(channelResizeOperator.output_tensors_name[0], input_name, strlen(input_name));
            str_copy(channelResizeOperator.input_tensors_name[0], name, strlen(name));
            str_copy(input_name, name, strlen(name));
        } else {
            str_copy(channelResizeOperator.input_tensors_name[0], input_name, strlen(input_name));
            str_copy(channelResizeOperator.output_tensors_name[0], name, strlen(name));
            str_copy(input_name, name, strlen(name));
        }
        mt_insert_operator(spec, index, channelResizeOperator);
    }

    bool isBlankChannelResizeOperator(OperatorSpec currentOperator)
    {
        return currentOperator.ps.channel_resize_spec.group == 0;
    }

    bool canMergeChannelCutSpan(OperatorSpec channelCutOperator, OperatorSpec channelSpanOperator)
    {
        CHECK_REQUIREMENT(channelSpanOperator.type == OT_ChannelResize &&
            channelCutOperator.type == OT_ChannelResize);
        if (isBlankChannelResizeOperator(channelSpanOperator)) {
            return true;
        }
        if (channelSpanOperator.ps.channel_resize_spec.group ==
                channelCutOperator.ps.channel_resize_spec.group &&
            channelSpanOperator.ps.channel_resize_spec.channel_before ==
                channelCutOperator.ps.channel_resize_spec.channel_after &&
            channelSpanOperator.ps.channel_resize_spec.channel_after ==
                channelCutOperator.ps.channel_resize_spec.channel_before) {
            return true;
        }
        return false;
    }

    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        int channelAlign = 8;
        std::string channelResizeNamePrefix = "ChannelResize_";
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Conv &&
                spec->ops[i].ps.conv_spec.convolution_type == CONVOLUTION_DEPTHWISE) {
                OperatorSpec currentOperator = spec->ops[i];
                U32 numKernels = currentOperator.ps.conv_spec.num_outputs;
                U32 paddingBase = channelAlign;
                if (numKernels % paddingBase == 0) {
                    continue;
                }
                U32 numKernelsNew =
                    (numKernels / paddingBase + (numKernels % paddingBase != 0)) * paddingBase;
                spec->ops[i].ps.conv_spec.num_outputs = numKernelsNew;
                int weightIndex = searchWeightIndex(spec, currentOperator.name);
                CHECK_REQUIREMENT(weightIndex >= 0);
                U32 weightSize = spec->ws[weightIndex].bytes_of_weight;
                U32 weightSizeNew =
                    spec->ws[weightIndex].bytes_of_weight / numKernels * numKernelsNew;
                U8 *weight = spec->ws[weightIndex].weight;
                spec->ws[weightIndex].bytes_of_weight = weightSizeNew;
                spec->ws[weightIndex].weight =
                    (U8 *)mt_malloc(spec->ws[weightIndex].bytes_of_weight);
                UNI_MEMCPY(spec->ws[weightIndex].weight, weight, weightSize);
                UNI_MEMSET(spec->ws[weightIndex].weight + weightSize, 0, weightSizeNew - weightSize);
                mt_free(weight, spec);
                U8 *vec = spec->ws[weightIndex].vec;
                if (vec != nullptr) {
                    U32 vecSize = spec->ws[weightIndex].bytes_of_vec;
                    U32 vecSizeNew = spec->ws[weightIndex].bytes_of_vec / numKernels * numKernelsNew;
                    spec->ws[weightIndex].bytes_of_vec = vecSizeNew;
                    spec->ws[weightIndex].vec = (U8 *)mt_malloc(spec->ws[weightIndex].bytes_of_vec);
                    UNI_MEMCPY((U8 *)(spec->ws[weightIndex].vec), vec, vecSize);
                    UNI_MEMSET((U8 *)(spec->ws[weightIndex].vec + vecSize), 0, vecSizeNew - vecSize);
                    mt_free(vec, spec);
                }
                std::string channelResizeName1 =
                    allocName(channelResizeNamePrefix + std::to_string(i));
                std::string channelResizeName2 =
                    allocName(channelResizeNamePrefix + std::to_string(i + 2));
                insertChannelResizeOperator(spec, i, channelResizeName1.c_str(),
                    currentOperator.input_tensors_name[0], channelResizeName2.c_str(), 1,
                    numKernels, numKernelsNew);
                insertChannelResizeOperator(spec, i + 2, channelResizeName2.c_str(),
                    currentOperator.output_tensors_name[0], nullptr, 1, numKernelsNew, numKernels);
                i += 2;
                hasOptimized = true;
                continue;
            }
            if ((spec->ops[i].type == OT_Conv &&
                    (spec->ops[i].ps.conv_spec.convolution_type == CONVOLUTION_POINTWISE ||
                        spec->ops[i].ps.conv_spec.convolution_type == CONVOLUTION_DILATION)) ||
                (spec->ops[i].type == OT_Deconvolution &&
                    spec->ops[i].ps.conv_spec.convolution_type == CONVOLUTION_DECONVOLUTION)) {
                OperatorSpec currentOperator = spec->ops[i];
                U32 groups = currentOperator.ps.conv_spec.group;
                U32 paddingBase = channelAlign * groups;
                U32 numKernels = currentOperator.ps.conv_spec.num_outputs;
                U32 numKernelsNew =
                    (numKernels / paddingBase + (numKernels % paddingBase != 0)) * paddingBase;
                int weightIndex = searchWeightIndex(spec, currentOperator.name);
                CHECK_REQUIREMENT(weightIndex >= 0);
                U32 weightSize = spec->ws[weightIndex].bytes_of_weight;
                U32 inputChannels = weightSize / bytesOf(spec->ws[weightIndex].mdt) / numKernels /
                    spec->ops[i].ps.conv_spec.kernel_t / spec->ops[i].ps.conv_spec.kernel_h /
                    spec->ops[i].ps.conv_spec.kernel_w;
                U32 inputChannelsNew = inputChannels;
                if (spec->ops[i].type == OT_Deconvolution) {
                    inputChannelsNew =
                        (inputChannels / paddingBase + (inputChannels % paddingBase != 0)) *
                        paddingBase;
                }
                if (inputChannelsNew == inputChannels && numKernels == numKernelsNew) {
                    continue;
                }
                spec->ops[i].ps.conv_spec.num_outputs = numKernelsNew;
                U32 tileSize = weightSize / numKernels / inputChannels;
                U32 weightSizeNew = tileSize * numKernelsNew * inputChannelsNew;
                U8 *weight = spec->ws[weightIndex].weight;
                spec->ws[weightIndex].bytes_of_weight = weightSizeNew;
                spec->ws[weightIndex].weight =
                    (U8 *)mt_malloc(spec->ws[weightIndex].bytes_of_weight);
                UNI_MEMSET(spec->ws[weightIndex].weight, 0, weightSizeNew);
                U32 ocGroupSize = numKernels / groups;
                U32 ocGroupSizeNew = numKernelsNew / groups;

                if (spec->ops[i].type == OT_Deconvolution) {  //CNHW
                    U32 icGroupSize = inputChannels / groups;
                    U32 icGroupSizeNew = inputChannelsNew / groups;
                    for (U32 cg = 0; cg < groups; ++cg) {
                        for (U32 ic = 0; ic < icGroupSize; ++ic) {
                            for (U32 og = 0; og < groups; ++og) {
                                U32 index =
                                    ((cg * icGroupSize + ic) * numKernels + og * ocGroupSize) *
                                    tileSize;
                                U32 indexNew = ((cg * icGroupSizeNew + ic) * numKernelsNew +
                                                   og * ocGroupSizeNew) *
                                    tileSize;
                                UNI_MEMCPY((U8 *)(spec->ws[weightIndex].weight) + indexNew,
                                    weight + index, tileSize * ocGroupSize);
                            }
                        }
                    }
                } else {  // NCHW
                    for (U32 og = 0; og < groups; ++og) {
                        for (U32 oc = 0; oc < ocGroupSize; ++oc) {
                            for (U32 c = 0; c < inputChannels; ++c) {
                                U32 index = ((og * ocGroupSize + oc) * inputChannels + c) * tileSize;
                                U32 indexNew =
                                    ((og * ocGroupSizeNew + oc) * inputChannels + c) * tileSize;
                                UNI_MEMCPY((U8 *)(spec->ws[weightIndex].weight) + indexNew,
                                    weight + index, tileSize);
                            }
                        }
                    }
                }
                mt_free(weight, spec);
                U8 *vec = spec->ws[weightIndex].vec;
                if (vec != nullptr && numKernels != numKernelsNew) {
                    U32 vecSize = spec->ws[weightIndex].bytes_of_vec;
                    U32 vecSizeNew = spec->ws[weightIndex].bytes_of_vec / numKernels * numKernelsNew;
                    spec->ws[weightIndex].bytes_of_vec = vecSizeNew;
                    spec->ws[weightIndex].vec = (U8 *)mt_malloc(spec->ws[weightIndex].bytes_of_vec);
                    U32 tile = vecSize / groups;
                    U32 tileNew = vecSizeNew / groups;
                    for (U32 g = 0; g < groups; g++) {
                        UNI_MEMCPY(
                            (U8 *)(spec->ws[weightIndex].vec) + g * tileNew, vec + g * tile, tile);
                        UNI_MEMSET((U8 *)(spec->ws[weightIndex].vec) + g * tileNew + tile, 0,
                            tileNew - tile);
                    }
                    mt_free(vec, spec);
                }
                int channelResizeIndex1 = i;
                int channelResizeIndex2 = i + 2;
                std::string channelResizeName1 =
                    allocName(channelResizeNamePrefix + std::to_string(channelResizeIndex1));
                std::string channelResizeName2 =
                    allocName(channelResizeNamePrefix + std::to_string(channelResizeIndex2));
                if (inputChannels != inputChannelsNew) {
                    const char *symmetric = channelResizeName2.c_str();
                    if (numKernels == numKernelsNew) {
                        symmetric = nullptr;
                    }
                    insertChannelResizeOperator(spec, channelResizeIndex1,
                        channelResizeName1.c_str(), currentOperator.input_tensors_name[0],
                        symmetric, groups, inputChannels, inputChannelsNew);
                    i += 1;
                }
                if (numKernels != numKernelsNew) {
                    if (inputChannels == inputChannelsNew) {
                        channelResizeIndex2 = i + 1;
                        channelResizeName2 = allocName(
                            channelResizeNamePrefix + std::to_string(channelResizeIndex2));
                    }
                    insertChannelResizeOperator(spec, channelResizeIndex2,
                        channelResizeName2.c_str(), currentOperator.output_tensors_name[0], nullptr,
                        groups, numKernelsNew, numKernels);
                    i += 1;
                }
                hasOptimized = true;
                continue;
            }
        }

        for (int i = 0; i < spec->num_operator_specs; i++) {
            // cut operator
            if (spec->ops[i].type == OT_ChannelResize) {
                // span operator
                std::vector<std::pair<int, int>> output = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                for (U32 j = 0; j < output.size(); j++) {
                    int nextIndex = output[j].first;
                    if (spec->ops[nextIndex].type == OT_ChannelResize &&
                        canMergeChannelCutSpan(spec->ops[i], spec->ops[nextIndex])) {
                        if (output.size() == 1) {
                            setOperatorInvalid(spec, i, true);
                        } else {
                            str_copy(spec->ops[nextIndex].input_tensors_name[0],
                                spec->ops[i].input_tensors_name[0], NAME_LEN);
                        }
                        setOperatorInvalid(spec, nextIndex, true);
                        if (nextIndex + 2 < spec->num_operator_specs &&
                            isBlankChannelResizeOperator(spec->ops[nextIndex + 2])) {
                            spec->ops[nextIndex + 2].ps = spec->ops[i].ps;
                        }
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
