// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <fstream>
#include <iostream>
#include <atomic>
#include "task.h"
#include "flow.h"
#include "audio_feature.h"

DataType inferencePrecision = DT_F32;
const int N_FILTERS = 128;

// prediction&joint&pinyin2hanzi
const int START_TOKEN = 0;
const int BLANK_TOKEN = 1600;

// pinyin2hanzi
const int PINYIN_FEATURE_GAP = 2;
const int PINYIN_BUFFER_SIZE = 32;
const int PINYIN_BUFFER_VALID_SIZE = 16;
std::shared_ptr<float> pinyinEmbeddingDict;
std::atomic<int> pinyinEmbeddingFlag(0);

EE encoderInferOutputSize(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    TensorDesc inputDesc = inputs["sounds"]->get_desc();
    TensorDesc desc = inputs["encoder_block0_trunk0_layer0_mem"]->get_desc();
    desc.dims[2] = UNI_MAX((int)desc.dims[2] + 1, 2);
    outputs["encoder_block0_conv0_neg_slice"]->resize(desc);
    outputs["encoder_block0_conv1_neg_slice"]->resize(
        inputs["encoder_block0_trunk0_layer1_mem"]->get_desc());

    int block1[2] = {5, 7};
    for (int i = 0; i < 2; i++) {
        std::string inputPrefix =
            std::string("encoder_block1_trunk1_layer") + std::to_string(i) + std::string("_");
        std::string outputPrefix =
            std::string("encoder_block1_transformer_layer") + std::to_string(i) + std::string("_");
        TensorDesc desc = inputs[inputPrefix + "kmem"]->get_desc();
        desc.dims[2] = UNI_MAX((int)desc.dims[2] + block1[i], block1[i]);
        outputs[outputPrefix + "k_neg_slice"]->resize(desc);
        outputs[outputPrefix + "v_neg_slice"]->resize(desc);
    }

    desc = inputs["encoder_block2_trunk0_layer0_mem"]->get_desc();
    desc.dims[1] = UNI_MAX((int)desc.dims[1] + 1, 2);
    outputs["encoder_block2_conv0_neg_slice"]->resize(desc);
    outputs["encoder_block2_conv1_neg_slice"]->resize(
        inputs["encoder_block2_trunk0_layer1_mem"]->get_desc());
    int block2[2] = {7, 9};
    for (int i = 0; i < 2; i++) {
        std::string inputPrefix =
            std::string("encoder_block2_trunk1_layer") + std::to_string(i) + std::string("_");
        std::string outputPrefix =
            std::string("encoder_block2_transformer_layer") + std::to_string(i) + std::string("_");
        TensorDesc desc = inputs[inputPrefix + "kmem"]->get_desc();
        int adder = 2;
        if (inputDesc.dims[1] == 15) {
            adder = 3;
        } else {
            if (inputDesc.dims[1] != 8) {
                UNI_ERROR_LOG("unmatched encoder input\n");
            }
        }
        desc.dims[2] = UNI_MAX((int)desc.dims[2] + adder, block2[i]);
        outputs[outputPrefix + "k_neg_slice"]->resize(desc);
        outputs[outputPrefix + "v_neg_slice"]->resize(desc);
    }

    desc = inputs["encoder_block3_trunk0_layer0_mem"]->get_desc();
    desc.dims[1] = UNI_MAX((int)desc.dims[1] + 1, 2);
    outputs["encoder_block3_conv0_neg_slice"]->resize(desc);
    outputs["encoder_block3_conv1_neg_slice"]->resize(
        inputs["encoder_block3_trunk0_layer1_mem"]->get_desc());
    int block3[4] = {9, 15, 23, 31};
    for (int i = 0; i < 4; i++) {
        std::string inputPrefix =
            std::string("encoder_block3_trunk1_layer") + std::to_string(i) + std::string("_");
        std::string outputPrefix =
            std::string("encoder_block3_transformer_layer") + std::to_string(i) + std::string("_");
        TensorDesc desc = inputs[inputPrefix + "kmem"]->get_desc();
        desc.dims[2] = UNI_MAX((int)desc.dims[2] + 1, block3[i]);
        outputs[outputPrefix + "k_neg_slice"]->resize(desc);
        outputs[outputPrefix + "v_neg_slice"]->resize(desc);
    }
    outputs["encoder_block3_transformer_ln"]->resize(
        tensor2df(inferencePrecision, DF_NORMAL, 1, 512));
    return SUCCESS;
}

EE predictionInferOutputSize(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    int block3[4] = {3, 5, 7, 9};
    for (int i = 0; i < 4; i++) {
        std::string inputPrefix =
            std::string("prediction_net_layer") + std::to_string(i) + std::string("_");
        std::string outputPrefix =
            std::string("prediction_net_layer") + std::to_string(i) + std::string("_");
        TensorDesc desc = inputs[inputPrefix + "kmem"]->get_desc();
        desc.dims[2] = UNI_MAX((int)desc.dims[2] + 1, block3[i]);
        outputs[outputPrefix + "k_neg_slice"]->resize(desc);
        outputs[outputPrefix + "v_neg_slice"]->resize(desc);
    }
    outputs["prediction_net_ln"]->resize(tensor2df(inferencePrecision, DF_NORMAL, 1, 512));
    return SUCCESS;
}

EE jointInferOutputSize(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    // outputs["joint_output_fc"]->resize(tensor2df(inferencePrecision, DF_NORMAL, 1, 512));
    outputs["output_argmax"]->resize(tensor2df(DT_I32, DF_NORMAL, 1, 1));
    return SUCCESS;
}

EE pinyin2hanziInferOutputSize(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    TensorDesc desc = inputs["pinyin"]->get_desc();
    outputs["hanzi_squeeze/Squeeze"]->resize(
        tensor4df(inferencePrecision, DF_NCHW, 1, 1, desc.dims[0], 7160));
    return SUCCESS;
}

EE encoderPreProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    int featureLength = N_FILTERS;
    // inputs and outputs can not be same one
    CHECK_REQUIREMENT(inputs.size() > 0);
    std::vector<float> weightA = {0.26793470448235757, 0.2597546401553133, 0.25070439183132637,
        0.2389518634030468, 0.22591939536296402, 0.21842706422127695, 0.21073101672676822,
        0.19888634668966934, 0.1934352819534865, 0.19483272371655574, 0.19307169092034548,
        0.19794880602465662, 0.2041545140444457, 0.20548612384306975, 0.205089112033574,
        0.202463874511741, 0.1997057297551323, 0.1986376615816107, 0.1953351397506247,
        0.19526630343057141, 0.19707734328352133, 0.19871668436383344, 0.19880258511761903,
        0.20143541652121727, 0.2044134862423108, 0.20602641560137125, 0.20564694818486318,
        0.206515308314549, 0.2092981906166021, 0.2105148453821694, 0.209482433282912,
        0.21072670095339943, 0.21295487096308688, 0.21402032655941866, 0.21254455731621794,
        0.21365817460879144, 0.2163171444197802, 0.21766703064503207, 0.21640375119276742,
        0.2177893882181534, 0.2205046640925341, 0.2218610679573307, 0.22053006469571076,
        0.22162170408445966, 0.22370872632630542, 0.22537803061334274, 0.22641169891592502,
        0.2274135200959736, 0.22817822886370503, 0.22850555770692876, 0.22849091616908523,
        0.22942646398018746, 0.23089530924664364, 0.23176498740499615, 0.23372326568964216,
        0.23547995759926693, 0.2364584692820128, 0.23713210245263003, 0.2375549912435519,
        0.23761757113350296, 0.23757638746581106, 0.23820814260735781, 0.2385523824231173,
        0.23896144410382456, 0.2397607819892432, 0.24065938255474512, 0.2416691468977067,
        0.24337672078468509, 0.24427940599421233, 0.24517506765424793, 0.24579829824437913,
        0.24723941129617125, 0.24809058963717726, 0.24874810693293706, 0.248877475370626,
        0.24951549731479883, 0.24955122418541695, 0.2492060337981675, 0.24902471798206796,
        0.24888344336656584, 0.24846182447195098, 0.24729274718749017, 0.24639018404388816,
        0.24659313647419556, 0.24630866444966484, 0.24585278398389177, 0.24605167118751672,
        0.24594061893719316, 0.24532106768133538, 0.24572437083432735, 0.2459548905112401,
        0.245982906631063, 0.24652363950502573, 0.24715790835692908, 0.2478608527450776,
        0.24889337178480928, 0.249329751248172, 0.24960285555075376, 0.24955584458875266,
        0.2497572027892517, 0.2499798759413889, 0.2500960262323433, 0.2506400682242264,
        0.2515477086314016, 0.25259227168784903, 0.25364113255322157, 0.25537851424540586,
        0.2573300627421209, 0.25956427589759357, 0.26117713995761727, 0.2624523374880242,
        0.2632993514075515, 0.26413640430134505, 0.26511896710476746, 0.2662951418810798,
        0.26744233631929915, 0.267688136864862, 0.2672668616086788, 0.26649503147446485,
        0.26594129076005935, 0.2659199727680806, 0.2664476518237045, 0.26695480256723025,
        0.2678133595844467, 0.2701192220836497, 0.2742489539853769, 0.2798973923783803,
        0.28540062392560295};
    std::vector<float> weightB = {4.594726366770656, 4.192752172632116, 3.9776274929119557,
        3.4349833759246713, 3.0983192175590126, 2.8131751675954018, 2.674216353771496,
        2.299024401714484, 2.2277405730898843, 2.2079989172157086, 2.2080042633425534,
        2.239013527979191, 2.41471012643739, 2.405628743225133, 2.45394225056771, 2.3372751727216574,
        2.3356523900751234, 2.2857494554648192, 2.263597932542921, 2.199953784963237,
        2.283013730372439, 2.287507759169855, 2.3248724084010197, 2.3234718339153364,
        2.428010836779634, 2.4391312085381363, 2.4676774757702, 2.4445873870383834,
        2.5379614937156854, 2.541529720288643, 2.552965909269937, 2.528893119611279,
        2.609828446143808, 2.611520901760278, 2.6113588465301225, 2.5879040353367735,
        2.670180890126309, 2.6768002097714785, 2.6745482022603047, 2.6589252525406937,
        2.7405675184409484, 2.748250039256346, 2.7504889136399346, 2.7279897692691324,
        2.803509804647416, 2.8033767975633253, 2.81782662029014, 2.8398580132615985,
        2.8634585052804473, 2.8850252018322435, 2.8939588401492355, 2.9149064619044824,
        2.938446538597044, 2.9491789310074474, 2.9655894539521057, 2.9814448232043804,
        2.9946988873469187, 2.9974272291551625, 2.9982878146018908, 2.997330908879054,
        2.9987101107447867, 2.9833493242668405, 2.9875125168844545, 3.0194390288802575,
        3.028980829234581, 3.0057895811449447, 3.076450198087296, 3.0683058012421935,
        3.0938844769593064, 3.11508333263089, 3.121912904965018, 3.146879175832384,
        3.1768447540457245, 3.1598400327144147, 3.190448649847769, 3.1933782870894385,
        3.1789337132666655, 3.1801368920926776, 3.1702021059419705, 3.1585067337253734,
        3.145159095452153, 3.124279154413975, 3.1068527554445096, 3.103454244479969,
        3.096145034068362, 3.0888735929867055, 3.0728735019732527, 3.0772210570154477,
        3.0684300226295047, 3.0504857878230385, 3.068488307579292, 3.051638660693075,
        3.0726374420353735, 3.0707974307243466, 3.088892965875781, 3.103242655729246,
        3.1090877750810226, 3.112699742574199, 3.111884782449412, 3.1145576667173303,
        3.1185679471418215, 3.1242895827009405, 3.136642993753398, 3.15245492583083,
        3.185308230069337, 3.2015540228767803, 3.245292124114324, 3.2826235672398743,
        3.3220448193935534, 3.3566443133338755, 3.3843542201410166, 3.406417064746228,
        3.4294187840241075, 3.458963279130731, 3.4864911772857177, 3.508984664352243,
        3.525467921720016, 3.5317980631290027, 3.5339991083767575, 3.5397785467806564,
        3.5511168000118016, 3.5702997212991785, 3.6000097146634724, 3.6546755683682086,
        3.763185000352641, 3.9092252627215855, 4.07891493530088, 4.22557473399065};
    TensorDesc inputDesc = inputs["sounds"]->get_desc();
    outputs = inputs;
    outputs["sounds"] = std::shared_ptr<Tensor>(new Tensor());
    outputs["sounds"]->resize(inputDesc);
    outputs["sounds"]->alloc();
    int num = tensorNumElements(inputDesc);
    int loops = num / featureLength;
    CHECK_REQUIREMENT(loops * featureLength == num);
    switch (inferencePrecision) {
        case DT_F32: {
            F32 *inPtr = (F32 *)((CpuMemory *)(inputs["sounds"]->get_memory()))->get_ptr();
            F32 *outPtr = (F32 *)((CpuMemory *)(outputs["sounds"]->get_memory()))->get_ptr();
            for (int i = 0, index = 0; i < loops; i++) {
                for (int j = 0; j < featureLength; j++, index++) {
                    outPtr[index] = weightA[j] * inPtr[index] + weightB[j];
                }
            }
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            F16 *inPtr = (F16 *)((CpuMemory *)(inputs["sounds"]->get_memory()))->get_ptr();
            F16 *outPtr = (F16 *)((CpuMemory *)(outputs["sounds"]->get_memory()))->get_ptr();
            for (int i = 0, index = 0; i < loops; i++) {
                for (int j = 0; j < featureLength; j++, index++) {
                    outPtr[index] = weightA[j] * inPtr[index] + weightB[j];
                }
            }
            break;
        }
#endif
        default:
            UNI_ERROR_LOG("unsupported precision type in asr encoder preprocess function\n");
            break;
    }
    return SUCCESS;
}

void loadBinary(const std::string fileName, char *data, size_t size)
{
    std::ifstream ifs(fileName, std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
    if (!ifs.good()) {
        UNI_ERROR_LOG("load binary data from %s failed\n", fileName.c_str());
    }
    size_t length = ifs.tellg();
    ifs.seekg(0, std::ifstream::beg);
    ifs.read(data, UNI_MIN(length, size));
    if (length < size) {
        memset(data + length, 0, size - length);
    }
    ifs.close();
}

EE pinyin2hanziPreProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    int embeddingSize = std::stoi(parameter[3]);
    if (!atomic_exchange(&pinyinEmbeddingFlag, 1)) {
        std::string embeddingFile = parameter[1];
        int classes = std::stoi(parameter[2]);
        size_t size = sizeof(float) * classes * embeddingSize;
        pinyinEmbeddingDict = std::shared_ptr<float>(reinterpret_cast<float *>(operator new(size)));
        loadBinary(embeddingFile, reinterpret_cast<char *>(pinyinEmbeddingDict.get()), size);
    }
    TensorDesc inputDesc = inputs["pinyin"]->get_desc();
    int batch = inputDesc.dims[inputDesc.nDims - 1];
    int inputSize = tensorNumElements(inputDesc);
    int inputSizePerBatch = inputSize / batch;
    unsigned int *inputPtr =
        (unsigned int *)((CpuMemory *)(inputs["pinyin"]->get_memory()))->get_ptr();
    std::string name = "lm_in_deploy";
    outputs[name] = std::shared_ptr<Tensor>(new Tensor());
    outputs[name]->resize(
        tensor4df(inferencePrecision, DF_NCHW, 1, embeddingSize, 1, inputDesc.dims[0]));
    outputs[name]->alloc();
    float *pinyinEmbeddingDictPtr = pinyinEmbeddingDict.get();
    switch (inferencePrecision) {
        case DT_F32: {
            F32 *outputPtr = (F32 *)((CpuMemory *)(outputs[name]->get_memory()))->get_ptr();
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < inputSizePerBatch; j++) {
                    int element = inputPtr[i * inputSizePerBatch + j];
                    for (int k = 0; k < embeddingSize; k++) {
                        outputPtr[(i * embeddingSize + k) * inputSizePerBatch + j] =
                            pinyinEmbeddingDictPtr[element * embeddingSize + k];
                    }
                }
            }
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            F16 *outputPtr = (F16 *)((CpuMemory *)(outputs[name]->get_memory()))->get_ptr();
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < inputSizePerBatch; j++) {
                    int element = inputPtr[i * inputSizePerBatch + j];
                    for (int k = 0; k < embeddingSize; k++) {
                        outputPtr[(i * embeddingSize + k) * inputSizePerBatch + j] =
                            pinyinEmbeddingDictPtr[element * embeddingSize + k];
                    }
                }
            }
            break;
        }
#endif
        default:
            UNI_ERROR_LOG("unsupported precision type in asr pinyin2hanzi preprocess function\n");
            break;
    }
    return SUCCESS;
}

std::map<std::string, std::shared_ptr<Tensor>> getEncoderInputOutput(
    std::vector<std::vector<std::vector<float>>> feature,
    int frameId,
    int frameLength,
    std::map<std::string, std::shared_ptr<Tensor>> cache)
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    int frameOffset = ((frameId > 0) ? 15 : 0) + ((frameId > 0) ? (frameId - 1) : 0) * 8;
    if (frameOffset + frameLength > static_cast<int>(feature[0].size())) {
        return tensors;
    }
    int featureLength = N_FILTERS;
    tensors["sounds"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["sounds"]->resize(tensor3df(inferencePrecision, DF_NCHW, 1, frameLength, featureLength));
    tensors["sounds"]->alloc();
    switch (inferencePrecision) {
        case DT_F32: {
            F32 *ptr = (F32 *)((CpuMemory *)(tensors["sounds"]->get_memory()))->get_ptr();
            for (int i = 0; i < frameLength; i++) {
                memcpy(ptr + i * featureLength, feature[0][i + frameOffset].data(),
                    featureLength * sizeof(float));
            }
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            F16 *ptr = (F16 *)((CpuMemory *)(tensors["sounds"]->get_memory()))->get_ptr();
            for (int i = 0; i < frameLength; i++) {
                for (int j = 0; j < featureLength; j++) {
                    ptr[i * featureLength + j] = feature[0][i + frameOffset][j];
                }
            }
            break;
        }
#endif
        default:
            UNI_ERROR_LOG("not support inference precision to get encoder input\n");
    }
    std::vector<std::string> outputName = {"encoder_block3_transformer_ln",
        "encoder_block0_conv0_neg_slice", "encoder_block0_conv1_neg_slice",
        "encoder_block1_transformer_layer0_k_neg_slice",
        "encoder_block1_transformer_layer0_v_neg_slice",
        "encoder_block1_transformer_layer1_k_neg_slice",
        "encoder_block1_transformer_layer1_v_neg_slice", "encoder_block2_conv0_neg_slice",
        "encoder_block2_conv1_neg_slice", "encoder_block2_transformer_layer0_k_neg_slice",
        "encoder_block2_transformer_layer0_v_neg_slice",
        "encoder_block2_transformer_layer1_k_neg_slice",
        "encoder_block2_transformer_layer1_v_neg_slice", "encoder_block3_conv0_neg_slice",
        "encoder_block3_conv1_neg_slice", "encoder_block3_transformer_layer0_k_neg_slice",
        "encoder_block3_transformer_layer0_v_neg_slice",
        "encoder_block3_transformer_layer1_k_neg_slice",
        "encoder_block3_transformer_layer1_v_neg_slice",
        "encoder_block3_transformer_layer2_k_neg_slice",
        "encoder_block3_transformer_layer2_v_neg_slice",
        "encoder_block3_transformer_layer3_k_neg_slice",
        "encoder_block3_transformer_layer3_v_neg_slice"};
    for (unsigned int i = 0; i < outputName.size(); i++) {
        tensors[outputName[i]] = std::shared_ptr<Tensor>(new Tensor());
    }

    if (cache.size() > 0 && frameLength == 8) {
        tensors["encoder_block0_trunk0_layer0_mem"] = cache["encoder_block0_conv0_neg_slice"];
        tensors["encoder_block0_trunk0_layer1_mem"] = cache["encoder_block0_conv1_neg_slice"];
        for (int i = 0; i < 2; i++) {
            std::string inputPrefix =
                std::string("encoder_block1_trunk1_layer") + std::to_string(i) + std::string("_");
            std::string outputPrefix = std::string("encoder_block1_transformer_layer") +
                std::to_string(i) + std::string("_");
            tensors[inputPrefix + "kmem"] = cache[outputPrefix + "k_neg_slice"];
            tensors[inputPrefix + "vmem"] = cache[outputPrefix + "v_neg_slice"];
        }

        tensors["encoder_block2_trunk0_layer0_mem"] = cache["encoder_block2_conv0_neg_slice"];
        tensors["encoder_block2_trunk0_layer1_mem"] = cache["encoder_block2_conv1_neg_slice"];
        for (int i = 0; i < 2; i++) {
            std::string inputPrefix =
                std::string("encoder_block2_trunk1_layer") + std::to_string(i) + std::string("_");
            std::string outputPrefix = std::string("encoder_block2_transformer_layer") +
                std::to_string(i) + std::string("_");
            tensors[inputPrefix + "kmem"] = cache[outputPrefix + "k_neg_slice"];
            tensors[inputPrefix + "vmem"] = cache[outputPrefix + "v_neg_slice"];
        }

        tensors["encoder_block3_trunk0_layer0_mem"] = cache["encoder_block3_conv0_neg_slice"];
        tensors["encoder_block3_trunk0_layer1_mem"] = cache["encoder_block3_conv1_neg_slice"];
        for (int i = 0; i < 4; i++) {
            std::string inputPrefix =
                std::string("encoder_block3_trunk1_layer") + std::to_string(i) + std::string("_");
            std::string outputPrefix = std::string("encoder_block3_transformer_layer") +
                std::to_string(i) + std::string("_");
            tensors[inputPrefix + "kmem"] = cache[outputPrefix + "k_neg_slice"];
            tensors[inputPrefix + "vmem"] = cache[outputPrefix + "v_neg_slice"];
        }
    } else {
        tensors["encoder_block0_trunk0_layer0_mem"] = std::shared_ptr<Tensor>(new Tensor());
        tensors["encoder_block0_trunk0_layer0_mem"]->resize(
            tensor4df(inferencePrecision, DF_NCHW, 1, 1, 128, 1));
        tensors["encoder_block0_trunk0_layer0_mem"]->alloc();
        tensors["encoder_block0_trunk0_layer1_mem"] = std::shared_ptr<Tensor>(new Tensor());
        tensors["encoder_block0_trunk0_layer1_mem"]->resize(
            tensor4df(inferencePrecision, DF_NCHWC8, 1, 32, 1, 64));
        tensors["encoder_block0_trunk0_layer1_mem"]->alloc();

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                std::string kv = std::string("k");
                if (j == 0) {
                    kv = std::string("v");
                }
                std::string name = std::string("encoder_block1_trunk1_layer") + std::to_string(i) +
                    std::string("_") + kv + "mem";
                tensors[name] = std::shared_ptr<Tensor>(new Tensor());
                tensors[name]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 0, 6, 64));
                tensors[name]->alloc();
            }
        }

        tensors["encoder_block2_trunk0_layer0_mem"] = std::shared_ptr<Tensor>(new Tensor());
        tensors["encoder_block2_trunk0_layer0_mem"]->resize(
            tensor3df(inferencePrecision, DF_NCHW, 1, 1, 384));
        tensors["encoder_block2_trunk0_layer0_mem"]->alloc();
        tensors["encoder_block2_trunk0_layer1_mem"] = std::shared_ptr<Tensor>(new Tensor());
        tensors["encoder_block2_trunk0_layer1_mem"]->resize(
            tensor4df(inferencePrecision, DF_NCHWC8, 1, 1024, 1, 1));
        tensors["encoder_block2_trunk0_layer1_mem"]->alloc();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                std::string kv = std::string("k");
                if (j == 0) {
                    kv = std::string("v");
                }
                std::string name = std::string("encoder_block2_trunk1_layer") + std::to_string(i) +
                    std::string("_") + kv + "mem";
                tensors[name] = std::shared_ptr<Tensor>(new Tensor());
                tensors[name]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 0, 8, 64));
                tensors[name]->alloc();
            }
        }

        tensors["encoder_block3_trunk0_layer0_mem"] = std::shared_ptr<Tensor>(new Tensor());
        tensors["encoder_block3_trunk0_layer0_mem"]->resize(
            tensor3df(inferencePrecision, DF_NCHW, 1, 1, 512));
        tensors["encoder_block3_trunk0_layer0_mem"]->alloc();
        tensors["encoder_block3_trunk0_layer1_mem"] = std::shared_ptr<Tensor>(new Tensor());
        tensors["encoder_block3_trunk0_layer1_mem"]->resize(
            tensor4df(inferencePrecision, DF_NCHWC8, 1, 1024, 1, 1));
        tensors["encoder_block3_trunk0_layer1_mem"]->alloc();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                std::string kv = std::string("k");
                if (j == 0) {
                    kv = std::string("v");
                }
                std::string name = std::string("encoder_block3_trunk1_layer") + std::to_string(i) +
                    std::string("_") + kv + "mem";
                tensors[name] = std::shared_ptr<Tensor>(new Tensor());
                tensors[name]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 0, 8, 64));
                tensors[name]->alloc();
            }
        }
        for (auto iter : tensors) {
            if (iter.first != std::string("sounds")) {
                TensorDesc desc = iter.second->get_desc();
                U8 *ptr = (U8 *)((CpuMemory *)(iter.second->get_memory()))->get_ptr();
                memset(ptr, 0, tensorNumBytes(desc));
            }
        }
    }
    std::shared_ptr<Tensor> tmp;
    encoderInferOutputSize(tensors, tmp, tensors);
    for (unsigned int i = 0; i < outputName.size(); i++) {
        tensors[outputName[i]]->alloc();
    }
    return tensors;
}

std::map<std::string, std::shared_ptr<Tensor>> getPredictionInputOutput(
    std::map<std::string, std::shared_ptr<Tensor>> jointResult,
    std::map<std::string, std::shared_ptr<Tensor>> cache)
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    if (jointResult.size() == 0) {
        tensors["label"] = std::shared_ptr<Tensor>(new Tensor());
        tensors["label"]->resize(tensor2df(DT_U32, DF_NORMAL, 1, 1));
        tensors["label"]->alloc();
        U32 *ptr = (U32 *)(((CpuMemory *)(tensors["label"]->get_memory()))->get_ptr());
        *ptr = START_TOKEN;
    } else {
        tensors["label"] = jointResult["output_argmax"];
    }
    if (cache.size() > 0) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                std::string kv = std::string("k");
                if (j == 0) {
                    kv = std::string("v");
                }
                std::string inputName = std::string("prediction_net_layer") + std::to_string(i) +
                    std::string("_") + kv + "mem";
                std::string outputName = std::string("prediction_net_layer") + std::to_string(i) +
                    std::string("_") + kv + "_neg_slice";
                tensors[inputName] = cache[outputName];
            }
        }
    } else {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                std::string kv = std::string("k");
                if (j == 0) {
                    kv = std::string("v");
                }
                std::string name = std::string("prediction_net_layer") + std::to_string(i) +
                    std::string("_") + kv + "mem";
                tensors[name] = std::shared_ptr<Tensor>(new Tensor());
                tensors[name]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 0, 8, 64));
                tensors[name]->alloc();
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            std::string kv = std::string("k");
            if (j == 0) {
                kv = std::string("v");
            }
            std::string name = std::string("prediction_net_layer") + std::to_string(i) +
                std::string("_") + kv + "_neg_slice";
            tensors[name] = std::shared_ptr<Tensor>(new Tensor());
        }
    }
    tensors["prediction_net_ln"] = std::shared_ptr<Tensor>(new Tensor());
    std::shared_ptr<Tensor> tmp;
    predictionInferOutputSize(tensors, tmp, tensors);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            std::string kv = std::string("k");
            if (j == 0) {
                kv = std::string("v");
            }
            std::string name = std::string("prediction_net_layer") + std::to_string(i) +
                std::string("_") + kv + "_neg_slice";
            tensors[name]->alloc();
        }
    }
    tensors["prediction_net_ln"]->alloc();
    return tensors;
}

std::map<std::string, std::shared_ptr<Tensor>> getJointInputOutput(
    std::map<std::string, std::shared_ptr<Tensor>> encoder,
    std::map<std::string, std::shared_ptr<Tensor>> prediction_net)
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    tensors["encoder"] = encoder["encoder_block3_transformer_ln"];
    tensors["prediction_net"] = prediction_net["prediction_net_ln"];
    tensors["output_argmax"] = std::shared_ptr<Tensor>(new Tensor());
    std::shared_ptr<Tensor> tmp;
    jointInferOutputSize(tensors, tmp, tensors);
    tensors["output_argmax"]->alloc();
    return tensors;
}

std::map<std::string, std::shared_ptr<Tensor>> getPinYin2HanZiInputOutput(int frameId,
    unsigned int *buffer,
    int bufferLength,
    int bufferValidSize,
    std::map<std::string, std::shared_ptr<Tensor>> joint)
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    tensors["pinyin"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["pinyin"]->resize(tensor2df(DT_U32, DF_NORMAL, 1, bufferLength));
    tensors["pinyin"]->alloc();
    if (frameId == 0) {
        memset(buffer, 0, sizeof(unsigned int) * bufferLength);
    }
    int pinyin = *((unsigned int *)((CpuMemory *)(joint["output_argmax"]->get_memory()))->get_ptr()) -
        PINYIN_FEATURE_GAP;
    CHECK_REQUIREMENT(pinyin >= 0);
    if (frameId < bufferValidSize) {
        buffer[frameId] = pinyin;
    } else {
        for (int i = 0; i < bufferValidSize - 1; i++) {
            buffer[i] = buffer[i + 1];
        }
        buffer[bufferValidSize - 1] = pinyin;
    }
    unsigned int *ptr = (unsigned int *)((CpuMemory *)(tensors["pinyin"]->get_memory()))->get_ptr();
    memcpy(ptr, buffer, sizeof(unsigned int) * bufferValidSize);
    memset(ptr + bufferValidSize, 0, sizeof(unsigned int) * (bufferLength - bufferValidSize));

    tensors["hanzi_squeeze/Squeeze"] = std::shared_ptr<Tensor>(new Tensor());
    std::shared_ptr<Tensor> tmp;
    pinyin2hanziInferOutputSize(tensors, tmp, tensors);
    tensors["hanzi_squeeze/Squeeze"]->alloc();
    return tensors;
}

std::vector<std::string> split(const std::string &str, const std::string &sep)
{
    std::vector<std::string> vec;
    if (str.empty()) {
        return vec;
    }

    size_t pos1;
    size_t pos2;
    pos2 = str.find(sep);
    pos1 = 0;
    while (std::string::npos != pos2) {
        vec.push_back(str.substr(pos1, pos2 - pos1));

        pos1 = pos2 + sep.size();
        pos2 = str.find(sep, pos1);
    }
    if (pos1 != str.length() * sizeof(typename std::string::value_type)) {
        vec.push_back(str.substr(pos1));
    }

    return vec;
}

std::map<std::string, std::vector<std::string>> loadLabels(std::string labelFilePath)
{
    std::map<std::string, std::vector<std::string>> labels;
    std::ifstream infile;
    infile.open(labelFilePath);
    if (!infile.is_open()) {
        return labels;
    }
    std::string s;
    int index = 0;
    while (getline(infile, s)) {
        switch (index) {
            case 0:
                labels["hanzi"] = split(s, std::string(" "));
                break;
            case 1:
                labels["pinyin"] = split(s, std::string(" "));
                break;
            default:
                UNI_WARNING_LOG("unrecognized label file line %s\n", s.c_str());
                break;
        }
        index++;
    }
    infile.close();
    return labels;
}

bool jointOutputIsBlank(std::map<std::string, std::shared_ptr<Tensor>> jointResult)
{
    if (jointResult.find("output_argmax") == jointResult.end()) {
        UNI_ERROR_LOG("unrecognized joint result");
    }
    TensorDesc desc = jointResult["output_argmax"]->get_desc();
    if (tensorNumElements(desc) != 1) {
        UNI_ERROR_LOG("unrecognized joint result(output_argmax) tensor");
    }
    U32 *ptr = (U32 *)((CpuMemory *)(jointResult["output_argmax"]->get_memory()))->get_ptr();
    bool ret = false;
    if (*ptr == BLANK_TOKEN) {
        ret = true;
    }
    return ret;
}

void freshPinYinResult(std::vector<std::pair<int, std::string>> &pinyinResult,
    std::vector<std::string> pinyinLabels,
    std::map<std::string, std::shared_ptr<Tensor>> joint,
    int frameId)
{
    int pinyin =
        *((unsigned int *)(((CpuMemory *)(joint["output_argmax"]->get_memory()))->get_ptr()));
    pinyinResult.push_back(std::pair<int, std::string>(frameId, pinyinLabels[pinyin]));
}

void freshHanZiResult(std::vector<std::pair<int, std::string>> &hanziResult,
    std::vector<std::string> hanziLabels,
    std::map<std::string, std::shared_ptr<Tensor>> pinyin2hanzi,
    int frameId)
{
    int pinyinBufferIndex = -1;
    if (frameId < PINYIN_BUFFER_VALID_SIZE) {
        pinyinBufferIndex = frameId;
    } else {
        pinyinBufferIndex = PINYIN_BUFFER_VALID_SIZE - 1;
    }
    int pinyin =
        ((U32 *)(((CpuMemory *)(pinyin2hanzi["pinyin"]->get_memory()))->get_ptr()))[pinyinBufferIndex] +
        PINYIN_FEATURE_GAP;
    if (pinyin == BLANK_TOKEN) {
        return;
    }
    hanziResult.push_back(std::pair<int, std::string>(frameId, "init"));
    std::shared_ptr<Tensor> hanziTensor = pinyin2hanzi["hanzi_squeeze/Squeeze"];
    TensorDesc hanziTensorDesc = hanziTensor->get_desc();
    int num = tensorNumElements(hanziTensorDesc);
    int loops = hanziTensorDesc.dims[1];
    int slots = hanziTensorDesc.dims[0];
    int batch = num / loops / slots;
    CHECK_REQUIREMENT(batch == 1);
    CHECK_REQUIREMENT(loops == PINYIN_BUFFER_SIZE);
    for (int i = hanziResult.size() - 1; i >= 0; i--) {
        std::pair<int, std::string> element = hanziResult[i];
        int lastFrameId = element.first;
        if (frameId - lastFrameId < PINYIN_BUFFER_VALID_SIZE) {
            int lastPinyinBufferIndex = pinyinBufferIndex - (frameId - lastFrameId);
            int offset = lastPinyinBufferIndex * slots;
            int maxIndex = offset;
            for (int j = 0, index = maxIndex; j < slots; j++, index++) {
                if (hanziTensor->element(maxIndex) < hanziTensor->element(index)) {
                    maxIndex = index;
                }
            }
            int hanziIndex = maxIndex - offset;
            hanziResult[i] = std::pair<int, std::string>(lastFrameId, hanziLabels[hanziIndex]);
        } else {
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    flowRegisterFunction("encoderInferOutputSize", encoderInferOutputSize);
    flowRegisterFunction("encoderPreProcess", encoderPreProcess);
    flowRegisterFunction("predictionInferOutputSize", predictionInferOutputSize);
    flowRegisterFunction("jointInferOutputSize", jointInferOutputSize);
    flowRegisterFunction("pinyin2hanziInferOutputSize", pinyin2hanziInferOutputSize);
    flowRegisterFunction("pinyin2hanziPreProcess", pinyin2hanziPreProcess);

    std::string wavFilePath = argv[6];
    AudioFeatureExtractor audioFeatureExtractor;
    std::vector<std::vector<std::vector<float>>> feature =
        audioFeatureExtractor.getEncoderInputFromWav(wavFilePath);

    std::string encoderGraphPath = argv[1];
    std::string predictionGraphPath = argv[2];
    std::string jointGraphPath = argv[3];
    std::string pinyin2hanziGraphPath = argv[4];
    std::string labelFilePath = argv[5];
    std::map<std::string, std::vector<std::string>> labels = loadLabels(labelFilePath);
    std::vector<std::string> graphPath = {
        encoderGraphPath, predictionGraphPath, jointGraphPath, pinyin2hanziGraphPath};
    // TODO(some): beam search conflict
    std::vector<unsigned int> pinyinBuffer(PINYIN_BUFFER_SIZE);

    int threads = 2;

    // TODO(some): beam search conflict
    int frameId = 0;
    Flow flowExample;
    flowExample.init(graphPath, DT_F32, AFFINITY_CPU_HIGH_PERFORMANCE, threads, false);
    sleep(5);

    std::map<std::string, std::shared_ptr<Tensor>> blankData;
    std::map<std::string, std::shared_ptr<Tensor>> encoderData =
        getEncoderInputOutput(feature, frameId, 15, blankData);
    if (encoderData.size() == 0) {
        return 0;
    }
    Task encoderTask(frameId, encoderGraphPath, encoderData);
    std::map<std::string, std::shared_ptr<Tensor>> predictionData =
        getPredictionInputOutput(blankData, blankData);
    Task predictionTask(frameId, predictionGraphPath, predictionData);
    double timeStart = ut_time_ms();
    flowExample.enqueue(encoderTask);
    flowExample.enqueue(predictionTask);
    frameId++;

    std::set<int> readyTaskId;
    std::map<int, Task> encoderResults;
    std::map<int, Task> predictionResults;
    std::map<int, Task> jointResults;
    std::vector<std::pair<int, std::string>> pinyinResult;
    std::vector<std::pair<int, std::string>> hanziResult;
    while (1) {
        std::vector<Task> results = flowExample.dequeue();
        for (unsigned int i = 0; i < results.size(); i++) {
            std::string graphPath = results[i].graphPath;
            if (graphPath == encoderGraphPath) {
                encoderResults[results[i].id] = results[i];
                readyTaskId.insert(results[i].id);
            } else if (graphPath == predictionGraphPath) {
                predictionResults[results[i].id] = results[i];
                readyTaskId.insert(results[i].id);
            } else if (graphPath == jointGraphPath) {
                jointResults[results[i].id] = results[i];
                // not skip blank will affect accuracy of result
                if (jointOutputIsBlank(results[i].data)) {
                    Task copyTask(&predictionResults[results[i].id]);
                    copyTask.id++;
                    predictionResults[copyTask.id] = copyTask;
                    readyTaskId.insert(copyTask.id);
                } else {
                    std::map<std::string, std::shared_ptr<Tensor>> predictionData =
                        getPredictionInputOutput(
                            results[i].data, predictionResults[results[i].id].data);
                    Task predictionTask(results[i].id + 1, predictionGraphPath, predictionData);
                    flowExample.enqueue(predictionTask);
                    freshPinYinResult(
                        pinyinResult, labels["pinyin"], results[i].data, results[i].id);
                }

                std::map<std::string, std::shared_ptr<Tensor>> pinyin2hanziData =
                    getPinYin2HanZiInputOutput(results[i].id, pinyinBuffer.data(),
                        PINYIN_BUFFER_SIZE, PINYIN_BUFFER_VALID_SIZE, results[i].data);
                Task pinyin2hanziTask(results[i].id, pinyin2hanziGraphPath, pinyin2hanziData);
                flowExample.enqueue(pinyin2hanziTask);
            } else if (graphPath == pinyin2hanziGraphPath) {
                freshHanZiResult(hanziResult, labels["hanzi"], results[i].data, results[i].id);
            }
        }
        for (std::set<int>::iterator iter = readyTaskId.begin(); iter != readyTaskId.end();) {
            int item = *iter;
            if (encoderResults.find(item) != encoderResults.end() &&
                predictionResults.find(item) != predictionResults.end()) {
                std::map<std::string, std::shared_ptr<Tensor>> jointData =
                    getJointInputOutput(encoderResults[item].data, predictionResults[item].data);
                Task jointTask(item, jointGraphPath, jointData);
                flowExample.enqueue(jointTask);
                iter = readyTaskId.erase(iter);
            } else {
                iter++;
            }
        }
        if (frameId < 1000 && encoderResults.find(frameId - 1) != encoderResults.end()) {
            std::map<std::string, std::shared_ptr<Tensor>> encoderData =
                getEncoderInputOutput(feature, frameId, 8, encoderResults[frameId - 1].data);
            if (encoderData.size() > 0) {
                Task encoderTask(frameId, encoderGraphPath, encoderData);
                frameId++;
                flowExample.enqueue(encoderTask);
            }
        }

        if (flowExample.size() == 0) {
            break;
        }
    }
    double timeEnd = ut_time_ms();
    std::string pinyinLine, hanziLine;
    for (unsigned int i = 0; i < pinyinResult.size(); i++) {
        pinyinLine += pinyinResult[i].second + " ";
        hanziLine += hanziResult[i].second;
    }
    std::cout << "[PROFILE] flow asr time: " << timeEnd - timeStart << " ms" << std::endl;
    std::cout << "[RESULT] length: " << pinyinResult.size() << std::endl;
    std::cout << "[RESULT] pinyin: " << pinyinLine << std::endl;
    std::cout << "[RESULT] hanzi: " << hanziLine << std::endl;
    return 0;
}
