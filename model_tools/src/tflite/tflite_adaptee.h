// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TFLITEADAPTEE
#define _H_TFLITEADAPTEE
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_adaptee.h"
#include "tensor_transpose.h"
#include "array_transpose.h"

class TfliteAdaptee : public ModelAdaptee {
public:
    TfliteAdaptee()
    {
        this->weightFormat = DF_NHWC;
    }

    ~TfliteAdaptee()
    {}

protected:
    std::vector<int> getOperatorTensorInputIndex(int operatorIndex)
    {
        std::vector<int> index;
        for (U32 i = 0; i < this->tfliteOperators[operatorIndex]->inputs.size(); i++) {
            if (this->tfliteModelBuffer
                    [this->tfliteTensors[this->tfliteOperators[operatorIndex]->inputs[i]]->buffer]
                        ->data.size() == 0) {
                index.push_back(i);
            }
        }
        return index;
    }

    std::vector<int> getOperatorWeightInputIndex(int operatorIndex)
    {
        std::vector<int> index;
        for (U32 i = 0; i < this->tfliteOperators[operatorIndex]->inputs.size(); i++) {
            if (this->tfliteModelBuffer
                    [this->tfliteTensors[this->tfliteOperators[operatorIndex]->inputs[i]]->buffer]
                        ->data.size() > 0) {
                index.push_back(i);
            }
        }
        return index;
    }

    OperatorType convert_tflite_type(tflite::BuiltinOperator tfliteOperatorType)
    {
        std::vector<int> weightInputIndex = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        if (tfliteOperatorType == tflite::BuiltinOperator_ADD ||
            tfliteOperatorType == tflite::BuiltinOperator_MUL ||
            tfliteOperatorType == tflite::BuiltinOperator_DIV ||
            tfliteOperatorType == tflite::BuiltinOperator_SUB) {
            if (weightInputIndex.size() > 0) {
                if (this->tfliteModelBuffer[this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]
                                                                    ->inputs[weightInputIndex[0]]]
                                                ->buffer]
                        ->data.size() == sizeof(float)) {
                    return OT_Power;
                } else {
                    return OT_Scale;
                }
            } else {
                return OT_Eltwise;
            }
        } else if (tfliteOperatorType == tflite::BuiltinOperator_CONCATENATION ||
            tfliteOperatorType == tflite::BuiltinOperator_PACK) {
            return OT_Concat;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_CONV_2D ||
            tfliteOperatorType == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
            return OT_Conv;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_LOGISTIC) {
            return OT_Sigmoid;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_MAX_POOL_2D ||
            tfliteOperatorType == tflite::BuiltinOperator_AVERAGE_POOL_2D) {
            return OT_Pooling;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_RESHAPE) {
            return OT_Reshape;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_RESIZE_BILINEAR ||
            tfliteOperatorType == tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR) {
            return OT_Resize;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_SOFTMAX) {
            return OT_Softmax;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_FULLY_CONNECTED) {
            if (weightInputIndex.size() > 0) {
                bool ttW = (weightInputIndex.size() == 1 && weightInputIndex[0] == 2) ? true : false;
                bool wtW = (weightInputIndex.size() == 2 && weightInputIndex[0] == 0 &&
                               weightInputIndex[1] == 2)
                    ? true
                    : false;
                if (ttW || wtW) {
                    bool fullZero = true;
                    const int biasIndex =
                        this->tfliteOperators[this->tfliteOperatorIndex]->inputs[2];
                    const auto &biasTensor = this->tfliteTensors[biasIndex];
                    std::vector<float> bias = transformTfliteTensorToVector(biasTensor);
                    for (U32 k = 0; k < bias.size(); k++) {
                        if (bias[k] != 0) {
                            fullZero = false;
                            break;
                        }
                    }
                    if (fullZero) {
                        return OT_MatMul;
                    } else {
                        UNI_ERROR_LOG("operator location:%d type:%s not support bias with "
                                      "non-zero weights.\n",
                            this->tfliteOperatorIndex,
                            tflite::EnumNamesBuiltinOperator()[tfliteOperatorType]);
                    }
                }
                return OT_FC;
            } else {
                return OT_MatMul;
            }
        } else if (tfliteOperatorType == tflite::BuiltinOperator_TRANSPOSE) {
            return OT_Transpose;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_SLICE ||
            tfliteOperatorType == tflite::BuiltinOperator_STRIDED_SLICE) {
            return OT_TfSlice;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_RELU ||
            tfliteOperatorType == tflite::BuiltinOperator_LEAKY_RELU) {
            return OT_Relu;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_RELU6) {
            return OT_Relu6;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_TANH) {
            return OT_TanH;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_MEAN) {
            if (this->tfliteModelBuffer
                    [this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]]
                            ->buffer]
                        ->data.size() != 8) {
                return OT_Reduction;
            } else {
                return OT_Pooling;
            }
        } else if (tfliteOperatorType == tflite::BuiltinOperator_MAXIMUM) {
            if (weightInputIndex.size() == 0) {
                return OT_Eltwise;
            } else {
                return OT_Clip;
            }
        } else if (tfliteOperatorType == tflite::BuiltinOperator_MINIMUM) {
            return OT_Clip;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_TRANSPOSE_CONV) {
            return OT_Deconvolution;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_SQUARED_DIFFERENCE) {
            return OT_SqDiff;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_SQRT ||
            tfliteOperatorType == tflite::BuiltinOperator_POW) {
            return OT_Power;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_L2_NORMALIZATION) {
            return OT_L2Normalization;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_PAD ||
            tfliteOperatorType == tflite::BuiltinOperator_MIRROR_PAD) {
            return OT_Pad;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_HARD_SWISH) {
            return OT_HSwish;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_SHAPE) {
            return OT_Shape;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_SQUEEZE) {
            return OT_Squeeze;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_EXPAND_DIMS) {
            return OT_Unsqueeze;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_NEG) {
            return OT_Power;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_TOPK_V2) {
            return OT_TopK;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_GATHER) {
            return OT_Embedding;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_PRELU) {
            return OT_PRelu;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_SPACE_TO_BATCH_ND) {
            return OT_SpaceToBatchNd;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_BATCH_TO_SPACE_ND) {
            return OT_BatchToSpaceNd;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_ABS) {
            return OT_Abs;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_QUANTIZE ||
            tfliteOperatorType == tflite::BuiltinOperator_FAKE_QUANT) {
            return OT_Split;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_SPLIT) {
            return OT_Slice;
        } else if (tfliteOperatorType == tflite::BuiltinOperator_EXP) {
            return OT_Exp;
        } else {
            UNI_ERROR_LOG("operator locate:%d type:%s not supported.\n", this->tfliteOperatorIndex,
                tflite::EnumNamesBuiltinOperator()[tfliteOperatorType]);
            return OT_None;
        }
    }

    EE parse_file(std::string dir, std::string mfn) override
    {
        EE ret = SUCCESS;
        std::string tfliteSuffix = ".tflite";

        this->modelName = mfn;

        std::string model_name = dir + "/" + mfn + tfliteSuffix;
        std::ifstream inputFile(model_name.c_str(), std::ios::binary);
        if (!inputFile.is_open()) {
            UNI_ERROR_LOG("can not find tflite model file %s.\n", model_name.c_str());
        }
        inputFile.seekg(0, std::ios::end);
        const auto size = inputFile.tellg();
        inputFile.seekg(0, std::ios::beg);

        char *buffer = new char[size];
        inputFile.read(buffer, size);
        inputFile.close();

        flatbuffers::Verifier verify((uint8_t *)buffer, size);
        CHECK_REQUIREMENT(tflite::VerifyModelBuffer(verify));

        auto tfliteModel = tflite::UnPackModel(buffer);

        tfliteOpSet.clear();
        for (int i = 0; i < (int)(tfliteModel->operator_codes).size(); i++) {
            tfliteOpSet.push_back(std::move((tfliteModel->operator_codes)[i]));
        }

        const auto subGraphsSize = tfliteModel->subgraphs.size();
        CHECK_REQUIREMENT(subGraphsSize == 1);

        tfliteModelBuffer.clear();
        for (int i = 0; i < (int)(tfliteModel->buffers).size(); i++) {
            tfliteModelBuffer.push_back(std::move((tfliteModel->buffers)[i]));
        }

        if (subGraphsSize != 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }

        this->tfliteOperators.clear();
        for (int i = 0; i < (int)(tfliteModel->subgraphs[0]->operators).size(); i++) {
            this->tfliteOperators.push_back(std::move((tfliteModel->subgraphs[0]->operators)[i]));
        }

        this->tfliteTensors.clear();
        for (int i = 0; i < (int)(tfliteModel->subgraphs[0]->tensors).size(); i++) {
            this->tfliteTensors.push_back(std::move((tfliteModel->subgraphs[0]->tensors)[i]));
        }

        inputs.clear();
        for (int i = 0; i < (int)(tfliteModel->subgraphs[0]->inputs).size(); i++) {
            inputs.push_back(std::move((tfliteModel->subgraphs[0]->inputs)[i]));
        }

        outputs.clear();
        for (int i = 0; i < (int)(tfliteModel->subgraphs[0]->outputs).size(); i++) {
            outputs.push_back(std::move((tfliteModel->subgraphs[0]->outputs)[i]));
        }

        return ret;
    }

    DataType TfliteTensorType2BoltDataType(tflite::TensorType type)
    {
        DataType dt = DT_F32;
        switch (type) {
            case tflite::TensorType_FLOAT32: {
                dt = DT_F32;
                break;
            }
            case tflite::TensorType_INT32: {
                dt = DT_I32;
                break;
            }
            default: {
                UNI_ERROR_LOG(
                    "can not process tflite data type %s.\n", tflite::EnumNamesTensorType()[type]);
                break;
            }
        }
        return dt;
    }

    EE adapt_operators(ModelSpec *ms) override
    {
        this->modelWeightOpNum = 0;
        EE ret = SUCCESS;
        ms->dt = DT_F32;
        str_copy(ms->model_name, modelName.c_str(), modelName.length());
        ms->num_inputs = inputs.size();
        ms->input_names = (I8 **)mt_new_storage(ms->num_inputs * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_new_storage(sizeof(TensorDesc) * ms->num_inputs);
        for (I32 i = 0; i < ms->num_inputs; i++) {
            const int inputIdx = inputs[i];
            const auto &inputTensor = this->tfliteTensors[inputIdx];
            ms->input_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ms->input_names[i], (inputTensor->name).c_str(), (inputTensor->name).length());
            std::vector<int> inputShape(inputTensor->shape);
            if (this->weightFormat == DF_NHWC) {
                shiftRight<int>(inputShape.data(), inputShape.size(), 1, inputShape.size() - 1);
            }
            ms->input_dims[i].nDims = inputShape.size();
            ms->input_dims[i].dt = TfliteTensorType2BoltDataType(inputTensor->type);
            ms->input_dims[i].df = getTensorDefaultDataFormat(ms->input_dims[i].nDims);
            for (U32 j = 0; j < ms->input_dims[i].nDims; j++) {
                ms->input_dims[i].dims[ms->input_dims[i].nDims - 1 - j] = inputShape[j];
            }
        }
        ms->num_outputs = outputs.size();
        ms->output_names = (I8 **)mt_new_storage(ms->num_outputs * sizeof(I8 *));
        for (I32 i = 0; i < ms->num_outputs; i++) {
            const int outputIdx = outputs[i];
            const auto &outputTensor = this->tfliteTensors[outputIdx];
            ms->output_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(
                ms->output_names[i], (outputTensor->name).c_str(), (outputTensor->name).length());
        }

        this->boltOperators = std::vector<OperatorSpec>(this->tfliteOperators.size());
        for (this->boltOperatorIndex = 0, this->tfliteOperatorIndex = 0;
             this->tfliteOperatorIndex < this->tfliteOperators.size();
             this->boltOperatorIndex++, this->tfliteOperatorIndex++) {
            UNI_DEBUG_LOG("process operator location:%d parameter.\n", this->tfliteOperatorIndex);
            std::string operatorName = "op" + std::to_string(this->tfliteOperatorIndex);
            str_copy(this->boltOperators[this->boltOperatorIndex].name, operatorName.c_str(),
                operatorName.length());
            const int opcodeIndex = this->tfliteOperators[this->tfliteOperatorIndex]->opcode_index;
            this->opCode = tfliteOpSet[opcodeIndex]->builtin_code;
            this->boltOperators[this->boltOperatorIndex].type = convert_tflite_type(this->opCode);
            this->boltOperators[this->boltOperatorIndex].num_inputs =
                (modifiedInputsOp.find(this->boltOperators[this->boltOperatorIndex].type) ==
                    modifiedInputsOp.end())
                ? this->tfliteOperators[this->tfliteOperatorIndex]->inputs.size()
                : modifiedInputsOp[this->boltOperators[this->boltOperatorIndex].type];
            this->boltOperators[this->boltOperatorIndex].input_tensors_name = (I8 **)mt_new_storage(
                this->boltOperators[this->boltOperatorIndex].num_inputs * sizeof(I8 *));

            int inputStartPoint = 0;
            if (opCode == tflite::BuiltinOperator_TRANSPOSE_CONV) {
                inputStartPoint = 2;
            } else if (opCode == tflite::BuiltinOperator_MUL) {
                std::vector<int> tensorInputIndex =
                    getOperatorTensorInputIndex(this->tfliteOperatorIndex);
                inputStartPoint = tensorInputIndex[0];
            } else if (opCode == tflite::BuiltinOperator_SPLIT) {
                inputStartPoint = 1;
            }

            for (U32 iter = 0; iter < this->boltOperators[this->boltOperatorIndex].num_inputs;
                 iter++) {
                const int inIndex =
                    this->tfliteOperators[this->tfliteOperatorIndex]->inputs[iter + inputStartPoint];
                const auto &inTensor = this->tfliteTensors[inIndex];
                this->boltOperators[this->boltOperatorIndex].input_tensors_name[iter] =
                    (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(this->boltOperators[this->boltOperatorIndex].input_tensors_name[iter],
                    (inTensor->name).c_str(), (inTensor->name).length());
            }
            this->boltOperators[this->boltOperatorIndex].num_outputs =
                this->tfliteOperators[this->tfliteOperatorIndex]->outputs.size();
            this->boltOperators[this->boltOperatorIndex].output_tensors_name = (I8 **)mt_new_storage(
                this->boltOperators[this->boltOperatorIndex].num_outputs * sizeof(I8 *));
            for (U32 iter = 0; iter < this->boltOperators[this->boltOperatorIndex].num_outputs;
                 iter++) {
                const int outIndex = this->tfliteOperators[this->tfliteOperatorIndex]->outputs[iter];
                const auto &outTensor = this->tfliteTensors[outIndex];
                this->boltOperators[this->boltOperatorIndex].output_tensors_name[iter] =
                    (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(this->boltOperators[this->boltOperatorIndex].output_tensors_name[iter],
                    outTensor->name.c_str(), outTensor->name.length());
            }
            this->boltOperatorInsertBefore = 0;
            this->boltOperatorInsertAfter = 0;
            ParameterSpec boltParameterSpec;
            ret = adapt_operator(
                this->boltOperators[this->boltOperatorIndex].type, &(boltParameterSpec));
            this->boltOperators[this->boltOperatorIndex + this->boltOperatorInsertBefore].ps =
                boltParameterSpec;
            this->boltOperatorIndex +=
                this->boltOperatorInsertBefore + this->boltOperatorInsertAfter;
        }

        ms->num_operator_specs = this->boltOperators.size();
        ms->ops = (OperatorSpec *)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
        memcpy(ms->ops, this->boltOperators.data(), sizeof(OperatorSpec) * ms->num_operator_specs);
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            this->boltOperatorNameMap[ms->ops[i].name] = i;
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }
        ms->num_weight_specs = modelWeightOpNum;
        return ret;
    }

    int NHWCAxisToNCHWAxis(int nhwcAxis, int dimSize)
    {
        // tflite may not record tensor shape
        if (dimSize == 0) {
            dimSize = 4;
        }
        if (nhwcAxis < 0) {
            nhwcAxis += dimSize;
        }
        int nchwAxis = nhwcAxis;
        // only transpose 4-dim parameter
        if (dimSize >= 4) {
            if (nhwcAxis != 0) {
                nchwAxis++;
            }
            if (nhwcAxis == dimSize - 1) {
                nchwAxis = 1;
            }
        }
        return nchwAxis;
    }

    void bitsToCharArray(int bit, char *array, int length)
    {
        for (int i = 0; i < length; i++) {
            array[i] = bit & 1;
            bit = bit >> 1;
        }
    }

    template <typename T>
    void shiftRight(T *array, int length, int left, int right)
    {
        // only transpose 4-dim parameter
        if (length >= 4) {
            T data = array[right];
            for (int i = right; i > left; i--) {
                array[i] = array[i - 1];
            }
            array[left] = data;
        }
    }

    std::vector<float> transformTfliteTensorToVector(const std::unique_ptr<tflite::TensorT> &tensor)
    {
        const auto &weightShape = tensor->shape;
        U32 size = 1;
        for (U32 i = 0; i < weightShape.size(); i++) {
            size *= weightShape[i];
        }
        std::vector<float> result(size);
        switch (tensor->type) {
            case tflite::TensorType_FLOAT32: {
                auto weight = reinterpret_cast<const float *>(
                    this->tfliteModelBuffer[tensor->buffer]->data.data());
                UNI_MEMCPY(result.data(), weight, sizeof(float) * size);
                break;
            }
            case tflite::TensorType_INT64: {
                auto weight = reinterpret_cast<const int64_t *>(
                    this->tfliteModelBuffer[tensor->buffer]->data.data());
                for (U32 i = 0; i < size; i++) {
                    result[i] = weight[i];
                }
                break;
            }
            case tflite::TensorType_INT32: {
                auto weight = reinterpret_cast<const int32_t *>(
                    this->tfliteModelBuffer[tensor->buffer]->data.data());
                for (U32 i = 0; i < size; i++) {
                    result[i] = weight[i];
                }
                break;
            }
            case tflite::TensorType_INT8: {
                auto weight = reinterpret_cast<const int8_t *>(
                    this->tfliteModelBuffer[tensor->buffer]->data.data());
                // if scale_size > 1, this is per channel quantization
                int scale_size = tensor->quantization->scale.size();
                int shift_size = tensor->quantization->zero_point.size();
                int loops = size / scale_size;
                for (int i = 0, index = 0; i < scale_size; i++) {
                    float scale = tensor->quantization->scale[i % scale_size];
                    float shift = 0;
                    if (shift_size > 0) {
                        shift = tensor->quantization->zero_point[i % shift_size];
                    }
                    for (int j = 0; j < loops; j++, index++) {
                        result[index] = weight[index] * scale + shift;
                    }
                }
                break;
            }
            case tflite::TensorType_UINT8: {
                auto weight = reinterpret_cast<const uint8_t *>(
                    this->tfliteModelBuffer[tensor->buffer]->data.data());
                // if scale_size > 1, this is per channel quantization
                int scale_size = tensor->quantization->scale.size();
                int shift_size = tensor->quantization->zero_point.size();
                int loops = size / scale_size;
                for (int i = 0, index = 0; i < scale_size; i++) {
                    float scale = tensor->quantization->scale[i % scale_size];
                    float shift = 0;
                    if (shift_size > 0) {
                        shift = tensor->quantization->zero_point[i % shift_size];
                    }
                    for (int j = 0; j < loops; j++, index++) {
                        result[index] = (weight[index] - shift) * scale;
                    }
                }
                break;
            }
            default: {
                UNI_ERROR_LOG("can not process operator location:%d %s type weight data.\n",
                    this->tfliteOperatorIndex, tflite::EnumNamesTensorType()[tensor->type]);
                break;
            }
        }
        return result;
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        WeightSpec *wsPtr = (WeightSpec *)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        for (int j = 0; j < ms->num_weight_specs; j++) {
            wsPtr[j].num_quant_scale = 0;
            wsPtr[j].weight_scale = nullptr;
        }
        ms->ws = wsPtr;
        UNI_MEMCPY(ms->ws, this->boltSharedWeights.data(),
            this->boltSharedWeights.size() * sizeof(WeightSpec));
        int weightMovIndex = this->boltSharedWeights.size();
        for (this->tfliteOperatorIndex = 0;
             this->tfliteOperatorIndex < this->tfliteOperators.size(); this->tfliteOperatorIndex++) {
            UNI_DEBUG_LOG("process operator location:%d weight.\n", this->tfliteOperatorIndex);
            std::string operatorName = "op" + std::to_string(this->tfliteOperatorIndex);
            this->boltOperatorIndex = this->boltOperatorNameMap[operatorName];
            const int opcodeIndex = this->tfliteOperators[this->tfliteOperatorIndex]->opcode_index;
            opCode = tfliteOpSet[opcodeIndex]->builtin_code;

            if (opCode == tflite::BuiltinOperator_CONV_2D ||
                opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
                str_copy(wsPtr[weightMovIndex].op_name, operatorName.c_str(), operatorName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;
                // input 2/3: input/weight/bias
                const int weightIndex = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
                const auto &weightTensor = this->tfliteTensors[weightIndex];
                std::vector<float> conv2DWeight = transformTfliteTensorToVector(weightTensor);
                const auto &weightShape = weightTensor->shape;
                CHECK_REQUIREMENT(weightShape.size() == 4);
                const int conv2d_co = weightShape[0];
                const int conv2d_kh = weightShape[1];
                const int conv2d_kw = weightShape[2];
                const int conv2d_ci = weightShape[3];
                wsPtr[weightMovIndex].bytes_of_weight =
                    conv2d_co * conv2d_kh * conv2d_kw * conv2d_ci * sizeof(float);
                wsPtr[weightMovIndex].weight =
                    (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                TensorDesc nhwcWeightDesc =
                    tensor4df(DT_F32, DF_NHWC, conv2d_co, conv2d_ci, conv2d_kh, conv2d_kw);
                TensorDesc nchwWeightDesc =
                    tensor4df(DT_F32, DF_NCHW, conv2d_co, conv2d_ci, conv2d_kh, conv2d_kw);
                transformToNCHW(nhwcWeightDesc, conv2DWeight.data(), nchwWeightDesc,
                    wsPtr[weightMovIndex].weight);

                if (this->tfliteOperators[this->tfliteOperatorIndex]->inputs.size() == 3) {
                    const int biasIndex =
                        this->tfliteOperators[this->tfliteOperatorIndex]->inputs[2];
                    const auto &biasTensor = this->tfliteTensors[biasIndex];
                    std::vector<float> conv2DBias = transformTfliteTensorToVector(biasTensor);
                    if (opCode == tflite::BuiltinOperator_CONV_2D) {
                        wsPtr[weightMovIndex].bytes_of_vec = conv2d_co * sizeof(float);
                    } else {
                        wsPtr[weightMovIndex].bytes_of_vec = conv2d_ci * sizeof(float);
                    }
                    wsPtr[weightMovIndex].vec =
                        (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_vec);
                    memcpy(wsPtr[weightMovIndex].vec, conv2DBias.data(),
                        wsPtr[weightMovIndex].bytes_of_vec);
                } else {
                    wsPtr[weightMovIndex].bytes_of_vec = 0;
                    wsPtr[weightMovIndex].vec = nullptr;
                }
                weightMovIndex++;
            } else if (OT_Scale == ms->ops[this->boltOperatorIndex].type) {
                str_copy(wsPtr[weightMovIndex].op_name, operatorName.c_str(), operatorName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;
                std::vector<int> weightInputIndex =
                    getOperatorWeightInputIndex(this->tfliteOperatorIndex);
                if (weightInputIndex.size() == 0) {
                    UNI_ERROR_LOG("can not map operator location:%d type:%s to Scale.\n",
                        this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
                }
                switch (opCode) {
                    case tflite::BuiltinOperator_ADD: {
                        wsPtr[weightMovIndex].bytes_of_weight = 0;
                        wsPtr[weightMovIndex].weight = nullptr;

                        const int biasIndex = this->tfliteOperators[this->tfliteOperatorIndex]
                                                  ->inputs[weightInputIndex[0]];
                        const auto &biasTensor = this->tfliteTensors[biasIndex];
                        std::vector<float> bias = transformTfliteTensorToVector(biasTensor);
                        wsPtr[weightMovIndex].bytes_of_vec = bias.size() * sizeof(float);
                        if (wsPtr[weightMovIndex].bytes_of_vec == 4) {
                            ms->ops[this->boltOperatorIndex].ps.scale_spec.axis = 0;
                        }
                        wsPtr[weightMovIndex].vec =
                            (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_vec);
                        memcpy(wsPtr[weightMovIndex].vec, bias.data(),
                            wsPtr[weightMovIndex].bytes_of_vec);
                        break;
                    }
                    case tflite::BuiltinOperator_SUB: {
                        wsPtr[weightMovIndex].bytes_of_weight = 0;
                        wsPtr[weightMovIndex].weight = nullptr;

                        const int biasIndex = this->tfliteOperators[this->tfliteOperatorIndex]
                                                  ->inputs[weightInputIndex[0]];
                        const auto &biasTensor = this->tfliteTensors[biasIndex];
                        std::vector<float> bias = transformTfliteTensorToVector(biasTensor);
                        wsPtr[weightMovIndex].bytes_of_vec = bias.size() * sizeof(float);
                        if (wsPtr[weightMovIndex].bytes_of_vec == 4) {
                            ms->ops[this->boltOperatorIndex].ps.scale_spec.axis = 0;
                        }
                        wsPtr[weightMovIndex].vec =
                            (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_vec);
                        F32 *ptr = (F32 *)wsPtr[weightMovIndex].vec;
                        for (U32 k = 0; k < bias.size(); k++) {
                            ptr[k] = -1 * bias[k];
                        }
                        break;
                    }
                    case tflite::BuiltinOperator_MUL: {
                        const int scaleIndex = this->tfliteOperators[this->tfliteOperatorIndex]
                                                   ->inputs[weightInputIndex[0]];
                        const auto &scaleTensor = this->tfliteTensors[scaleIndex];
                        std::vector<float> scale = transformTfliteTensorToVector(scaleTensor);
                        wsPtr[weightMovIndex].bytes_of_weight = scale.size() * sizeof(float);
                        if (wsPtr[weightMovIndex].bytes_of_weight == 4) {
                            ms->ops[this->boltOperatorIndex].ps.scale_spec.axis = 0;
                        }
                        wsPtr[weightMovIndex].weight =
                            (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                        memcpy(wsPtr[weightMovIndex].weight, scale.data(),
                            wsPtr[weightMovIndex].bytes_of_weight);

                        wsPtr[weightMovIndex].bytes_of_vec = 0;
                        wsPtr[weightMovIndex].vec = nullptr;
                        break;
                    }
                    case tflite::BuiltinOperator_DIV: {
                        const int scaleIndex = this->tfliteOperators[this->tfliteOperatorIndex]
                                                   ->inputs[weightInputIndex[0]];
                        const auto &scaleTensor = this->tfliteTensors[scaleIndex];
                        std::vector<float> scale = transformTfliteTensorToVector(scaleTensor);
                        wsPtr[weightMovIndex].bytes_of_weight = scale.size() * sizeof(float);
                        if (wsPtr[weightMovIndex].bytes_of_weight == 4) {
                            ms->ops[this->boltOperatorIndex].ps.scale_spec.axis = 0;
                        }
                        wsPtr[weightMovIndex].weight =
                            (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                        F32 *ptr = (F32 *)wsPtr[weightMovIndex].weight;
                        for (U32 k = 0; k < scale.size(); k++) {
                            ptr[k] = 1 / scale[k];
                        }

                        wsPtr[weightMovIndex].bytes_of_vec = 0;
                        wsPtr[weightMovIndex].vec = nullptr;
                        break;
                    }
                    default: {
                        UNI_ERROR_LOG("can not map operator location:%d type:%s to Scale.\n",
                            this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[opCode]);
                        break;
                    }
                }
                weightMovIndex++;
            } else if (OT_FC == ms->ops[this->boltOperatorIndex].type) {
                str_copy(wsPtr[weightMovIndex].op_name, operatorName.c_str(), operatorName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;
                const int weightIndex = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
                const auto &weightTensor = this->tfliteTensors[weightIndex];
                std::vector<float> fcWeight = transformTfliteTensorToVector(weightTensor);
                const auto &weightShape = weightTensor->shape;
                CHECK_REQUIREMENT(weightShape.size() == 2);
                wsPtr[weightMovIndex].bytes_of_weight = fcWeight.size() * sizeof(float);
                wsPtr[weightMovIndex].weight =
                    (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                memcpy(wsPtr[weightMovIndex].weight, fcWeight.data(),
                    wsPtr[weightMovIndex].bytes_of_weight);

                if (this->tfliteOperators[this->tfliteOperatorIndex]->inputs.size() == 3) {
                    const int biasIndex =
                        this->tfliteOperators[this->tfliteOperatorIndex]->inputs[2];
                    const auto &biasTensor = this->tfliteTensors[biasIndex];
                    std::vector<float> fcBias = transformTfliteTensorToVector(biasTensor);
                    wsPtr[weightMovIndex].bytes_of_vec = fcBias.size() * sizeof(float);
                    wsPtr[weightMovIndex].vec =
                        (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_vec);
                    memcpy(wsPtr[weightMovIndex].vec, fcBias.data(),
                        wsPtr[weightMovIndex].bytes_of_vec);
                } else {
                    wsPtr[weightMovIndex].bytes_of_vec = 0;
                    wsPtr[weightMovIndex].vec = nullptr;
                }
                weightMovIndex++;
            } else if (OT_Deconvolution == ms->ops[this->boltOperatorIndex].type) {
                str_copy(wsPtr[weightMovIndex].op_name, operatorName.c_str(), operatorName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;
                const int weightIndex = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
                const auto &weightTensor = this->tfliteTensors[weightIndex];
                std::vector<float> deConvWeight = transformTfliteTensorToVector(weightTensor);
                const auto &weightShape = weightTensor->shape;
                CHECK_REQUIREMENT(weightShape.size() == 4);
                U32 conv2d_co = weightShape[0];
                U32 conv2d_kh = weightShape[1];
                U32 conv2d_kw = weightShape[2];
                U32 conv2d_ci = weightShape[3];
                wsPtr[weightMovIndex].bytes_of_weight = deConvWeight.size() * sizeof(float);
                wsPtr[weightMovIndex].weight =
                    (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                U32 filterDims[4] = {conv2d_ci, conv2d_kw, conv2d_kh, conv2d_co};
                U32 ftmDims[4] = {conv2d_kw, conv2d_kh, conv2d_co, conv2d_ci};
                U32 filterTransformDims[4] = {3, 0, 1, 2};
                array_transpose(bytesOf(DT_F32), filterDims, deConvWeight.data(), ftmDims,
                    wsPtr[weightMovIndex].weight, filterTransformDims, 4, 4);
                if (this->tfliteOperators[this->tfliteOperatorIndex]->inputs.size() == 4) {
                    UNI_ERROR_LOG(
                        "can not process operator location:%d bias.\n", this->tfliteOperatorIndex);
                }
                wsPtr[weightMovIndex].bytes_of_vec = 0;
                wsPtr[weightMovIndex].vec = nullptr;
                weightMovIndex++;
            } else if (OT_PRelu == ms->ops[this->boltOperatorIndex].type) {
                str_copy(wsPtr[weightMovIndex].op_name, operatorName.c_str(), operatorName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;
                const int slopeIndex = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
                const auto &slopeTensor = this->tfliteTensors[slopeIndex];
                std::vector<float> slopeWeight = transformTfliteTensorToVector(slopeTensor);
                wsPtr[weightMovIndex].bytes_of_weight = slopeWeight.size() * sizeof(float);
                wsPtr[weightMovIndex].weight =
                    (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                memcpy(wsPtr[weightMovIndex].weight, &(slopeWeight[0]),
                    wsPtr[weightMovIndex].bytes_of_weight);
                wsPtr[weightMovIndex].bytes_of_vec = 0;
                wsPtr[weightMovIndex].vec = nullptr;
                weightMovIndex++;
            } else if (OT_SpaceToBatchNd == ms->ops[this->boltOperatorIndex].type ||
                OT_BatchToSpaceNd == ms->ops[this->boltOperatorIndex].type) {
                str_copy(wsPtr[weightMovIndex].op_name, operatorName.c_str(), operatorName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;
                const int blockShapeIndex =
                    this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
                const auto &blockShapeTp = this->tfliteTensors[blockShapeIndex];
                std::vector<float> blockShapeWeight = transformTfliteTensorToVector(blockShapeTp);
                wsPtr[weightMovIndex].bytes_of_weight = blockShapeWeight.size() * sizeof(float);
                wsPtr[weightMovIndex].weight =
                    (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_weight);
                memcpy(wsPtr[weightMovIndex].weight, &(blockShapeWeight[0]),
                    wsPtr[weightMovIndex].bytes_of_weight);

                const int paddingsIndex =
                    this->tfliteOperators[this->tfliteOperatorIndex]->inputs[2];
                const auto &paddingsTp = this->tfliteTensors[paddingsIndex];
                std::vector<float> paddingsWeight = transformTfliteTensorToVector(paddingsTp);
                wsPtr[weightMovIndex].bytes_of_vec = paddingsWeight.size() * sizeof(float);
                wsPtr[weightMovIndex].vec = (U8 *)mt_new_storage(wsPtr[weightMovIndex].bytes_of_vec);
                memcpy(wsPtr[weightMovIndex].vec, &(paddingsWeight[0]),
                    wsPtr[weightMovIndex].bytes_of_vec);
                weightMovIndex++;
            }
        }
        return SUCCESS;
    }

    ActivationMode getActivationOperatorType(
        const tflite::ActivationFunctionType &tfliteActivationType)
    {
        ActivationMode ret = ACTIVATION_NULL;
        switch (tfliteActivationType) {
            case tflite::ActivationFunctionType_NONE:
                ret = ACTIVATION_NULL;
                break;
            case tflite::ActivationFunctionType_RELU:
                ret = ACTIVATION_RELU;
                break;
            case tflite::ActivationFunctionType_RELU6:
                ret = ACTIVATION_RELU6;
                break;
            default:
                UNI_ERROR_LOG("can not process operator location:%d type:%s.\n", tfliteOperatorIndex,
                    tflite::EnumNamesActivationFunctionType()[tfliteActivationType]);
                break;
        }
        return ret;
    }

    void insertActivationOperator(ActivationMode activationMode)
    {
        if (activationMode == ACTIVATION_NULL) {
            return;
        }
        OperatorSpec activation;
        int index = this->boltOperatorIndex + this->boltOperatorInsertBefore +
            this->boltOperatorInsertAfter;
        const char *name = this->boltOperators[index].output_tensors_name[0];
        if (activationMode == ACTIVATION_RELU) {
            activation = mt_create_operator(name, OT_Relu, 1, 1);
            activation.ps.relu_spec.neg_slope = 0;
        } else if (activationMode == ACTIVATION_RELU6) {
            activation = mt_create_operator(name, OT_Relu6, 1, 1);
        } else {
            UNI_ERROR_LOG("operator location:%d not support %d type activation fusion.\n",
                this->tfliteOperatorIndex, activationMode);
        }
        str_copy(activation.input_tensors_name[0], name, NAME_LEN);
        str_copy(activation.output_tensors_name[0], name, NAME_LEN);
        this->boltOperators.insert(this->boltOperators.begin() + index + 1, activation);
        this->boltOperatorInsertAfter++;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        EltwiseParamSpec eltPs;
        memset(&eltPs, 0, sizeof(eltPs));
        ActivationMode activationMode = ACTIVATION_NULL;
        if (opCode == tflite::BuiltinOperator_ADD) {
            eltPs.elt_mode = ELTWISE_SUM;
            EltwiseSumSpec elt_sum_spec;
            elt_sum_spec.coeff_size = 2;
            for (I32 j = 0; j < elt_sum_spec.coeff_size; j++) {
                elt_sum_spec.coeff_values[j] = 1.0;
            }
            eltPs.elt_sum_spec = elt_sum_spec;
            const auto &tfliteEltwiseOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsAddOptions();
            activationMode =
                getActivationOperatorType(tfliteEltwiseOption->fused_activation_function);
        } else if (opCode == tflite::BuiltinOperator_SUB) {
            eltPs.elt_mode = ELTWISE_SUB;
        } else if (opCode == tflite::BuiltinOperator_MAXIMUM) {
            eltPs.elt_mode = ELTWISE_MAX;
        } else if (opCode == tflite::BuiltinOperator_MINIMUM) {
            eltPs.elt_mode = ELTWISE_MIN;
        } else if (opCode == tflite::BuiltinOperator_DIV) {
            eltPs.elt_mode = ELTWISE_DIV;
        } else if (opCode == tflite::BuiltinOperator_MUL) {
            eltPs.elt_mode = ELTWISE_PROD;
            const auto &tfliteEltwiseOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsMulOptions();
            activationMode =
                getActivationOperatorType(tfliteEltwiseOption->fused_activation_function);
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Eltwise.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        eltPs.activation_type = activationMode;
        curPs.eltwise_spec = eltPs;
        return curPs;
    }

    ParameterSpec adapt_Scale() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        modelWeightOpNum++;
        ScaleParamSpec scalePs;
        memset(&scalePs, 0, sizeof(scalePs));
        scalePs.axis = 1;
        curPs.scale_spec = scalePs;

        if (opCode == tflite::BuiltinOperator_ADD) {
            const auto &addOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsAddOptions();
            if (getActivationOperatorType(addOption->fused_activation_function) != ACTIVATION_NULL) {
                insertActivationOperator(
                    getActivationOperatorType(addOption->fused_activation_function));
            }
        }

        return curPs;
    }

    ParameterSpec adapt_Conv() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        modelWeightOpNum++;
        const int weightIndex = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
        const auto &weightTensor = this->tfliteTensors[weightIndex];

        const auto &weightShape = weightTensor->shape;
        CHECK_REQUIREMENT(weightShape.size() == 4);

        ConvolutionParamSpec convPs;
        memset(&convPs, 0, sizeof(convPs));
        convPs.kernel_h = weightShape[1];
        convPs.kernel_w = weightShape[2];
        convPs.kernel_t = 1;
        convPs.stride_t = 1;
        convPs.padding_before = 0;
        convPs.padding_after = 0;
        convPs.dilatedRate_t = 1;
        if (opCode == tflite::BuiltinOperator_CONV_2D) {
            convPs.num_outputs = weightShape[0];
            convPs.num_outputs_origin = convPs.num_outputs;

            const auto &tfliteConvOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsConv2DOptions();
            convPs.dilatedRate_h = tfliteConvOption->dilation_h_factor;
            convPs.dilatedRate_w = tfliteConvOption->dilation_w_factor;
            convPs.stride_h = tfliteConvOption->stride_h;
            convPs.stride_w = tfliteConvOption->stride_w;
            const auto activationFunc = tfliteConvOption->fused_activation_function;
            if (1 == tfliteConvOption->padding) {  // VALID
                convPs.padding_top = 0;
                convPs.padding_bottom = 0;
                convPs.padding_left = 0;
                convPs.padding_right = 0;
            } else {  // SAME
                const auto &inputTensor =
                    this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
                const auto &inputShape = inputTensor->shape;
                convPs.padding_top = (convPs.kernel_h - 1) / 2;
                convPs.padding_bottom = (convPs.kernel_h - 1) / 2;
                if (convPs.kernel_h % 2 == 0) {
                    convPs.padding_top += 1;
                }
                if (convPs.padding_top != 0 && inputShape[1] % 2 == 0 &&
                    tfliteConvOption->stride_h % 2 == 0) {
                    convPs.padding_top -= 1;
                }
                convPs.padding_left = (convPs.kernel_w - 1) / 2;
                convPs.padding_right = (convPs.kernel_w - 1) / 2;
                if (convPs.kernel_w % 2 == 0) {
                    convPs.padding_left += 1;
                }
                if (convPs.padding_left != 0 && inputShape[2] % 2 == 0 &&
                    tfliteConvOption->stride_w % 2 == 0) {
                    convPs.padding_left -= 1;
                }
            }
            convPs.group = 1;
            convPs.dw_activation_type = ACTIVATION_NULL;
            convPs.pw_activation_type = getActivationOperatorType(activationFunc);
            if (convPs.dilatedRate_h > 1 || convPs.dilatedRate_w > 1) {
                convPs.convolution_type = Convolution_Dilation;
            } else {
                convPs.convolution_type = Convolution_Pointwise;
            }
        } else if (opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
            convPs.num_outputs = weightShape[3];
            convPs.num_outputs_origin = convPs.num_outputs;

            const auto &tfliteConvOption = this->tfliteOperators[this->tfliteOperatorIndex]
                                               ->builtin_options.AsDepthwiseConv2DOptions();
            convPs.dilatedRate_h = tfliteConvOption->dilation_h_factor;
            convPs.dilatedRate_w = tfliteConvOption->dilation_w_factor;
            convPs.stride_h = tfliteConvOption->stride_h;
            convPs.stride_w = tfliteConvOption->stride_w;
            const auto activationFunc = tfliteConvOption->fused_activation_function;

            if (1 == tfliteConvOption->padding) {  // VALID
                convPs.padding_top = 0;
                convPs.padding_bottom = 0;
                convPs.padding_left = 0;
                convPs.padding_right = 0;
            } else {  // SAME
                const auto &inputTensor =
                    this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
                const auto &inputShape = inputTensor->shape;
                convPs.padding_top = (convPs.kernel_h - 1) / 2;
                convPs.padding_bottom = (convPs.kernel_h - 1) / 2;
                if (convPs.kernel_h % 2 == 0) {
                    convPs.padding_top += 1;
                }
                if (convPs.padding_top != 0 && inputShape[1] % 2 == 0 &&
                    tfliteConvOption->stride_h % 2 == 0) {
                    convPs.padding_top -= 1;
                }
                convPs.padding_left = (convPs.kernel_w - 1) / 2;
                convPs.padding_right = (convPs.kernel_w - 1) / 2;
                if (convPs.kernel_w % 2 == 0) {
                    convPs.padding_left += 1;
                }
                if (convPs.padding_left != 0 && inputShape[2] % 2 == 0 &&
                    tfliteConvOption->stride_w % 2 == 0) {
                    convPs.padding_left -= 1;
                }
            }

            convPs.group = convPs.num_outputs;
            // process the situation: when depth_multiplier > 1 && fn == depth_multiplier, depthwise ==> pointwise
            if (tfliteConvOption->depth_multiplier > 1 &&
                tfliteConvOption->depth_multiplier == weightShape[3]) {
                convPs.convolution_type = Convolution_Pointwise;
                convPs.dw_activation_type = ACTIVATION_NULL;
                convPs.pw_activation_type = getActivationOperatorType(activationFunc);
                convPs.group = 1;
            } else {
                convPs.convolution_type = Convolution_Depthwise;
                convPs.dw_activation_type = getActivationOperatorType(activationFunc);
                convPs.pw_activation_type = ACTIVATION_NULL;
            }
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Convolution.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        curPs.conv_spec = convPs;
        return curPs;
    }

    ParameterSpec adapt_Reduction() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ReductionParamSpec reductionPs;
        memset(&curPs, 0, sizeof(reductionPs));
        if (opCode == tflite::BuiltinOperator_MEAN) {
            reductionPs.reduction_mode = REDUCTION_MEAN;
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Reduction.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }

        const auto &inputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
        const auto &inputShape = inputTensor->shape;
        const auto &axisTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &axisData = tfliteModelBuffer[axisTensor->buffer]->data;
        reductionPs.axes_num = axisData.size() / sizeof(int);
        auto axisPtr = reinterpret_cast<const int32_t *>(axisData.data());
        memcpy(reductionPs.axes, axisPtr, axisData.size());
        if (this->weightFormat == DF_NHWC) {
            for (int i = 0; i < reductionPs.axes_num; i++) {
                reductionPs.axes[i] = NHWCAxisToNCHWAxis(reductionPs.axes[i], inputShape.size());
            }
        }
        reductionPs.coeff = 1;
        reductionPs.keep_dim = false;
        curPs.reduction_spec = reductionPs;
        return curPs;
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        PoolingParamSpec poolingPs;
        memset(&poolingPs, 0, sizeof(poolingPs));
        poolingPs.kernel_t = 1;
        poolingPs.stride_t = 1;
        poolingPs.padding_before = 0;
        poolingPs.padding_after = 0;
        poolingPs.padding_top = 0;
        poolingPs.padding_bottom = 0;
        poolingPs.padding_left = 0;
        poolingPs.padding_right = 0;
        poolingPs.rm = CEIL;

        const auto &inputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
        const auto &inputShape = inputTensor->shape;
        CHECK_REQUIREMENT(inputShape.size() == 4);
        if (opCode == tflite::BuiltinOperator_MEAN) {  // Interpret as global pooling
            const auto &axisTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
            const auto &axisData = tfliteModelBuffer[axisTensor->buffer]->data;
            auto axisPtr = reinterpret_cast<const int32_t *>(axisData.data());
            CHECK_REQUIREMENT(1 == axisPtr[0] && 2 == axisPtr[1]);
            poolingPs.mode = POOLING_MEAN;
            poolingPs.kernel_h = 0;
            poolingPs.kernel_w = 0;
            poolingPs.stride_h = 1;
            poolingPs.stride_w = 1;
        } else {
            const auto &tflitePoolOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsPool2DOptions();
            poolingPs.kernel_h = tflitePoolOption->filter_height;
            poolingPs.kernel_w = tflitePoolOption->filter_width;
            poolingPs.stride_h = tflitePoolOption->stride_h;
            poolingPs.stride_w = tflitePoolOption->stride_w;
            int tfPaddingRoundMode = tflitePoolOption->padding;
            if (tfPaddingRoundMode == 0) {
                poolingPs.rm = TF_SAME;

                int oLength = (inputShape[2] + poolingPs.stride_w - 1) / poolingPs.stride_w;
                int padLength = UNI_MAX(
                    (oLength - 1) * poolingPs.stride_w + poolingPs.kernel_w - inputShape[2], 0);
                poolingPs.padding_left = padLength / 2;
                poolingPs.padding_right = padLength - poolingPs.padding_left;

                oLength = (inputShape[1] + poolingPs.stride_h - 1) / poolingPs.stride_h;
                padLength = UNI_MAX(
                    (oLength - 1) * poolingPs.stride_h + poolingPs.kernel_h - inputShape[1], 0);
                poolingPs.padding_top = padLength / 2;
                poolingPs.padding_bottom = padLength - poolingPs.padding_top;
            } else if (tfPaddingRoundMode == 1) {
                poolingPs.rm = TF_VALID;
            } else {
                UNI_ERROR_LOG("can not process operator location:%d Pooling round mode.\n",
                    this->tfliteOperatorIndex);
            }
            if (opCode == tflite::BuiltinOperator_MAX_POOL_2D) {
                poolingPs.mode = POOLING_MAX;
            } else if (opCode == tflite::BuiltinOperator_AVERAGE_POOL_2D) {
                poolingPs.mode = POOLING_MEAN;
            }
            insertActivationOperator(
                getActivationOperatorType(tflitePoolOption->fused_activation_function));
        }
        curPs.pooling_spec = poolingPs;
        return curPs;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        const auto &shapeTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &shapeData = tfliteModelBuffer[shapeTensor->buffer]->data;
        ReshapeParamSpec reshapePs;
        memset(&reshapePs, 0, sizeof(reshapePs));
        reshapePs.shape_size = (int)(shapeData.size() / sizeof(int));
        auto reshapeDimPtr = reinterpret_cast<const int32_t *>(shapeData.data());
        for (int iter = 0; iter < reshapePs.shape_size; iter++) {
            int axis = iter;
            if (this->weightFormat == DF_NHWC) {
                axis = NHWCAxisToNCHWAxis(iter, reshapePs.shape_size);
            }
            reshapePs.shape_dims[axis] = reshapeDimPtr[iter];
        }
        reshapePs.axis = 8;
        reshapePs.num_axes = -1;
        curPs.reshape_spec = reshapePs;
        return curPs;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TransposeParamSpec transPs;
        memset(&transPs, 0, sizeof(transPs));
        const auto &dimsTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &dimsData = tfliteModelBuffer[dimsTensor->buffer]->data;
        CHECK_REQUIREMENT((dimsTensor->shape[0]) == (int)(dimsData.size() / sizeof(int)));
        transPs.trans_size = dimsTensor->shape[0];
        auto dims = reinterpret_cast<const int32_t *>(dimsData.data());
        for (U32 i = 0; i < transPs.trans_size; i++) {
            if (this->weightFormat == DF_NHWC) {
                transPs.trans_dims[i] = NHWCAxisToNCHWAxis(dims[i], transPs.trans_size);
            } else {
                transPs.trans_dims[i] = dims[i];
            }
        }
        curPs.transpose_spec = transPs;
        return curPs;
    }

    ParameterSpec adapt_TfSlice() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TfSliceParamSpec tfSlicePs;
        memset(&tfSlicePs, 0, sizeof(tfSlicePs));
        if (opCode == tflite::BuiltinOperator_STRIDED_SLICE) {
            const auto &stridedSliceOption = this->tfliteOperators[this->tfliteOperatorIndex]
                                                 ->builtin_options.AsStridedSliceOptions();
            const auto &beginTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
            tfSlicePs.dim_size = beginTensor->shape[0];
            auto beginData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[beginTensor->buffer]->data).data());
            memcpy(tfSlicePs.begin, beginData, sizeof(int) * tfSlicePs.dim_size);
            const auto &endTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[2]];
            auto endData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[endTensor->buffer]->data).data());
            memcpy(tfSlicePs.end, endData, sizeof(int) * tfSlicePs.dim_size);
            const auto &stridesTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[3]];
            auto stridesData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[stridesTensor->buffer]->data).data());
            memcpy(tfSlicePs.strides, stridesData, sizeof(int) * tfSlicePs.dim_size);
            bitsToCharArray(
                stridedSliceOption->begin_mask, tfSlicePs.begin_mask, tfSlicePs.dim_size);
            bitsToCharArray(stridedSliceOption->end_mask, tfSlicePs.end_mask, tfSlicePs.dim_size);
            bitsToCharArray(
                stridedSliceOption->ellipsis_mask, tfSlicePs.ellipsis_mask, tfSlicePs.dim_size);
            bitsToCharArray(
                stridedSliceOption->new_axis_mask, tfSlicePs.new_axis_mask, tfSlicePs.dim_size);
            bitsToCharArray(stridedSliceOption->shrink_axis_mask, tfSlicePs.shrink_axis_mask,
                tfSlicePs.dim_size);
        } else {
            const auto &beginTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
            tfSlicePs.dim_size = beginTensor->shape[0];
            auto beginData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[beginTensor->buffer]->data).data());
            memcpy(tfSlicePs.begin, beginData, sizeof(int) * tfSlicePs.dim_size);
            const auto &sizeTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[2]];
            auto sizeData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[sizeTensor->buffer]->data).data());
            for (U32 i = 0; i < tfSlicePs.dim_size; i++) {
                tfSlicePs.end[i] = tfSlicePs.begin[i] + sizeData[i];
                tfSlicePs.strides[i] = 1;
            }
            memset(tfSlicePs.begin_mask, 0, sizeof(char) * tfSlicePs.dim_size);
            memset(tfSlicePs.end_mask, 0, sizeof(char) * tfSlicePs.dim_size);
            memset(tfSlicePs.ellipsis_mask, 0, sizeof(char) * tfSlicePs.dim_size);
            memset(tfSlicePs.new_axis_mask, 0, sizeof(char) * tfSlicePs.dim_size);
            memset(tfSlicePs.shrink_axis_mask, 0, sizeof(char) * tfSlicePs.dim_size);
        }
        if (this->weightFormat == DF_NHWC) {
            shiftRight<int>(tfSlicePs.begin, tfSlicePs.dim_size, 1, tfSlicePs.dim_size - 1);
            shiftRight<int>(tfSlicePs.end, tfSlicePs.dim_size, 1, tfSlicePs.dim_size - 1);
            shiftRight<int>(tfSlicePs.strides, tfSlicePs.dim_size, 1, tfSlicePs.dim_size - 1);
            shiftRight<char>(tfSlicePs.begin_mask, tfSlicePs.dim_size, 1, tfSlicePs.dim_size - 1);
            shiftRight<char>(tfSlicePs.end_mask, tfSlicePs.dim_size, 1, tfSlicePs.dim_size - 1);
            shiftRight<char>(tfSlicePs.ellipsis_mask, tfSlicePs.dim_size, 1, tfSlicePs.dim_size - 1);
            shiftRight<char>(tfSlicePs.new_axis_mask, tfSlicePs.dim_size, 1, tfSlicePs.dim_size - 1);
            shiftRight<char>(
                tfSlicePs.shrink_axis_mask, tfSlicePs.dim_size, 1, tfSlicePs.dim_size - 1);
        }
        curPs.tfslice_spec = tfSlicePs;
        return curPs;
    }

    ParameterSpec adapt_MatMul() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        MatMulParamSpec matmulPs;
        memset(&matmulPs, 0, sizeof(matmulPs));
        matmulPs.transpose_a = false;
        matmulPs.transpose_b = false;
        std::vector<int> weightInputIndex = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        if (weightInputIndex.size() == 2 && weightInputIndex[0] == 0 && weightInputIndex[1] == 2) {
            matmulPs.transpose_b = true;

            OperatorSpec sharedWeight = mt_create_operator(
                this->boltOperators[this->boltOperatorIndex].input_tensors_name[0], OT_SharedWeight,
                0, 1);
            str_copy(sharedWeight.output_tensors_name[0],
                this->boltOperators[this->boltOperatorIndex].input_tensors_name[0], NAME_LEN);
            const auto &inputWeightTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
            std::vector<int> inputWeightShape(inputWeightTensor->shape);
            SharedWeightParamSpec sharedWeightPs;
            sharedWeightPs.desc =
                tensor2df(DT_F32, DF_NORMAL, inputWeightShape[0], inputWeightShape[1]);
            WeightSpec weightSpec =
                mt_create_weight(this->boltOperators[this->boltOperatorIndex].input_tensors_name[0],
                    DT_F32, tensorNumBytes(sharedWeightPs.desc), 0, 0);
            std::vector<float> inputWeightFloats = transformTfliteTensorToVector(inputWeightTensor);
            memcpy(weightSpec.weight, &(inputWeightFloats[0]), tensorNumBytes(sharedWeightPs.desc));
            this->boltSharedWeights.push_back(weightSpec);
            sharedWeight.ps.shared_weight_spec = sharedWeightPs;
            this->boltOperators.insert(
                this->boltOperators.begin() + this->boltOperatorIndex, sharedWeight);
            this->boltOperatorInsertBefore++;
            this->modelWeightOpNum++;
        }
        curPs.matmul_spec = matmulPs;
        return curPs;
    }

    ParameterSpec adapt_Fc() override
    {
        modelWeightOpNum++;
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        FullyConnectedParamSpec ips;
        memset(&ips, 0, sizeof(ips));
        const int index = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
        const auto &tensor = this->tfliteTensors[index];
        I32 size = tfliteModelBuffer[tensor->buffer]->data.size();
        CHECK_REQUIREMENT(size != 0);
        const auto &weightShape = tensor->shape;
        ips.num_outputs = weightShape[0];
        ips.num_slices = 1;
        ips.slice_point[0] = ips.num_outputs;
        curPs.fc_spec = ips;
        const auto &tfliteFullyConnectedOption = this->tfliteOperators[this->tfliteOperatorIndex]
                                                     ->builtin_options.AsFullyConnectedOptions();
        insertActivationOperator(
            getActivationOperatorType(tfliteFullyConnectedOption->fused_activation_function));
        return curPs;
    }

    ParameterSpec adapt_Concat() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ConcatParamSpec concatPs;
        memset(&concatPs, 0, sizeof(concatPs));
        std::vector<int> pinnedInput;
        if (this->opCode == tflite::BuiltinOperator_CONCATENATION) {
            const auto &tfliteConcatOption = this->tfliteOperators[this->tfliteOperatorIndex]
                                                 ->builtin_options.AsConcatenationOptions();
            insertActivationOperator(
                getActivationOperatorType(tfliteConcatOption->fused_activation_function));
            concatPs.axis = tfliteConcatOption->axis;
            pinnedInput = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        } else {
            const auto &tflitePackOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsPackOptions();
            concatPs.axis = tflitePackOption->axis;
            int id = tflitePackOption->values_count - 1;
            pinnedInput.push_back(id);
        }
        for (U32 i = 0; i < pinnedInput.size(); i++) {
            int id = pinnedInput[i];
            std::string tmpOpName =
                std::string(this->boltOperators[this->boltOperatorIndex].input_tensors_name[id]);
            if (sharedWeightName.find(tmpOpName) == sharedWeightName.end()) {
                sharedWeightName[tmpOpName] = 1;
            } else {
                continue;
            }
            OperatorSpec sharedWeight = mt_create_operator(
                this->boltOperators[this->boltOperatorIndex].input_tensors_name[id],
                OT_SharedWeight, 0, 1);
            str_copy(sharedWeight.output_tensors_name[0],
                this->boltOperators[this->boltOperatorIndex].input_tensors_name[id], NAME_LEN);
            SharedWeightParamSpec sharedWeightPs;
            const auto &weightTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[id]];

            sharedWeightPs.desc.nDims = weightTensor->shape.size();
            sharedWeightPs.desc.dt = TfliteTensorType2BoltDataType(weightTensor->type);
            sharedWeightPs.desc.df = getTensorDefaultDataFormat(sharedWeightPs.desc.nDims);
            if (sharedWeightPs.desc.nDims == 4 && this->weightFormat == DF_NHWC) {
                sharedWeightPs.desc.df = DF_NHWC;
            }
            for (U32 j = 0; j < sharedWeightPs.desc.nDims; j++) {
                sharedWeightPs.desc.dims[sharedWeightPs.desc.nDims - 1 - j] = weightTensor->shape[j];
            }

            auto weightData =
                reinterpret_cast<U8 *>((tfliteModelBuffer[weightTensor->buffer]->data).data());
            WeightSpec weightSpec = mt_create_weight(
                this->boltOperators[this->boltOperatorIndex].input_tensors_name[id],
                sharedWeightPs.desc.dt, tensorNumBytes(sharedWeightPs.desc), 0, 0);
            if (sharedWeightPs.desc.df == DF_NHWC) {
                TensorDesc nchwDesc = tensor4df(sharedWeightPs.desc.dt, DF_NCHW,
                    weightTensor->shape[0], weightTensor->shape[3], weightTensor->shape[1],
                    weightTensor->shape[2]);
                transformToNCHW(sharedWeightPs.desc, weightData, nchwDesc, weightSpec.weight);
                sharedWeightPs.desc = nchwDesc;
            } else {
                memcpy(weightSpec.weight, weightData, tensorNumBytes(sharedWeightPs.desc));
            }
            this->boltSharedWeights.push_back(weightSpec);
            sharedWeight.ps.shared_weight_spec = sharedWeightPs;
            this->boltOperators.insert(
                this->boltOperators.begin() + this->boltOperatorIndex, sharedWeight);
            this->boltOperatorInsertBefore++;
            this->modelWeightOpNum++;
        }
        const auto &outputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->outputs[0]];
        const auto &outputShape = outputTensor->shape;
        if (this->weightFormat == DF_NHWC) {
            concatPs.axis = NHWCAxisToNCHWAxis(concatPs.axis, outputShape.size());
        }
        curPs.concat_spec = concatPs;
        return curPs;
    }

    ParameterSpec adapt_Softmax() override
    {
        const auto &tfliteSoftmaxOption =
            this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsSoftmaxOptions();
        CHECK_REQUIREMENT(1 == tfliteSoftmaxOption->beta);

        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        SoftmaxParamSpec softmaxPs;
        memset(&softmaxPs, 0, sizeof(softmaxPs));
        softmaxPs.axis = -1;

        const auto &inputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
        const auto &inputShape = inputTensor->shape;
        if (this->weightFormat == DF_NHWC) {
            softmaxPs.axis = NHWCAxisToNCHWAxis(softmaxPs.axis, inputShape.size());
        }
        curPs.softmax_spec = softmaxPs;
        return curPs;
    }

    ParameterSpec adapt_Resize() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ResizeParamSpec resizePs;
        memset(&resizePs, 0, sizeof(resizePs));
        const auto &dimsTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &dimsData = tfliteModelBuffer[dimsTensor->buffer]->data;
        CHECK_REQUIREMENT((dimsTensor->shape[0]) == 2);
        auto dims = reinterpret_cast<const int32_t *>(dimsData.data());
        resizePs.sizes[0] = dims[0];
        resizePs.sizes[1] = dims[1];
        resizePs.num_sizes = 2;
        resizePs.num_scales = 0;
        if (this->opCode == tflite::BuiltinOperator_RESIZE_BILINEAR) {
            resizePs.mode = LINEAR;
        } else if (this->opCode == tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR) {
            resizePs.mode = NEAREST;
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Resize.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        curPs.resize_spec = resizePs;
        return curPs;
    }

    ParameterSpec adapt_Clip() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ClipParamSpec clipPs;
        memset(&clipPs, 0, sizeof(clipPs));
        const auto &clipTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &clipData = tfliteModelBuffer[clipTensor->buffer]->data;
        if (opCode == tflite::BuiltinOperator_MINIMUM) {
            clipPs.max = clipData[0];
            clipPs.min = std::numeric_limits<float>::min();
        } else if (opCode == tflite::BuiltinOperator_MAXIMUM) {
            clipPs.max = std::numeric_limits<float>::max();
            clipPs.min = clipData[0];
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Clip.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        curPs.clip_spec = clipPs;
        return curPs;
    }

    ParameterSpec adapt_Deconvolution() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        modelWeightOpNum++;
        const int weightIndex = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
        const auto &weightTensor = this->tfliteTensors[weightIndex];

        const auto &weightShape = weightTensor->shape;
        CHECK_REQUIREMENT(weightShape.size() == 4);

        ConvolutionParamSpec convPs;
        memset(&convPs, 0, sizeof(convPs));
        convPs.kernel_t = 1;
        convPs.kernel_h = weightShape[1];
        convPs.kernel_w = weightShape[2];
        convPs.num_outputs = weightShape[0];
        convPs.num_outputs_origin = convPs.num_outputs;

        const auto &tfliteDeConvOption =
            this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsTransposeConvOptions();
        convPs.stride_t = 1;
        convPs.stride_h = tfliteDeConvOption->stride_h;
        convPs.stride_w = tfliteDeConvOption->stride_w;
        convPs.group = 1;

        convPs.dilatedRate_t = 1;
        convPs.dilatedRate_h = 1;
        convPs.dilatedRate_w = 1;
        convPs.convolution_type = Convolution_Deconvolution;
        convPs.dw_activation_type = ACTIVATION_NULL;
        convPs.pw_activation_type = ACTIVATION_NULL;

        convPs.padding_before = 0;
        convPs.padding_after = 0;
        if (tfliteDeConvOption->padding == 1) {
            convPs.rm = TF_VALID;
            convPs.padding_top = 0;
            convPs.padding_bottom = 0;
            convPs.padding_left = 0;
            convPs.padding_right = 0;
        } else {
            convPs.rm = TF_SAME;
            if (convPs.kernel_h < convPs.stride_h) {
                convPs.padding_top = 0;
                convPs.padding_bottom = 0;
            } else {
                convPs.padding_top = (convPs.kernel_h - convPs.stride_h) / 2;
                convPs.padding_bottom = convPs.kernel_h - convPs.stride_h - convPs.padding_top;
            }
            if (convPs.kernel_w < convPs.stride_w) {
                convPs.padding_left = 0;
                convPs.padding_right = 0;
            } else {
                convPs.padding_left = (convPs.kernel_w - convPs.stride_w) / 2;
                convPs.padding_right = convPs.kernel_w - convPs.stride_w - convPs.padding_left;
            }
        }

        curPs.conv_spec = convPs;
        return curPs;
    }

    ParameterSpec adapt_Power() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        PowerParamSpec powerPs;
        memset(&powerPs, 0, sizeof(powerPs));
        powerPs.scale = 1;
        powerPs.shift = 0;
        powerPs.power = 1;
        float weight = 0;
        std::vector<int> weightInputIndex = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        if (weightInputIndex.size() > 0) {
            const auto &weightTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]
                                        ->inputs[weightInputIndex[0]]];
            weight = transformTfliteTensorToVector(weightTensor)[0];
        }
        if (opCode == tflite::BuiltinOperator_SQRT) {
            powerPs.power = 0.5;
        } else if (opCode == tflite::BuiltinOperator_POW) {
            powerPs.power = weight;
        } else if (opCode == tflite::BuiltinOperator_ADD) {
            powerPs.shift = weight;
        } else if (opCode == tflite::BuiltinOperator_SUB) {
            powerPs.shift = weight * -1;
        } else if (opCode == tflite::BuiltinOperator_MUL) {
            powerPs.scale = weight;
        } else if (opCode == tflite::BuiltinOperator_DIV) {
            powerPs.scale = 1.0 / weight;
        } else if (opCode == tflite::BuiltinOperator_NEG) {
            powerPs.scale = -1.0;
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Power.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        curPs.power_spec = powerPs;

        if (opCode == tflite::BuiltinOperator_ADD) {
            const auto &addOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsAddOptions();
            if (getActivationOperatorType(addOption->fused_activation_function) != ACTIVATION_NULL) {
                insertActivationOperator(
                    getActivationOperatorType(addOption->fused_activation_function));
            }
        }

        return curPs;
    }

    ParameterSpec adapt_Pad() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        PadParamSpec padPs;
        memset(&padPs, 0, sizeof(padPs));
        const auto &beginTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        auto beginData = reinterpret_cast<const int32_t *>(
            (tfliteModelBuffer[beginTensor->buffer]->data).data());
        padPs.before = 0;
        padPs.after = 0;
        padPs.top = beginData[2];
        padPs.bottom = beginData[3];
        padPs.left = beginData[4];
        padPs.right = beginData[5];
        padPs.constant_value = 0;
        if (this->opCode == tflite::BuiltinOperator_PAD) {
            padPs.pad_mode = Pad_Constant;
        } else {
            padPs.pad_mode = Pad_Reflect;
        }
        curPs.pad_spec = padPs;
        return curPs;
    }

    ParameterSpec adapt_Relu() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        ReLUParamSpec reluPs;
        memset(&reluPs, 0, sizeof(reluPs));
        if (this->opCode == tflite::BuiltinOperator_RELU) {
            reluPs.neg_slope = 0;
        } else {
            const auto &tfliteLeakyReluOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsLeakyReluOptions();
            reluPs.neg_slope = tfliteLeakyReluOption->alpha;
        }
        curPs.relu_spec = reluPs;
        return curPs;
    }

    ParameterSpec adapt_Squeeze() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        SqueezeParamSpec squeezePs;
        memset(&squeezePs, 0, sizeof(squeezePs));
        const auto &tfliteSqueezeOption =
            this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsSqueezeOptions();
        squeezePs.axes_num = tfliteSqueezeOption->squeeze_dims.size();
        const auto &inputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
        const auto &inputShape = inputTensor->shape;
        for (int i = 0; i < squeezePs.axes_num; i++) {
            if (this->weightFormat == DF_NHWC) {
                squeezePs.axes[i] =
                    NHWCAxisToNCHWAxis(tfliteSqueezeOption->squeeze_dims[i], inputShape.size());
            } else {
                squeezePs.axes[i] = tfliteSqueezeOption->squeeze_dims[i];
            }
        }
        curPs.squeeze_spec = squeezePs;
        return curPs;
    }

    ParameterSpec adapt_Unsqueeze() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        UnsqueezeParamSpec unsqueezePs;
        memset(&unsqueezePs, 0, sizeof(unsqueezePs));
        const auto &weightTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        auto weightData = transformTfliteTensorToVector(weightTensor);
        const auto &inputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
        const auto &inputShape = inputTensor->shape;
        if (this->weightFormat == DF_NHWC) {
            unsqueezePs.axes[0] = NHWCAxisToNCHWAxis((int)weightData[0], inputShape.size());
        } else {
            unsqueezePs.axes[0] = weightData[0];
        }
        unsqueezePs.axes_num = 1;
        curPs.unsqueeze_spec = unsqueezePs;
        return curPs;
    }

    ParameterSpec adapt_TopK() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        TopKParamSpec p;
        memset(&p, 0, sizeof(p));
        p.axis = 0;
        p.largest = 1;
        p.sorted = 1;
        const auto &weightTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        auto weightData = transformTfliteTensorToVector(weightTensor);
        p.topk = weightData[0];
        curPs.topk_spec = p;
        return curPs;
    }

    ParameterSpec adapt_Embedding() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        EmbedParamSpec p;
        memset(&p, 0, sizeof(p));
        p.input_dim = -1;
        p.num_output = -1;
        p.bias_term = false;
        p.transpose = false;
        curPs.embed_spec = p;
        return curPs;
    }

    ParameterSpec adapt_PRelu() override
    {
        modelWeightOpNum++;
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        return curPs;
    }

    ParameterSpec adapt_SpaceToBatchNd() override
    {
        modelWeightOpNum++;
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        return curPs;
    }

    ParameterSpec adapt_BatchToSpaceNd() override
    {
        modelWeightOpNum++;
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        return curPs;
    }

    ParameterSpec adapt_Slice() override
    {
        ParameterSpec curPs;
        memset(&curPs, 0, sizeof(curPs));
        SliceParamSpec slice_ps;
        memset(&slice_ps, 0, sizeof(slice_ps));
        if (this->opCode == tflite::BuiltinOperator_SPLIT) {
            const auto &tfliteSplit =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsSplitOptions();
            const auto &weightTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
            const auto &inputTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
            const auto &inputShape = inputTensor->shape;
            std::vector<float> axisData = transformTfliteTensorToVector(weightTensor);
            if (this->weightFormat == DF_NHWC) {
                slice_ps.axis = NHWCAxisToNCHWAxis((int)axisData[0], inputShape.size());
            } else {
                slice_ps.axis = axisData[0];
            }
            slice_ps.slice_size = tfliteSplit->num_splits - 1;
            memset(slice_ps.slice_points, 0, slice_ps.slice_size * sizeof(I32));
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Slice.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        curPs.slice_spec = slice_ps;
        return curPs;
    }

public:
    std::map<OperatorType, int> modifiedInputsOp{{OT_Conv, 1}, {OT_Reshape, 1}, {OT_Resize, 1},
        {OT_Transpose, 1}, {OT_FC, 1}, {OT_Slice, 1}, {OT_Scale, 1}, {OT_Pooling, 1}, {OT_Clip, 1},
        {OT_Deconvolution, 1}, {OT_SqDiff, 1}, {OT_Reduction, 1}, {OT_Pad, 1}, {OT_Power, 1},
        {OT_TfSlice, 1}, {OT_SpaceToBatchNd, 1}, {OT_BatchToSpaceNd, 1}, {OT_MatMul, 2},
        {OT_PRelu, 1}};

private:
    DataFormat weightFormat;
    std::vector<std::unique_ptr<tflite::BufferT>> tfliteModelBuffer;
    std::vector<std::unique_ptr<tflite::OperatorCodeT>> tfliteOpSet;
    std::vector<std::unique_ptr<tflite::OperatorT>> tfliteOperators;
    std::vector<std::unique_ptr<tflite::TensorT>> tfliteTensors;
    std::vector<int> inputs;
    std::vector<int> outputs;
    U32 tfliteOperatorIndex;
    tflite::BuiltinOperator opCode;
    int modelWeightOpNum;
    std::string modelName;

    U32 boltOperatorIndex;
    U32 boltOperatorInsertBefore;  // 1 tflite operator -> (before + 1 + after) bolt operators
    U32 boltOperatorInsertAfter;
    std::map<std::string, int> boltOperatorNameMap;
    std::vector<OperatorSpec> boltOperators;
    std::vector<WeightSpec> boltSharedWeights;
    std::map<std::string, int> sharedWeightName;
};
#endif
