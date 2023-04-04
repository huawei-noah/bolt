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
#include "model_adaptee.h"

#include <tensorflow/lite/schema/schema_generated.h>
#include <google/protobuf/message.h>
#include "tensor_transpose.h"

class TfliteAdaptee : public ModelAdaptee {
public:
    TfliteAdaptee()
    {
        this->weightFormat = DF_NHWC;
    }

    ~TfliteAdaptee()
    {
        google::protobuf::ShutdownProtobufLibrary();
    }

protected:
    std::vector<int> getOperatorTensorInputIndex(int operatorIndex)
    {
        std::vector<int> index;
        for (U32 i = 0; i < this->tfliteOperators[operatorIndex]->inputs.size(); i++) {
            int tensorId = this->tfliteOperators[operatorIndex]->inputs[i];
            if (tensorId >= 0 &&
                this->tfliteModelBuffer[this->tfliteTensors[tensorId]->buffer]->data.size() == 0) {
                index.push_back(i);
            }
        }
        return index;
    }

    std::vector<int> getOperatorWeightInputIndex(int operatorIndex)
    {
        std::vector<int> index;
        for (U32 i = 0; i < this->tfliteOperators[operatorIndex]->inputs.size(); i++) {
            int tensorId = this->tfliteOperators[operatorIndex]->inputs[i];
            if (tensorId >= 0 &&
                this->tfliteModelBuffer[this->tfliteTensors[tensorId]->buffer]->data.size() > 0) {
                index.push_back(i);
            }
        }
        return index;
    }

    bool is_multi_dim(const std::unique_ptr<tflite::TensorT> &tensor)
    {
        std::vector<int> shape(tensor->shape);
        int multidim = 0;
        for (U32 idx = 0; idx < shape.size(); ++idx) {
            if (shape[idx] >= 1) {
                ++multidim;
            }
        }
        return (multidim > 1);
    }

    OperatorType convert_tflite_type(tflite::BuiltinOperator tfliteOperatorType)
    {
        std::map<tflite::BuiltinOperator, OperatorType> operatorMap = {
            {tflite::BuiltinOperator_CONCATENATION, OT_Concat},
            {tflite::BuiltinOperator_PACK, OT_Concat},
            {tflite::BuiltinOperator_CONV_2D, OT_Conv},
            {tflite::BuiltinOperator_DEPTHWISE_CONV_2D, OT_Conv},
            {tflite::BuiltinOperator_LOGISTIC, OT_Sigmoid},
            {tflite::BuiltinOperator_MAX_POOL_2D, OT_Pooling},
            {tflite::BuiltinOperator_AVERAGE_POOL_2D, OT_Pooling},
            {tflite::BuiltinOperator_RESHAPE, OT_Reshape},
            {tflite::BuiltinOperator_RESIZE_BILINEAR, OT_Resize},
            {tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, OT_Resize},
            {tflite::BuiltinOperator_SOFTMAX, OT_Softmax},
            {tflite::BuiltinOperator_TRANSPOSE, OT_Transpose},
            {tflite::BuiltinOperator_SLICE, OT_TfSlice},
            {tflite::BuiltinOperator_STRIDED_SLICE, OT_TfSlice},
            {tflite::BuiltinOperator_RELU, OT_Relu},
            {tflite::BuiltinOperator_LEAKY_RELU, OT_Relu},
            {tflite::BuiltinOperator_RELU6, OT_Relu6},
            {tflite::BuiltinOperator_TANH, OT_TanH},
            {tflite::BuiltinOperator_MINIMUM, OT_Clip},
            {tflite::BuiltinOperator_TRANSPOSE_CONV, OT_Deconvolution},
            {tflite::BuiltinOperator_SQUARED_DIFFERENCE, OT_SqDiff},
            {tflite::BuiltinOperator_SQRT, OT_Power},
            {tflite::BuiltinOperator_POW, OT_Power},
            {tflite::BuiltinOperator_L2_NORMALIZATION, OT_L2Norm},
            {tflite::BuiltinOperator_PAD, OT_Pad},
            {tflite::BuiltinOperator_MIRROR_PAD, OT_Pad},
            {tflite::BuiltinOperator_HARD_SWISH, OT_HSwish},
            {tflite::BuiltinOperator_SHAPE, OT_Shape},
            {tflite::BuiltinOperator_SQUEEZE, OT_Squeeze},
            {tflite::BuiltinOperator_EXPAND_DIMS, OT_Unsqueeze},
            {tflite::BuiltinOperator_NEG, OT_Power},
            {tflite::BuiltinOperator_TOPK_V2, OT_TopK},
            {tflite::BuiltinOperator_GATHER, OT_Gather},
            {tflite::BuiltinOperator_GATHER_ND, OT_Gather},
            {tflite::BuiltinOperator_PRELU, OT_PRelu},
            {tflite::BuiltinOperator_SPACE_TO_BATCH_ND, OT_SpaceToBatchNd},
            {tflite::BuiltinOperator_BATCH_TO_SPACE_ND, OT_BatchToSpaceNd},
            {tflite::BuiltinOperator_ABS, OT_Abs},
            {tflite::BuiltinOperator_QUANTIZE, OT_Slice},
            {tflite::BuiltinOperator_FAKE_QUANT, OT_Slice},
            {tflite::BuiltinOperator_SPLIT, OT_Slice},
            {tflite::BuiltinOperator_EXP, OT_Exp},
            {tflite::BuiltinOperator_EQUAL, OT_Check},
            {tflite::BuiltinOperator_NOT_EQUAL, OT_Check},
            {tflite::BuiltinOperator_CAST, OT_Cast},
            {tflite::BuiltinOperator_SUM, OT_Reduction},
            {tflite::BuiltinOperator_REDUCE_MAX, OT_Reduction},
            {tflite::BuiltinOperator_SELECT, OT_Select},
            {tflite::BuiltinOperator_RSQRT, OT_Power},
        };
        if (operatorMap.find(tfliteOperatorType) != operatorMap.end()) {
            return operatorMap[tfliteOperatorType];
        }
        std::vector<int> weightInputIndex = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        if (tfliteOperatorType == tflite::BuiltinOperator_ADD ||
            tfliteOperatorType == tflite::BuiltinOperator_MUL ||
            tfliteOperatorType == tflite::BuiltinOperator_DIV ||
            tfliteOperatorType == tflite::BuiltinOperator_SUB) {
            if (weightInputIndex.size() > 0) {
                const auto &tensor =
                    this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]
                                            ->inputs[weightInputIndex[0]]];
                if (this->tfliteModelBuffer[tensor->buffer]->data.size() == sizeof(float)) {
                    return OT_Power;
                } else if (is_multi_dim(tensor)) {
                    return OT_Eltwise;
                } else {
                    return OT_Scale;
                }
            } else {
                return OT_Eltwise;
            }
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

        char *buffer = (char *)mt_malloc(size);
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

        mt_free(buffer);
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

    TensorDesc getDescFromTp(const std::unique_ptr<tflite::TensorT> &tensor, bool NCHW2NHWC = false)
    {
        TensorDesc desc = tensor0d();
        desc.dt = TfliteTensorType2BoltDataType(tensor->type);
        std::vector<int> inputShape(tensor->shape);
        desc.nDims = inputShape.size();
        desc.df = getTensorDefaultDataFormat(desc.nDims);
        if (NCHW2NHWC) {
            if (this->weightFormat == DF_NHWC) {
                shiftRight<int>(inputShape.data(), inputShape.size(), 1, inputShape.size() - 1);
            }
        }
        for (U32 j = 0; j < desc.nDims; j++) {
            desc.dims[desc.nDims - 1 - j] = inputShape[j];
        }
        if (desc.nDims == 0) {
            auto v = transformTfliteTensorToVector(tensor);
            if (v.size() > 0) {
                desc.nDims = 1;
                desc.df = DF_SCALAR;
                desc.dims[0] = v.size();
            }
        }
        return desc;
    }

    EE adapt_operators(ModelSpec *ms) override
    {
        this->weightNumber = 0;
        EE ret = SUCCESS;
        ms->dt = DT_F32;
        str_copy(ms->model_name, modelName.c_str(), modelName.length());
        ms->num_inputs = inputs.size();
        ms->input_names = (I8 **)mt_malloc(ms->num_inputs * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_malloc(sizeof(TensorDesc) * ms->num_inputs);
        for (I32 i = 0; i < ms->num_inputs; i++) {
            const int inputIdx = inputs[i];
            const auto &inputTensor = this->tfliteTensors[inputIdx];
            ms->input_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(ms->input_names[i], (inputTensor->name).c_str(), (inputTensor->name).length());
            ms->input_dims[i] = getDescFromTp(inputTensor, true);
        }
        ms->num_outputs = outputs.size();
        ms->output_names = (I8 **)mt_malloc(ms->num_outputs * sizeof(I8 *));
        for (I32 i = 0; i < ms->num_outputs; i++) {
            const int outputIdx = outputs[i];
            const auto &outputTensor = this->tfliteTensors[outputIdx];
            ms->output_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
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
            convert_tflite_type(this->opCode);
            this->boltOperators[this->boltOperatorIndex].type = convert_tflite_type(this->opCode);
            this->boltOperators[this->boltOperatorIndex].num_inputs =
                (modifiedInputsOp.find(this->boltOperators[this->boltOperatorIndex].type) ==
                    modifiedInputsOp.end())
                ? this->tfliteOperators[this->tfliteOperatorIndex]->inputs.size()
                : modifiedInputsOp[this->boltOperators[this->boltOperatorIndex].type];
            this->boltOperators[this->boltOperatorIndex].input_tensors_name = (I8 **)mt_malloc(
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
                    (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                str_copy(this->boltOperators[this->boltOperatorIndex].input_tensors_name[iter],
                    (inTensor->name).c_str(), (inTensor->name).length());
            }
            this->boltOperators[this->boltOperatorIndex].num_outputs =
                this->tfliteOperators[this->tfliteOperatorIndex]->outputs.size();
            this->boltOperators[this->boltOperatorIndex].output_tensors_name = (I8 **)mt_malloc(
                this->boltOperators[this->boltOperatorIndex].num_outputs * sizeof(I8 *));
            for (U32 iter = 0; iter < this->boltOperators[this->boltOperatorIndex].num_outputs;
                 iter++) {
                const int outIndex = this->tfliteOperators[this->tfliteOperatorIndex]->outputs[iter];
                const auto &outTensor = this->tfliteTensors[outIndex];
                this->boltOperators[this->boltOperatorIndex].output_tensors_name[iter] =
                    (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
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
        ms->ops = (OperatorSpec *)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
        UNI_MEMCPY(
            ms->ops, this->boltOperators.data(), sizeof(OperatorSpec) * ms->num_operator_specs);
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            this->boltOperatorNameMap[ms->ops[i].name] = i;
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }
        ms->ws = nullptr;
        ms->num_weight_specs = this->weightNumber;
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

    template <typename T = float>
    std::vector<T> transformTfliteTensorToVector(const std::unique_ptr<tflite::TensorT> &tensor)
    {
        std::vector<T> result;
        if (this->tfliteModelBuffer[tensor->buffer]->data.data() == nullptr) {
            return result;
        }
        const auto &weightShape = tensor->shape;
        U32 size = 1;
        for (U32 i = 0; i < weightShape.size(); i++) {
            size *= weightShape[i];
        }
        result = std::vector<T>(size);
        switch (tensor->type) {
            case tflite::TensorType_FLOAT32: {
                auto weight = reinterpret_cast<const float *>(
                    this->tfliteModelBuffer[tensor->buffer]->data.data());
                for (U32 i = 0; i < size; i++) {
                    result[i] = weight[i];
                }
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

    void assign_weight(WeightSpec &ws, std::string opName, int weights_index, int bias_index)
    {
        str_copy(ws.op_name, opName.c_str(), opName.length());
        ws.mdt = DT_F32;
        if (weights_index != -1) {
            const int weightIndex =
                this->tfliteOperators[this->tfliteOperatorIndex]->inputs[weights_index];
            const auto &weightTensor = this->tfliteTensors[weightIndex];
            std::vector<float> weight_data = transformTfliteTensorToVector(weightTensor);
            const auto &weight_shape = weightTensor->shape;
            int weight_num = 1;
            for (auto item : weight_shape) {
                weight_num *= item;
            }
            ws.bytes_of_weight = weight_num * sizeof(float);
            ws.weight = (U8 *)mt_malloc(ws.bytes_of_weight);
            UNI_MEMCPY(ws.weight, weight_data.data(), ws.bytes_of_weight);
        } else {
            ws.bytes_of_weight = 0;
            ws.weight = nullptr;
        }
        if (bias_index != -1) {
            const int biasIndex =
                this->tfliteOperators[this->tfliteOperatorIndex]->inputs[bias_index];
            const auto &biasTensor = this->tfliteTensors[biasIndex];
            std::vector<float> bias_data = transformTfliteTensorToVector(biasTensor);
            const auto &bias_shape = biasTensor->shape;
            int bias_num = 1;
            for (auto item : bias_shape) {
                bias_num *= item;
            }
            ws.bytes_of_vec = bias_num * sizeof(float);
            ws.vec = (U8 *)mt_malloc(ws.bytes_of_vec);
            UNI_MEMCPY(ws.vec, bias_data.data(), ws.bytes_of_vec);
        } else {
            ws.bytes_of_vec = 0;
            ws.vec = nullptr;
        }
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        WeightSpec *wsPtr = (WeightSpec *)mt_malloc(sizeof(WeightSpec) * ms->num_weight_specs);
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

            if (OT_Conv == ms->ops[this->boltOperatorIndex].type) {
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
                    (U8 *)mt_malloc(wsPtr[weightMovIndex].bytes_of_weight);
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
                    wsPtr[weightMovIndex].vec = (U8 *)mt_malloc(wsPtr[weightMovIndex].bytes_of_vec);
                    UNI_MEMCPY(wsPtr[weightMovIndex].vec, conv2DBias.data(),
                        wsPtr[weightMovIndex].bytes_of_vec);
                } else {
                    wsPtr[weightMovIndex].bytes_of_vec = 0;
                    wsPtr[weightMovIndex].vec = nullptr;
                }
            } else if (OT_Scale == ms->ops[this->boltOperatorIndex].type) {
                str_copy(wsPtr[weightMovIndex].op_name, operatorName.c_str(), operatorName.length());
                wsPtr[weightMovIndex].mdt = DT_F32;
                std::vector<int> weightInputIndex =
                    getOperatorWeightInputIndex(this->tfliteOperatorIndex);
                if (weightInputIndex.size() == 0) {
                    UNI_ERROR_LOG("can not map operator location:%d type:%s to Scale.\n",
                        this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
                }

                int cur_weights_index = -1;
                int cur_bias_index = -1;
                if (opCode == tflite::BuiltinOperator_ADD || opCode == tflite::BuiltinOperator_SUB) {
                    cur_bias_index = weightInputIndex[0];
                } else {  // MUL || DIV
                    cur_weights_index = weightInputIndex[0];
                }
                assign_weight(
                    wsPtr[weightMovIndex], operatorName, cur_weights_index, cur_bias_index);
                if (wsPtr[weightMovIndex].bytes_of_vec == 4) {
                    ms->ops[this->boltOperatorIndex].ps.scale_spec.axis = 0;
                }
                // special deal
                if (opCode == tflite::BuiltinOperator_SUB) {
                    F32 *ptr = (F32 *)wsPtr[weightMovIndex].vec;
                    for (U32 k = 0; k < wsPtr[weightMovIndex].bytes_of_vec / sizeof(float); k++) {
                        ptr[k] *= -1;
                    }
                } else if (opCode == tflite::BuiltinOperator_DIV) {
                    F32 *ptr = (F32 *)wsPtr[weightMovIndex].weight;
                    for (U32 k = 0; k < wsPtr[weightMovIndex].bytes_of_weight / sizeof(float); k++) {
                        ptr[k] = 1.0 / ptr[k];
                    }
                }
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
                    (U8 *)mt_malloc(wsPtr[weightMovIndex].bytes_of_weight);
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
            } else if (ordinary_weight_op.find(ms->ops[this->boltOperatorIndex].type) !=
                ordinary_weight_op.end()) {
                // input 2/3: input/weight/bias
                std::vector<int> weight_inputs =
                    getOperatorWeightInputIndex(this->tfliteOperatorIndex);
                int cur_weights_index = (weight_inputs.size() > 0) ? weight_inputs[0] : -1;
                int cur_bias_index = (weight_inputs.size() > 1) ? weight_inputs[1] : -1;
                assign_weight(
                    wsPtr[weightMovIndex], operatorName, cur_weights_index, cur_bias_index);
            } else {
                weightMovIndex--;
            }
            weightMovIndex++;
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
        ParameterSpec ps;
        EltwiseParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        ActivationMode activationMode = ACTIVATION_NULL;
        if (opCode == tflite::BuiltinOperator_ADD) {
            p.mode = ELTWISE_SUM;
            EltwiseSumSpec sum_spec;
            sum_spec.num_coeff = 2;
            for (I32 j = 0; j < sum_spec.num_coeff; j++) {
                sum_spec.coeff[j] = 1.0;
            }
            p.sum_spec = sum_spec;
            const auto &tfliteEltwiseOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsAddOptions();
            activationMode =
                getActivationOperatorType(tfliteEltwiseOption->fused_activation_function);
        } else if (opCode == tflite::BuiltinOperator_SUB) {
            p.mode = ELTWISE_SUB;
        } else if (opCode == tflite::BuiltinOperator_MAXIMUM) {
            p.mode = ELTWISE_MAX;
        } else if (opCode == tflite::BuiltinOperator_MINIMUM) {
            p.mode = ELTWISE_MIN;
        } else if (opCode == tflite::BuiltinOperator_DIV) {
            p.mode = ELTWISE_DIV;
        } else if (opCode == tflite::BuiltinOperator_MUL) {
            p.mode = ELTWISE_PROD;
            const auto &tfliteEltwiseOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsMulOptions();
            activationMode =
                getActivationOperatorType(tfliteEltwiseOption->fused_activation_function);
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Eltwise.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        p.activation_type = activationMode;
        ps.eltwise_spec = p;
        std::vector<int> weights = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        for (U32 i = 0; i < weights.size(); i++) {
            insertSharedWeight(this->tfliteOperators[this->tfliteOperatorIndex]->inputs[weights[i]]);
        }
        return ps;
    }

    ParameterSpec adapt_Scale() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        ScaleParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = 1;
        ps.scale_spec = p;

        if (opCode == tflite::BuiltinOperator_ADD) {
            const auto &addOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsAddOptions();
            if (getActivationOperatorType(addOption->fused_activation_function) != ACTIVATION_NULL) {
                insertActivationOperator(
                    getActivationOperatorType(addOption->fused_activation_function));
            }
        }

        return ps;
    }

    ParameterSpec adapt_Conv() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        ConvolutionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const int weightIndex = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
        const auto &weightTensor = this->tfliteTensors[weightIndex];
        const auto &weightShape = weightTensor->shape;
        CHECK_REQUIREMENT(weightShape.size() == 4);
        p.kernel_h = weightShape[1];
        p.kernel_w = weightShape[2];
        p.kernel_t = 1;
        p.stride_t = 1;
        p.pad_before = 0;
        p.pad_after = 0;
        p.dilatedRate_t = 1;
        if (opCode == tflite::BuiltinOperator_CONV_2D) {
            p.num_outputs = weightShape[0];
            p.num_outputs_origin = p.num_outputs;

            const auto &tfliteConvOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsConv2DOptions();
            p.dilatedRate_h = tfliteConvOption->dilation_h_factor;
            p.dilatedRate_w = tfliteConvOption->dilation_w_factor;
            p.stride_h = tfliteConvOption->stride_h;
            p.stride_w = tfliteConvOption->stride_w;
            const auto activationFunc = tfliteConvOption->fused_activation_function;
            if (1 == tfliteConvOption->padding) {  // VALID
                p.pad_top = 0;
                p.pad_bottom = 0;
                p.pad_left = 0;
                p.pad_right = 0;
            } else {  // SAME
                const auto &inputTensor =
                    this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
                const auto &inputShape = inputTensor->shape;
                p.pad_top = (p.kernel_h - 1) / 2;
                p.pad_bottom = (p.kernel_h - 1) / 2;
                if (p.kernel_h % 2 == 0) {
                    p.pad_bottom += 1;
                }
                if (p.pad_top != 0 && inputShape[1] % 2 == 0 && tfliteConvOption->stride_h % 2 == 0) {
                    p.pad_top -= 1;
                }
                p.pad_left = (p.kernel_w - 1) / 2;
                p.pad_right = (p.kernel_w - 1) / 2;
                if (p.kernel_w % 2 == 0) {
                    p.pad_right += 1;
                }
                if (p.pad_left != 0 && inputShape[2] % 2 == 0 &&
                    tfliteConvOption->stride_w % 2 == 0) {
                    p.pad_left -= 1;
                }
            }
            p.group = 1;
            p.dw_activation_type = ACTIVATION_NULL;
            p.pw_activation_type = getActivationOperatorType(activationFunc);
            p.convolution_type = CONVOLUTION_POINTWISE;
        } else if (opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
            p.num_outputs = weightShape[3];
            p.num_outputs_origin = p.num_outputs;

            const auto &tfliteConvOption = this->tfliteOperators[this->tfliteOperatorIndex]
                                               ->builtin_options.AsDepthwiseConv2DOptions();
            p.dilatedRate_h = tfliteConvOption->dilation_h_factor;
            p.dilatedRate_w = tfliteConvOption->dilation_w_factor;
            p.stride_h = tfliteConvOption->stride_h;
            p.stride_w = tfliteConvOption->stride_w;
            const auto activationFunc = tfliteConvOption->fused_activation_function;

            if (1 == tfliteConvOption->padding) {  // VALID
                p.pad_top = 0;
                p.pad_bottom = 0;
                p.pad_left = 0;
                p.pad_right = 0;
            } else {  // SAME
                const auto &inputTensor =
                    this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
                const auto &inputShape = inputTensor->shape;
                p.pad_top = (p.kernel_h - 1) / 2;
                p.pad_bottom = (p.kernel_h - 1) / 2;
                if (p.kernel_h % 2 == 0) {
                    p.pad_bottom += 1;
                }
                if (p.pad_top != 0 && inputShape[1] % 2 == 0 && tfliteConvOption->stride_h % 2 == 0) {
                    p.pad_top -= 1;
                }
                p.pad_left = (p.kernel_w - 1) / 2;
                p.pad_right = (p.kernel_w - 1) / 2;
                if (p.kernel_w % 2 == 0) {
                    p.pad_right += 1;
                }
                if (p.pad_left != 0 && inputShape[2] % 2 == 0 &&
                    tfliteConvOption->stride_w % 2 == 0) {
                    p.pad_left -= 1;
                }
            }

            p.group = p.num_outputs;
            // process the situation: when depth_multiplier > 1 && fn == depth_multiplier, depthwise ==> pointwise
            if (tfliteConvOption->depth_multiplier > 1 &&
                tfliteConvOption->depth_multiplier == weightShape[3]) {
                p.convolution_type = CONVOLUTION_POINTWISE;
                p.dw_activation_type = ACTIVATION_NULL;
                p.pw_activation_type = getActivationOperatorType(activationFunc);
                p.group = 1;
            } else {
                p.convolution_type = CONVOLUTION_DEPTHWISE;
                p.dw_activation_type = getActivationOperatorType(activationFunc);
                p.pw_activation_type = ACTIVATION_NULL;
            }
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Convolution.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        ps.conv_spec = p;
        return ps;
    }

    ParameterSpec adapt_Reduction() override
    {
        ParameterSpec ps;
        ReductionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (opCode == tflite::BuiltinOperator_MEAN) {
            p.mode = REDUCTION_MEAN;
        } else if (opCode == tflite::BuiltinOperator_SUM) {
            p.mode = REDUCTION_SUM;
        } else if (opCode == tflite::BuiltinOperator_REDUCE_MAX) {
            p.mode = REDUCTION_MAX;
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
        p.num_axes = axisData.size() / sizeof(int);
        auto axisPtr = reinterpret_cast<const int32_t *>(axisData.data());
        UNI_MEMCPY(p.axes, axisPtr, axisData.size());
        if (this->weightFormat == DF_NHWC) {
            for (int i = 0; i < p.num_axes; i++) {
                p.axes[i] = NHWCAxisToNCHWAxis(p.axes[i], inputShape.size());
            }
        }
        p.coeff = 1;
        p.keep_dim = false;
        ps.reduction_spec = p;
        return ps;
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec ps;
        PoolingParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.kernel_t = 1;
        p.stride_t = 1;
        p.pad_before = 0;
        p.pad_after = 0;
        p.pad_top = 0;
        p.pad_bottom = 0;
        p.pad_left = 0;
        p.pad_right = 0;
        p.round_mode = ROUND_CEIL;

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
            p.mode = POOLING_MEAN;
            p.kernel_h = 0;
            p.kernel_w = 0;
            p.stride_h = 1;
            p.stride_w = 1;
        } else {
            const auto &tflitePoolOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsPool2DOptions();
            p.kernel_h = tflitePoolOption->filter_height;
            p.kernel_w = tflitePoolOption->filter_width;
            p.stride_h = tflitePoolOption->stride_h;
            p.stride_w = tflitePoolOption->stride_w;
            int tfPaddingRoundMode = tflitePoolOption->padding;
            if (tfPaddingRoundMode == 0) {
                p.round_mode = ROUND_TF_SAME;

                int oLength = (inputShape[2] + p.stride_w - 1) / p.stride_w;
                int padLength = UNI_MAX((oLength - 1) * p.stride_w + p.kernel_w - inputShape[2], 0);
                p.pad_left = padLength / 2;
                p.pad_right = padLength - p.pad_left;

                oLength = (inputShape[1] + p.stride_h - 1) / p.stride_h;
                padLength = UNI_MAX((oLength - 1) * p.stride_h + p.kernel_h - inputShape[1], 0);
                p.pad_top = padLength / 2;
                p.pad_bottom = padLength - p.pad_top;
            } else if (tfPaddingRoundMode == 1) {
                p.round_mode = ROUND_TF_VALID;
            } else {
                UNI_ERROR_LOG("can not process operator location:%d Pooling round mode.\n",
                    this->tfliteOperatorIndex);
            }
            if (opCode == tflite::BuiltinOperator_MAX_POOL_2D) {
                p.mode = POOLING_MAX;
            } else if (opCode == tflite::BuiltinOperator_AVERAGE_POOL_2D) {
                p.mode = POOLING_MEAN;
            }
            insertActivationOperator(
                getActivationOperatorType(tflitePoolOption->fused_activation_function));
        }
        p.count_include_pad = false;
        ps.pooling_spec = p;
        return ps;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec ps;
        ReshapeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> shape;
        if (this->tfliteOperators[this->tfliteOperatorIndex]->inputs.size() == 1) {
            const auto &tp =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsReshapeOptions();
            shape = tp->new_shape;
        } else {
            const auto &t =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
            const auto &v = tfliteModelBuffer[t->buffer]->data;
            shape = std::vector<int>(v.size() / sizeof(int));
            UNI_MEMCPY(shape.data(), v.data(), v.size());
        }
        p.num_shape = shape.size();
        for (int iter = 0; iter < p.num_shape; iter++) {
            int axis = iter;
            if (this->weightFormat == DF_NHWC) {
                axis = NHWCAxisToNCHWAxis(iter, p.num_shape);
            }
            p.shape[axis] = shape[iter];
        }
        p.axis = 8;
        p.num_axes = -1;
        ps.reshape_spec = p;
        return ps;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec ps;
        TransposeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const auto &dimsTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &dimsData = tfliteModelBuffer[dimsTensor->buffer]->data;
        CHECK_REQUIREMENT((dimsTensor->shape[0]) == (int)(dimsData.size() / sizeof(int)));
        p.num_axes = dimsTensor->shape[0];
        auto dims = reinterpret_cast<const int32_t *>(dimsData.data());
        for (U32 i = 0; i < p.num_axes; i++) {
            if (this->weightFormat == DF_NHWC) {
                p.axes[i] = NHWCAxisToNCHWAxis(dims[i], p.num_axes);
            } else {
                p.axes[i] = dims[i];
            }
        }
        ps.transpose_spec = p;
        return ps;
    }

    ParameterSpec adapt_TfSlice() override
    {
        ParameterSpec ps;
        TfSliceParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (opCode == tflite::BuiltinOperator_STRIDED_SLICE) {
            const auto &stridedSliceOption = this->tfliteOperators[this->tfliteOperatorIndex]
                                                 ->builtin_options.AsStridedSliceOptions();
            const auto &beginTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
            p.num_dims = beginTensor->shape[0];
            auto beginData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[beginTensor->buffer]->data).data());
            UNI_MEMCPY(p.begin, beginData, sizeof(int) * p.num_dims);
            const auto &endTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[2]];
            auto endData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[endTensor->buffer]->data).data());
            UNI_MEMCPY(p.end, endData, sizeof(int) * p.num_dims);
            const auto &stridesTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[3]];
            auto stridesData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[stridesTensor->buffer]->data).data());
            UNI_MEMCPY(p.strides, stridesData, sizeof(int) * p.num_dims);
            bitsToCharArray(stridedSliceOption->begin_mask, p.begin_mask, p.num_dims);
            bitsToCharArray(stridedSliceOption->end_mask, p.end_mask, p.num_dims);
            bitsToCharArray(stridedSliceOption->ellipsis_mask, p.ellipsis_mask, p.num_dims);
            bitsToCharArray(stridedSliceOption->new_axis_mask, p.new_axis_mask, p.num_dims);
            bitsToCharArray(stridedSliceOption->shrink_axis_mask, p.shrink_axis_mask, p.num_dims);
        } else {
            const auto &beginTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
            p.num_dims = beginTensor->shape[0];
            auto beginData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[beginTensor->buffer]->data).data());
            UNI_MEMCPY(p.begin, beginData, sizeof(int) * p.num_dims);
            const auto &sizeTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[2]];
            auto sizeData = reinterpret_cast<const int32_t *>(
                (tfliteModelBuffer[sizeTensor->buffer]->data).data());
            for (U32 i = 0; i < p.num_dims; i++) {
                p.end[i] = p.begin[i] + sizeData[i];
                p.strides[i] = 1;
            }
            UNI_MEMSET(p.begin_mask, 0, sizeof(char) * p.num_dims);
            UNI_MEMSET(p.end_mask, 0, sizeof(char) * p.num_dims);
            UNI_MEMSET(p.ellipsis_mask, 0, sizeof(char) * p.num_dims);
            UNI_MEMSET(p.new_axis_mask, 0, sizeof(char) * p.num_dims);
            UNI_MEMSET(p.shrink_axis_mask, 0, sizeof(char) * p.num_dims);
        }
        if (this->weightFormat == DF_NHWC) {
            shiftRight<int>(p.begin, p.num_dims, 1, p.num_dims - 1);
            shiftRight<int>(p.end, p.num_dims, 1, p.num_dims - 1);
            shiftRight<int>(p.strides, p.num_dims, 1, p.num_dims - 1);
            shiftRight<char>(p.begin_mask, p.num_dims, 1, p.num_dims - 1);
            shiftRight<char>(p.end_mask, p.num_dims, 1, p.num_dims - 1);
            shiftRight<char>(p.ellipsis_mask, p.num_dims, 1, p.num_dims - 1);
            shiftRight<char>(p.new_axis_mask, p.num_dims, 1, p.num_dims - 1);
            shiftRight<char>(p.shrink_axis_mask, p.num_dims, 1, p.num_dims - 1);
        }
        ps.tfslice_spec = p;
        return ps;
    }

    void insertSharedWeight(int tensorId)
    {
        const auto &tensor = this->tfliteTensors[tensorId];
        std::string name = tensor->name;
        OperatorSpec sharedWeight = mt_create_operator(name.c_str(), OT_SharedWeight, 0, 1);
        str_copy(sharedWeight.output_tensors_name[0], name.c_str(), NAME_LEN);
        SharedWeightParamSpec p;
        p.desc = getDescFromTp(tensor);
        if (p.desc.nDims == 4 && this->weightFormat == DF_NHWC) {
            p.desc.df = DF_NHWC;
        }

        WeightSpec weightSpec =
            mt_create_weight(name.c_str(), p.desc.dt, tensorNumBytes(p.desc), 0, 0);
        switch (p.desc.dt) {
            case DT_F32: {
                std::vector<float> v = transformTfliteTensorToVector(tensor);
                if (p.desc.df == DF_NHWC) {
                    TensorDesc nchwDesc = p.desc;
                    nchwDesc.df = DF_NCHW;
                    transformToNCHW(p.desc, v.data(), nchwDesc, weightSpec.weight);
                    p.desc = nchwDesc;
                } else {
                    UNI_MEMCPY(weightSpec.weight, v.data(), tensorNumBytes(p.desc));
                }
                break;
            }
            case DT_I32:
            case DT_U32: {
                std::vector<int> v = transformTfliteTensorToVector<int>(tensor);
                UNI_MEMCPY(weightSpec.weight, v.data(), tensorNumBytes(p.desc));
                break;
            }
            default:
                UNI_ERROR_LOG(
                    "can not map %s type tensor to shared weight.\n", DataTypeName()[p.desc.dt]);
                break;
        }
        this->boltSharedWeights.push_back(weightSpec);
        sharedWeight.ps.shared_weight_spec = p;
        this->boltOperators.insert(
            this->boltOperators.begin() + this->boltOperatorIndex, sharedWeight);
        this->boltOperatorInsertBefore++;
        this->weightNumber++;
    }

    ParameterSpec adapt_MatMul() override
    {
        ParameterSpec ps;
        MatMulParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.transpose_a = false;
        p.transpose_b = false;
        std::vector<int> weightInputIndex = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        if (weightInputIndex.size() == 2 && weightInputIndex[0] == 0 && weightInputIndex[1] == 2) {
            p.transpose_b = true;
            insertSharedWeight(this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]);
        }
        ps.matmul_spec = p;
        return ps;
    }

    ParameterSpec adapt_Fc() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        FullyConnectedParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const int index = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
        const auto &tensor = this->tfliteTensors[index];
        I32 size = tfliteModelBuffer[tensor->buffer]->data.size();
        CHECK_REQUIREMENT(size != 0);
        const auto &weightShape = tensor->shape;
        p.num_outputs = weightShape[0];
        p.num_slices = 1;
        p.slice_point[0] = p.num_outputs;
        ps.fc_spec = p;
        const auto &tfliteFullyConnectedOption = this->tfliteOperators[this->tfliteOperatorIndex]
                                                     ->builtin_options.AsFullyConnectedOptions();
        insertActivationOperator(
            getActivationOperatorType(tfliteFullyConnectedOption->fused_activation_function));
        return ps;
    }

    ParameterSpec adapt_Concat() override
    {
        ParameterSpec ps;
        ConcatParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::vector<int> pinnedInput;
        if (this->opCode == tflite::BuiltinOperator_CONCATENATION) {
            const auto &tfliteConcatOption = this->tfliteOperators[this->tfliteOperatorIndex]
                                                 ->builtin_options.AsConcatenationOptions();
            insertActivationOperator(
                getActivationOperatorType(tfliteConcatOption->fused_activation_function));
            p.axis = tfliteConcatOption->axis;
            pinnedInput = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        } else {
            const auto &tflitePackOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsPackOptions();
            p.axis = tflitePackOption->axis;
            pinnedInput = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
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
            insertSharedWeight(this->tfliteOperators[this->tfliteOperatorIndex]->inputs[id]);
        }
        const auto &outputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->outputs[0]];
        const auto &outputShape = outputTensor->shape;
        if (this->weightFormat == DF_NHWC) {
            p.axis = NHWCAxisToNCHWAxis(p.axis, outputShape.size());
        }
        ps.concat_spec = p;
        return ps;
    }

    ParameterSpec adapt_Softmax() override
    {
        const auto &tfliteSoftmaxOption =
            this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsSoftmaxOptions();
        CHECK_REQUIREMENT(1 == tfliteSoftmaxOption->beta);

        ParameterSpec ps;
        SoftmaxParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = -1;
        const auto &inputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
        const auto &inputShape = inputTensor->shape;
        if (this->weightFormat == DF_NHWC) {
            p.axis = NHWCAxisToNCHWAxis(p.axis, inputShape.size());
        }
        ps.softmax_spec = p;
        return ps;
    }

    ParameterSpec adapt_Resize() override
    {
        ParameterSpec ps;
        ResizeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const auto &dimsTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &dimsData = tfliteModelBuffer[dimsTensor->buffer]->data;
        CHECK_REQUIREMENT((dimsTensor->shape[0]) == 2);
        auto dims = reinterpret_cast<const int32_t *>(dimsData.data());
        p.sizes[0] = dims[0];
        p.sizes[1] = dims[1];
        p.num_sizes = 2;
        p.num_scales = 0;
        if (this->opCode == tflite::BuiltinOperator_RESIZE_BILINEAR) {
            p.mode = RESIZE_LINEAR;
            const auto &tp = this->tfliteOperators[this->tfliteOperatorIndex]
                                 ->builtin_options.AsResizeBilinearOptions();
            p.trans_mode = (tp->align_corners) ? COORDINATE_TRANS_ALIGN_CORNERS
                                               : COORDINATE_TRANS_ASYMMETRIC;
        } else if (this->opCode == tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR) {
            p.mode = RESIZE_NEAREST;
            const auto &tp = this->tfliteOperators[this->tfliteOperatorIndex]
                                 ->builtin_options.AsResizeNearestNeighborOptions();
            p.trans_mode = (tp->align_corners) ? COORDINATE_TRANS_ALIGN_CORNERS
                                               : COORDINATE_TRANS_ASYMMETRIC;
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Resize.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        p.round_mode = ROUND_FLOOR;
        ps.resize_spec = p;
        return ps;
    }

    ParameterSpec adapt_Clip() override
    {
        ParameterSpec ps;
        ClipParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const auto &clipTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &clipData = tfliteModelBuffer[clipTensor->buffer]->data;
        if (opCode == tflite::BuiltinOperator_MINIMUM) {
            p.max = clipData[0];
            p.min = std::numeric_limits<float>::min();
        } else if (opCode == tflite::BuiltinOperator_MAXIMUM) {
            p.max = std::numeric_limits<float>::max();
            p.min = clipData[0];
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Clip.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        ps.clip_spec = p;
        return ps;
    }

    ParameterSpec adapt_Deconvolution() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        ConvolutionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const int weightIndex = this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1];
        const auto &weightTensor = this->tfliteTensors[weightIndex];
        const auto &weightShape = weightTensor->shape;
        CHECK_REQUIREMENT(weightShape.size() == 4);
        p.kernel_t = 1;
        p.kernel_h = weightShape[1];
        p.kernel_w = weightShape[2];
        p.num_outputs = weightShape[0];
        p.num_outputs_origin = p.num_outputs;

        const auto &tfliteDeConvOption =
            this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsTransposeConvOptions();
        p.stride_t = 1;
        p.stride_h = tfliteDeConvOption->stride_h;
        p.stride_w = tfliteDeConvOption->stride_w;
        p.group = 1;

        p.dilatedRate_t = 1;
        p.dilatedRate_h = 1;
        p.dilatedRate_w = 1;
        p.convolution_type = CONVOLUTION_DECONVOLUTION;
        p.dw_activation_type = ACTIVATION_NULL;
        p.pw_activation_type = ACTIVATION_NULL;

        p.pad_before = 0;
        p.pad_after = 0;
        if (tfliteDeConvOption->padding == 1) {
            p.round_mode = ROUND_TF_VALID;
            p.pad_top = 0;
            p.pad_bottom = 0;
            p.pad_left = 0;
            p.pad_right = 0;
        } else {
            p.round_mode = ROUND_TF_SAME;
            if (p.kernel_h < p.stride_h) {
                p.pad_top = 0;
                p.pad_bottom = 0;
            } else {
                p.pad_top = (p.kernel_h - p.stride_h) / 2;
                p.pad_bottom = p.kernel_h - p.stride_h - p.pad_top;
            }
            if (p.kernel_w < p.stride_w) {
                p.pad_left = 0;
                p.pad_right = 0;
            } else {
                p.pad_left = (p.kernel_w - p.stride_w) / 2;
                p.pad_right = p.kernel_w - p.stride_w - p.pad_left;
            }
        }

        ps.conv_spec = p;
        return ps;
    }

    ParameterSpec adapt_Power() override
    {
        ParameterSpec ps;
        PowerParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.scale = 1;
        p.shift = 0;
        p.power = 1;
        float weight = 0;
        std::vector<int> weightInputIndex = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        if (weightInputIndex.size() > 0) {
            const auto &weightTensor =
                this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]
                                        ->inputs[weightInputIndex[0]]];
            weight = transformTfliteTensorToVector(weightTensor)[0];
        }
        if (opCode == tflite::BuiltinOperator_SQRT) {
            p.power = 0.5;
        } else if (opCode == tflite::BuiltinOperator_RSQRT) {
            p.power = -0.5;
        } else if (opCode == tflite::BuiltinOperator_POW) {
            p.power = weight;
        } else if (opCode == tflite::BuiltinOperator_ADD) {
            p.shift = weight;
        } else if (opCode == tflite::BuiltinOperator_SUB) {
            p.shift = weight * -1;
        } else if (opCode == tflite::BuiltinOperator_MUL) {
            p.scale = weight;
        } else if (opCode == tflite::BuiltinOperator_DIV) {
            p.scale = 1.0 / weight;
        } else if (opCode == tflite::BuiltinOperator_NEG) {
            p.scale = -1.0;
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Power.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        ps.power_spec = p;

        if (opCode == tflite::BuiltinOperator_ADD) {
            const auto &addOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsAddOptions();
            if (getActivationOperatorType(addOption->fused_activation_function) != ACTIVATION_NULL) {
                insertActivationOperator(
                    getActivationOperatorType(addOption->fused_activation_function));
            }
        }

        return ps;
    }

    ParameterSpec adapt_Pad() override
    {
        ParameterSpec ps;
        PadParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const auto &t =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        const auto &v = tfliteModelBuffer[t->buffer]->data;
        auto data = reinterpret_cast<const int32_t *>(v.data());
        int num = v.size() / sizeof(int);
        if (num == 8) {
            // nhwc
            p.top = data[2];
            p.bottom = data[3];
            p.left = data[4];
            p.right = data[5];
            p.front = data[6];
            p.back = data[7];
        } else if (num == 6) {
            // nhc
            p.top = data[2];
            p.bottom = data[3];
            p.front = data[4];
            p.back = data[5];
        } else {
            UNI_ERROR_LOG("can not process operator location:%d type:%s parameter.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        p.constant_value = 0;
        if (this->opCode == tflite::BuiltinOperator_PAD) {
            p.pad_mode = PAD_CONSTANT;
        } else {
            p.pad_mode = PAD_REFLECT;
        }
        ps.pad_spec = p;
        return ps;
    }

    ParameterSpec adapt_Relu() override
    {
        ParameterSpec ps;
        ReLUParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (this->opCode == tflite::BuiltinOperator_RELU) {
            p.neg_slope = 0;
        } else {
            const auto &tfliteLeakyReluOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsLeakyReluOptions();
            p.neg_slope = tfliteLeakyReluOption->alpha;
        }
        ps.relu_spec = p;
        return ps;
    }

    ParameterSpec adapt_Squeeze() override
    {
        ParameterSpec ps;
        SqueezeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const auto &tfliteSqueezeOption =
            this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsSqueezeOptions();
        p.num_axes = tfliteSqueezeOption->squeeze_dims.size();
        const auto &inputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
        const auto &inputShape = inputTensor->shape;
        for (int i = 0; i < p.num_axes; i++) {
            if (this->weightFormat == DF_NHWC) {
                p.axes[i] =
                    NHWCAxisToNCHWAxis(tfliteSqueezeOption->squeeze_dims[i], inputShape.size());
            } else {
                p.axes[i] = tfliteSqueezeOption->squeeze_dims[i];
            }
        }
        ps.squeeze_spec = p;
        return ps;
    }

    ParameterSpec adapt_Unsqueeze() override
    {
        ParameterSpec ps;
        UnsqueezeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const auto &weightTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        auto weightData = transformTfliteTensorToVector(weightTensor);
        const auto &inputTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[0]];
        const auto &inputShape = inputTensor->shape;
        if (this->weightFormat == DF_NHWC) {
            p.axes[0] = NHWCAxisToNCHWAxis((int)weightData[0], inputShape.size());
        } else {
            p.axes[0] = weightData[0];
        }
        p.num_axes = 1;
        ps.unsqueeze_spec = p;
        return ps;
    }

    ParameterSpec adapt_TopK() override
    {
        ParameterSpec ps;
        TopKParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = 0;
        p.largest = 1;
        p.sorted = 1;
        const auto &weightTensor =
            this->tfliteTensors[this->tfliteOperators[this->tfliteOperatorIndex]->inputs[1]];
        auto weightData = transformTfliteTensorToVector(weightTensor);
        p.k = weightData[0];
        ps.topk_spec = p;
        return ps;
    }

    ParameterSpec adapt_Gather() override
    {
        ParameterSpec ps;
        GatherParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (this->opCode == tflite::BuiltinOperator_GATHER) {
            const auto &gatherOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsGatherOptions();
            p.axis = gatherOption->axis;
        } else {
            p.axis = INT_MAX;
        }
        p.element_level = false;
        if (this->opCode == tflite::BuiltinOperator_GATHER_ND) {
            const auto &gatherNDOption =
                this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsGatherNdOptions();
            p.batch_dims = 0;  //gatherNDOption->batch_dim;
        } else {
            p.batch_dims = 0;
        }
        ps.gather_spec = p;
        std::vector<int> weights = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        for (U32 i = 0; i < weights.size(); i++) {
            insertSharedWeight(this->tfliteOperators[this->tfliteOperatorIndex]->inputs[weights[i]]);
        }
        return ps;
    }

    ParameterSpec adapt_PRelu() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        return ps;
    }

    ParameterSpec adapt_SpaceToBatchNd() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        return ps;
    }

    ParameterSpec adapt_BatchToSpaceNd() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        UNI_MEMSET(&ps, 0, sizeof(ps));
        return ps;
    }

    ParameterSpec adapt_Slice() override
    {
        ParameterSpec ps;
        SliceParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
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
                p.axis = NHWCAxisToNCHWAxis((int)axisData[0], inputShape.size());
            } else {
                p.axis = axisData[0];
            }
            p.num_slice = tfliteSplit->num_splits - 1;
            UNI_MEMSET(p.slice_points, 0, p.num_slice * sizeof(I32));
        } else if (this->opCode == tflite::BuiltinOperator_QUANTIZE ||
            this->opCode == tflite::BuiltinOperator_FAKE_QUANT) {
            p.num_slice = 0;
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Slice.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        ps.slice_spec = p;
        return ps;
    }

    ParameterSpec adapt_Check() override
    {
        ParameterSpec ps;
        CheckParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (this->opCode == tflite::BuiltinOperator_EQUAL) {
            p.mode = CHECK_EQUAL;
        } else if (this->opCode == tflite::BuiltinOperator_NOT_EQUAL) {
            p.mode = CHECK_NOT_EQUAL;
        } else {
            UNI_ERROR_LOG("can not map operator location:%d type:%s to Check.\n",
                this->tfliteOperatorIndex, tflite::EnumNamesBuiltinOperator()[this->opCode]);
        }
        ps.check_spec = p;
        std::vector<int> weights = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        for (U32 i = 0; i < weights.size(); i++) {
            insertSharedWeight(this->tfliteOperators[this->tfliteOperatorIndex]->inputs[weights[i]]);
        }
        return ps;
    }

    ParameterSpec adapt_Select() override
    {
        ParameterSpec ps;
        std::vector<int> weights = getOperatorWeightInputIndex(this->tfliteOperatorIndex);
        for (U32 i = 0; i < weights.size(); i++) {
            insertSharedWeight(this->tfliteOperators[this->tfliteOperatorIndex]->inputs[weights[i]]);
        }
        return ps;
    }

    ParameterSpec adapt_Cast() override
    {
        ParameterSpec ps;
        CastParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        const auto &tfliteCast =
            this->tfliteOperators[this->tfliteOperatorIndex]->builtin_options.AsCastOptions();
        if (tfliteCast != nullptr) {
            p.dt = TfliteTensorType2BoltDataType(tfliteCast->out_data_type);
        } else {
            p.dt = DT_F32;
        }
        ps.cast_spec = p;
        return ps;
    }

public:
    std::set<OperatorType> ordinary_weight_op = {
        OT_PRelu, OT_SpaceToBatchNd, OT_BatchToSpaceNd, OT_FC};

    std::map<OperatorType, int> modifiedInputsOp{{OT_Conv, 1}, {OT_Reshape, 1}, {OT_Resize, 1},
        {OT_Transpose, 1}, {OT_FC, 1}, {OT_Slice, 1}, {OT_Scale, 1}, {OT_Pooling, 1}, {OT_Clip, 1},
        {OT_Deconvolution, 1}, {OT_SqDiff, 1}, {OT_Reduction, 1}, {OT_Pad, 1}, {OT_Power, 1},
        {OT_TfSlice, 1}, {OT_SpaceToBatchNd, 1}, {OT_BatchToSpaceNd, 1}, {OT_MatMul, 2},
        {OT_PRelu, 1}, {OT_Gather, 2}, {OT_Check, 2}};

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
    int weightNumber;
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
