// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CAFFEADAPTEE
#define _H_CAFFEADAPTEE
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "caffe.pb.h"

#include "model_adaptee.h"

class CaffeAdaptee : public ModelAdaptee {
public:
    CaffeAdaptee()
    {}

    ~CaffeAdaptee()
    {
        google::protobuf::ShutdownProtobufLibrary();
    }

protected:
    // read prototxt
    EE read_from_prototxt(const char *path, google::protobuf::Message *message)
    {
        std::ifstream fs(path, std::ifstream::in);
        if (!fs.is_open()) {
            UNI_ERROR_LOG("can not open caffe model file %s.\n", path);
        }

        google::protobuf::io::IstreamInputStream input(&fs);
        bool ret = google::protobuf::TextFormat::Parse(&input, message);
        fs.close();
        return (ret) ? SUCCESS : NOT_SUPPORTED;
    }

    // read caffemodel(bin)
    EE read_from_caffemodel(const char *path, google::protobuf::Message *message)
    {
        std::ifstream fs(path, std::ifstream::in | std::ifstream::binary);
        if (!fs.is_open()) {
            UNI_ERROR_LOG("can not open caffe prototxt file %s.\n", path);
        }

        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::io::CodedInputStream codedstr(&input);

        codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

        bool ret = message->ParseFromCodedStream(&codedstr);
        fs.close();

        return (ret) ? SUCCESS : NOT_SUPPORTED;
    }

    OperatorType convert_caffe_type(std::string inputType)
    {
        std::map<std::string, OperatorType> operatorMap = {{"Convolution", OT_Conv},
            {"Deconvolution", OT_Deconvolution}, {"L2Norm", OT_L2Norm}, {"BatchNorm", OT_BatchNorm},
            {"Scale", OT_Scale}, {"Eltwise", OT_Eltwise}, {"InnerProduct", OT_FC},
            {"Pooling", OT_Pooling}, {"ReLU", OT_Relu}, {"ReLU6", OT_Relu6}, {"HSwish", OT_HSwish},
            {"Sigmoid", OT_Sigmoid}, {"HSigmoid", OT_HSigmoid}, {"Softmax", OT_Softmax},
            {"Concat", OT_Concat}, {"Embed", OT_Embedding}, {"Gelu", OT_Gelu},
            {"LayerNorm", OT_LayerNorm}, {"MatMul", OT_MatMul}, {"Power", OT_Power},
            {"Reshape", OT_Reshape}, {"Slice", OT_Slice}, {"Attention", OT_Attention},
            {"Input", OT_Input}, {"LSTM", OT_RNN}, {"TanH", OT_TanH},
            {"SoftmaxWithLoss", OT_SoftmaxWithLoss}, {"Squeeze", OT_Squeeze},
            {"Unsqueeze", OT_Unsqueeze}, {"Reduction", OT_Reduction}, {"ArgMax", OT_ArgMax},
            {"PreAllocatedMemory", OT_PreAllocatedMemory}, {"SharedWeight", OT_SharedWeight},
            {"Copy", OT_Copy}, {"Check", OT_Check}, {"Repeat", OT_Repeat}, {"Interp", OT_Resize},
            {"Jump", OT_Jump}, {"AttentionMask", OT_AttentionMask},
            {"RelativePositionEmbed", OT_RelativePositionEmbedding},
            {"RelativeShift", OT_RelativeShift}, {"Dropout", OT_Dropout}, {"Flatten", OT_Reshape},
            {"Permute", OT_Transpose}, {"Clip", OT_Clip}, {"PriorBox", OT_PriorBox},
            {"DetectionOutput", OT_DetectionOutput}, {"LogSoftmax", OT_LogSoftmax},
            {"Yolov3DetectionOutput", OT_Yolov3DetectionOutput}, {"Mish", OT_Mish},
            {"PReLU", OT_PRelu}, {"Tile", OT_Tile}, {"Pad", OT_Pad}, {"SoftPlus", OT_Softplus},
            {"Exp", OT_Exp}, {"AbsVal", OT_Abs}, {"Silence", OT_None},
            {"BilateralSliceApply", OT_BilateralSliceApply}, {"ConvertColor", OT_ConvertColor},
            {"Cum", OT_Cum}, {"OneHot", OT_OneHot}, {"LutPreprocess", OT_LutPreprocess},
            {"Lut", OT_Lut}, {"Resize", OT_Resize}};
        if (operatorMap.find(inputType) == operatorMap.end()) {
            UNI_ERROR_LOG("operator name:%s type:%s not supported.\n", this->layer.name().c_str(),
                inputType.c_str());
            return OT_None;
        } else {
            return operatorMap[inputType];
        }
    }

    int net_search_layerId(caffe::NetParameter &netParams, std::string &layerName)
    {
        int i = 0;
        if (netParams.layer_size() > 0) {
            for (i = 0; i < netParams.layer_size(); i++) {
                if (netParams.layer(i).name() == layerName) {
                    return i;
                }
            }
        } else {
            for (i = 0; i < netParams.layers_size(); i++) {
                if (netParams.layers(i).name() == layerName) {
                    return i;
                }
            }
        }
        return -1;
    }

    caffe::BlobProto net_get_blob(caffe::NetParameter &netParams, int layerId, int blobId)
    {
        if (netParams.layer_size() > 0) {
            return netParams.layer(layerId).blobs(blobId);
        } else {
            return netParams.layers(layerId).blobs(blobId);
        }
    }

    int net_get_blobs_size(caffe::NetParameter &netParams, int layerId)
    {
        if (netParams.layer_size() > 0) {
            return netParams.layer(layerId).blobs_size();
        } else {
            return netParams.layers(layerId).blobs_size();
        }
    }

    void net_copy_blob(WeightSpec *wsPtr,
        int weightIndex,
        caffe::NetParameter &netParams,
        int netLayerId,
        int blobNum,
        OperatorType operatorType)
    {
        wsPtr[weightIndex].mdt = DT_F32;
        wsPtr[weightIndex].bytes_of_weight = 0;
        wsPtr[weightIndex].weight = nullptr;
        wsPtr[weightIndex].bytes_of_vec = 0;
        wsPtr[weightIndex].vec = nullptr;

        std::vector<std::pair<caffe::BlobProto, U32>> weights;
        std::vector<std::pair<caffe::BlobProto, U32>> biases;
        // Batchnorm may have 3 blobs, but the third blob can be ignored
        if (operatorType == OT_BatchNorm) {
            if (blobNum >= 3) {
                blobNum = 2;
            }
        }
        if (blobNum >= 1) {
            caffe::BlobProto blob0 = net_get_blob(netParams, netLayerId, 0);
            U32 elemSize = sizeof(*(blob0.data().data()));
            CHECK_REQUIREMENT(elemSize == bytesOf(wsPtr[weightIndex].mdt));
            U32 blobSize = elemSize * blob0.data_size();
            wsPtr[weightIndex].bytes_of_weight += blobSize;
            weights.push_back(std::make_pair(blob0, blobSize));
        }
        if (blobNum >= 2) {
            caffe::BlobProto blob1 = net_get_blob(netParams, netLayerId, 1);
            U32 elemSize = sizeof(*(blob1.data().data()));
            CHECK_REQUIREMENT(sizeof(*(blob1.data().data())) == bytesOf(wsPtr[weightIndex].mdt));
            U32 blobSize = elemSize * blob1.data_size();
            wsPtr[weightIndex].bytes_of_vec += blobSize;
            biases.push_back(std::make_pair(blob1, blobSize));
        }
        if (blobNum >= 3) {
            caffe::BlobProto blob2 = net_get_blob(netParams, netLayerId, 2);
            U32 elemSize = sizeof(*(blob2.data().data()));
            CHECK_REQUIREMENT(elemSize == bytesOf(wsPtr[weightIndex].mdt));
            U32 blobSize = elemSize * blob2.data_size();
            wsPtr[weightIndex].bytes_of_weight += blobSize;
            weights.push_back(std::make_pair(blob2, blobSize));
        }
        if (weights.size() > 0) {
            wsPtr[weightIndex].weight = (U8 *)mt_malloc(wsPtr[weightIndex].bytes_of_weight);
            U8 *ptr = wsPtr[weightIndex].weight;
            for (U32 i = 0; i < weights.size(); i++) {
                UNI_MEMCPY(ptr, weights[i].first.data().data(), weights[i].second);
                ptr += weights[i].second;
            }
        }
        if (biases.size() > 0) {
            wsPtr[weightIndex].vec = (U8 *)mt_malloc(wsPtr[weightIndex].bytes_of_vec);
            U8 *ptr = wsPtr[weightIndex].vec;
            for (U32 i = 0; i < biases.size(); i++) {
                UNI_MEMCPY(ptr, biases[i].first.data().data(), biases[i].second);
                ptr += biases[i].second;
            }
        }
    }

    EE parse_file(std::string dir, std::string mfn) override
    {
        EE ret = SUCCESS;
        std::string prototxtSuffix = ".prototxt";
        std::string caffeModelSuffix = ".caffemodel";
        std::string prototxtPath = dir + "/" + mfn + prototxtSuffix;
        std::string caffeModelPath = dir + "/" + mfn + caffeModelSuffix;

        // load prototxt
        ret = read_from_prototxt(prototxtPath.c_str(), (google::protobuf::Message *)(&proto));
        if (proto.layer_size() <= 0 || ret != SUCCESS) {
            UNI_ERROR_LOG("can not read caffe prototxt file %s.\n", prototxtPath.c_str());
        }

        // load model bin.
        ret = read_from_caffemodel(caffeModelPath.c_str(), (google::protobuf::Message *)(&net));
        if (ret != SUCCESS) {
            UNI_ERROR_LOG("can not read caffe model file %s.\n", caffeModelPath.c_str());
        }
        return ret;
    }

    DataType get_type(const caffe::DataType &dt)
    {
        DataType ret;
        switch (dt) {
            case caffe::DataType::FLOAT32:
            case caffe::DataType::FLOAT16:
                ret = DT_F32;
                break;
            case caffe::DataType::UINT32:
                ret = DT_U32;
                break;
            case caffe::DataType::INT32:
                ret = DT_I32;
                break;
            case caffe::DataType::UINT8:
                ret = DT_U8;
                break;
            case caffe::DataType::INT8:
                ret = DT_I8;
                break;
            default: {
                const google::protobuf::EnumDescriptor *descriptor = caffe::DataType_descriptor();
                UNI_ERROR_LOG("can not process operator name:%s %s type memory.\n",
                    this->layer.name().c_str(), descriptor->FindValueByNumber(dt)->name().c_str());
            }
        }
        return ret;
    }

    DataFormat get_format(const caffe::DataFormat &df)
    {
        DataFormat ret;
        switch (df) {
            case caffe::DataFormat::NHWC:
                ret = DF_NHWC;
                break;
            case caffe::DataFormat::NCHW:
                ret = DF_NCHW;
                break;
            default: {
                const google::protobuf::EnumDescriptor *descriptor = caffe::DataFormat_descriptor();
                UNI_ERROR_LOG("can not process operator name:%s %s format memory.\n",
                    this->layer.name().c_str(), descriptor->FindValueByNumber(df)->name().c_str());
            }
        }
        return ret;
    }

    // the first loop can specify the input info and output info
    EE adapt_operators(ModelSpec *ms) override
    {
        EE ret = SUCCESS;
        // model_name
        str_copy(ms->model_name, proto.name().c_str(), proto.name().length());
        ms->dt = DT_F32;  // set default value

        ms->num_operator_specs = proto.layer_size();
        OperatorSpec *opsPtr =
            (OperatorSpec *)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
            ms->ops[i].num_quant_feature = 0;
            ms->ops[i].feature_scale = nullptr;
        }

        int inputsNumber = 0;
        this->weightNumber = 0;
        std::map<std::string, int> outputCounts;
        std::set<std::string> sharedWeightCounts;
        for (int i = 0; i < proto.input_size(); i++) {
            outputCounts[proto.input(i).c_str()] = 1;
        }
        for (int i = 0; i < proto.layer_size(); i++) {
            if (proto.layer(i).type() == "SharedWeight") {
                sharedWeightCounts.insert(proto.layer(i).top(0));
            }
        }
        for (int i = 0; i < proto.layer_size(); i++) {
            const caffe::LayerParameter curLayer = proto.layer(i);
            this->layer = curLayer;
            UNI_DEBUG_LOG("process operator name:%s parameter.\n", this->layer.name().c_str());

            if (layer.type() == "Input") {  // layer,the global variable
                inputsNumber++;
            }
            str_copy(opsPtr[i].name, layer.name().c_str(), layer.name().length());

            this->op = layer.type();
            opsPtr[i].type = convert_caffe_type(layer.type());
            int bottomSize = layer.bottom_size();
            opsPtr[i].num_inputs = bottomSize;
            opsPtr[i].input_tensors_name = (I8 **)mt_malloc(bottomSize * sizeof(I8 *));
            for (int j = 0; j < bottomSize; j++) {
                opsPtr[i].input_tensors_name[j] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[i].input_tensors_name[j], layer.bottom(j).c_str(),
                    layer.bottom(j).length());
                if (outputCounts.find(layer.bottom(j)) == outputCounts.end()) {
                    if (opsPtr[i].type != OT_Jump) {
                        UNI_ERROR_LOG("no tensor is operator name:%s input %s.\n",
                            layer.name().c_str(), layer.bottom(j).c_str());
                    }
                } else {
                    outputCounts[layer.bottom(j)]--;
                }
            }
            int topSize = layer.top_size();
            opsPtr[i].num_outputs = topSize;
            opsPtr[i].output_tensors_name = (I8 **)mt_malloc(topSize * sizeof(I8 *));
            for (int j = 0; j < topSize; j++) {
                opsPtr[i].output_tensors_name[j] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                str_copy(
                    opsPtr[i].output_tensors_name[j], layer.top(j).c_str(), layer.top(j).length());
                if (outputCounts.find(layer.top(j)) == outputCounts.end()) {
                    outputCounts[layer.top(j)] = 1;
                } else {
                    outputCounts[layer.top(j)]++;
                }
            }

            CHECK_STATUS(adapt_operator(opsPtr[i].type, &(ms->ops[i].ps)));
            if (opsPtr[i].type == OT_MatMul && sharedWeightCounts.count(layer.bottom(1))) {
                this->weightNumber++;
            }
        }

        inputsNumber = (inputsNumber > proto.input_size()) ? inputsNumber : proto.input_size();
        ms->num_inputs = inputsNumber;
        ms->input_names = (I8 **)mt_malloc(inputsNumber * sizeof(I8 *));
        ms->input_dims = (TensorDesc *)mt_malloc(sizeof(TensorDesc) * inputsNumber);
        for (int i = 0; i < inputsNumber; i++) {
            ms->input_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            ms->input_dims[i] = tensor0d();
            if (proto.input_size() > 0) {
                str_copy(ms->input_names[i], proto.input(i).c_str(), proto.input(i).length());
                ms->input_dims[i].nDims = proto.input_dim_size();
                for (U32 j = 0; j < ms->input_dims[i].nDims; j++) {
                    ms->input_dims[i].dims[ms->input_dims[i].nDims - 1 - j] = proto.input_dim(j);
                }
            }
            if (i < proto.input_shape_size()) {
                str_copy(ms->input_names[i], proto.input(i).c_str(), proto.input(i).length());
                ms->input_dims[i].nDims = proto.input_shape(i).dim_size();
                for (U32 j = 0; j < ms->input_dims[i].nDims; j++) {
                    ms->input_dims[i].dims[ms->input_dims[i].nDims - 1 - j] =
                        proto.input_shape(i).dim(j);
                }
            }
            ms->input_dims[i].dt = DT_F32;
            ms->input_dims[i].df = getTensorDefaultDataFormat(ms->input_dims[i].nDims);
        }
        for (int i = 0; i < proto.output_size(); i++) {
            std::string name = proto.output(i);
            if (outputCounts.find(name) == outputCounts.end()) {
                UNI_ERROR_LOG("can not find model output %s in tensors.\n", name.c_str());
            } else {
                outputCounts[name] = (outputCounts[name] > 0) ? outputCounts[name] : 1;
            }
        }
        int outputsNumber = 0;
        for (auto iter : outputCounts) {
            if (iter.second > 0) {
                outputsNumber++;
            }
        }
        ms->num_outputs = outputsNumber;
        ms->output_names = (I8 **)mt_malloc(outputsNumber * sizeof(I8 *));
        outputsNumber = 0;
        for (auto iter : outputCounts) {
            if (iter.second > 0) {
                ms->output_names[outputsNumber] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                str_copy(ms->output_names[outputsNumber], iter.first.c_str(), iter.first.length());
                outputsNumber++;
            }
        }
        ms->num_weight_specs = this->weightNumber;
        return ret;
    }

    EE adapt_weights(ModelSpec *ms) override
    {
        EE ret = SUCCESS;
        WeightSpec *wsPtr = (WeightSpec *)mt_malloc(sizeof(WeightSpec) * ms->num_weight_specs);
        for (int j = 0; j < ms->num_weight_specs; j++) {
            wsPtr[j].num_quant_scale = 0;
            wsPtr[j].weight_scale = nullptr;
        }
        ms->ws = wsPtr;
        int inNamesIndex = 0;
        int weightIndex = 0;

        std::set<std::string> sharedWeightCounts;
        for (int i = 0; i < proto.layer_size(); i++) {
            if (proto.layer(i).type() == "SharedWeight") {
                sharedWeightCounts.insert(proto.layer(i).top(0));
            }
        }

        for (int i = 0; i < proto.layer_size(); i++) {
            this->layer = proto.layer(i);
            std::string layerName = layer.name();
            std::string layerType = layer.type();
            UNI_DEBUG_LOG(
                "process operator name:%s type:%s weight.\n", layerName.c_str(), layerType.c_str());

            if (layerType == "Input") {
                std::string dataName = layerName;
                if (layer.top_size() > 0) {
                    dataName = layer.top(0);
                }
                str_copy(ms->input_names[inNamesIndex], dataName.c_str(), dataName.length());
                ms->input_dims[inNamesIndex].nDims = layer.input_param().shape(0).dim_size();
                ms->input_dims[inNamesIndex].dt = DT_F32;
                ms->input_dims[inNamesIndex].df =
                    getTensorDefaultDataFormat(ms->input_dims[inNamesIndex].nDims);
                for (U32 j = 0; j < ms->input_dims[inNamesIndex].nDims; j++) {
                    ms->input_dims[inNamesIndex].dims[ms->input_dims[inNamesIndex].nDims - 1 - j] =
                        layer.input_param().shape(0).dim(j);
                }
                ms->input_dims[inNamesIndex].dt = get_type(layer.input_param().type());
                ms->input_dims[inNamesIndex].df = get_format(layer.input_param().format());
                inNamesIndex++;
            } else if (layerType == "Convolution" || layerType == "InnerProduct" ||
                layerType == "BatchNorm" || layerType == "Embed" || layerType == "LSTM" ||
                layerType == "SharedWeight" || layerType == "RelativePositionEmbed" ||
                layerType == "Deconvolution" || layerType == "PReLU") {
                int netLayerId = net_search_layerId(net, layerName);
                CHECK_REQUIREMENT(netLayerId >= 0);
                str_copy(wsPtr[weightIndex].op_name, layerName.c_str(), layerName.length());
                U32 blobNum = net_get_blobs_size(net, netLayerId);
                net_copy_blob(
                    wsPtr, weightIndex, net, netLayerId, blobNum, convert_caffe_type(layerType));

                if (layerType == "BatchNorm" && blobNum > 2) {
                    caffe::BlobProto blob2 = net_get_blob(net, netLayerId, 2);
                    float cur_gama = blob2.data().data()[0] == 0 ? 1.0
                                                                 : 1.0 / blob2.data().data()[0];
                    ms->ops[i].ps.bn_spec.gama = cur_gama;
                }

                weightIndex++;
            } else if (layerType == "Scale" || layerType == "LayerNorm") {
                int netLayerId = net_search_layerId(net, layerName);
                CHECK_REQUIREMENT(netLayerId >= 0);
                str_copy(wsPtr[weightIndex].op_name, layerName.c_str(), layerName.length());
                U32 blobNum = net_get_blobs_size(net, netLayerId);
                if (layer.bottom_size() == 1) {
                    CHECK_REQUIREMENT(blobNum >= 1);
                } else {
                    CHECK_REQUIREMENT(blobNum == 0);
                }
                net_copy_blob(
                    wsPtr, weightIndex, net, netLayerId, blobNum, convert_caffe_type(layerType));
                weightIndex++;
            } else if (layerType == "MatMul" &&
                (layer.bottom_size() > 0 &&
                    sharedWeightCounts.find(layer.bottom(1)) != sharedWeightCounts.end() &&
                    sharedWeightCounts.count(layer.bottom(1)))) {
                int netLayerId = net_search_layerId(net, layerName);
                CHECK_REQUIREMENT(netLayerId >= 0);
                str_copy(wsPtr[weightIndex].op_name, layerName.c_str(), layerName.length());
                net_copy_blob(wsPtr, weightIndex, net, netLayerId, 0, convert_caffe_type(layerType));
                weightIndex++;
            }
        }

        CHECK_REQUIREMENT(weightIndex == this->weightNumber);
        // relationship init null
        ms->num_op_tensor_entries = 0;
        ms->op_relationship_entries = nullptr;
        return ret;
    }

    ParameterSpec adapt_Resize() override
    {
        ParameterSpec ps;
        ResizeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (this->op == "Interp") {
            auto cp = layer.interp_param();
            if (cp.has_zoom_factor()) {
                p.zoom_factor = cp.zoom_factor();
                if (cp.has_pad_beg()) {
                    p.pad_begin = cp.pad_beg();
                }
                if (cp.has_pad_end()) {
                    p.pad_end = cp.pad_end();
                }
            } else {
                p.sizes[0] = cp.height();
                p.sizes[1] = cp.width();
                p.num_sizes = 2;
            }
            p.mode = RESIZE_LINEAR;
        } else {
            auto cp = layer.resize_param();
            U32 h = cp.height();
            U32 w = cp.width();
            F32 hs = cp.height_scale();
            F32 ws = cp.width_scale();
            if (h != 0 && w != 0) {
                p.num_sizes = 2;
                p.sizes[0] = h;
                p.sizes[1] = w;
            }
            if (hs != 0 && ws != 0) {
                p.num_scales = 2;
                p.scales[0] = hs;
                p.scales[1] = ws;
            }
            caffe::Interp_mode mode = (caffe::Interp_mode)(cp.interp_mode()[0]);
            p.mode = get_interp_mode(mode);
            p.trans_mode = COORDINATE_TRANS_ALIGN_CORNERS;
            p.round_mode = ROUND_FLOOR;
        }
        ps.resize_spec = p;
        return ps;
    }

    ParameterSpec adapt_Conv() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        ConvolutionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.convolution_param();
        p.num_outputs = cp.num_output();
        p.num_outputs_origin = p.num_outputs;
        p.kernel_t = 1;
        p.stride_t = 1;
        p.pad_before = 0;
        p.pad_after = 0;
        p.dilatedRate_t = 1;
        if (cp.has_kernel_w() && cp.has_kernel_h()) {
            p.kernel_w = cp.kernel_w();
            p.kernel_h = cp.kernel_h();
        } else {
            p.kernel_h = (cp.kernel_size_size() > 0) ? cp.kernel_size(0) : 1;
            p.kernel_w = (cp.kernel_size_size() > 1) ? cp.kernel_size(1) : p.kernel_h;
        }

        p.group = (cp.has_group()) ? cp.group() : 1;
        p.dilatedRate_h = (cp.dilation_size() != 0) ? cp.dilation(0) : 1;
        p.dilatedRate_w = p.dilatedRate_h;

        if (p.group != 1 && p.group == p.num_outputs) {
            p.convolution_type = CONVOLUTION_DEPTHWISE;
        } else {
            p.convolution_type = CONVOLUTION_POINTWISE;
        }
        p.dw_activation_type = ACTIVATION_NULL;
        p.pw_activation_type = ACTIVATION_NULL;
        if (cp.has_stride_w() && cp.has_stride_h()) {
            p.stride_w = cp.stride_w();
            p.stride_h = cp.stride_h();
        } else {
            p.stride_h = (cp.stride_size() != 0) ? cp.stride(0) : 1;  // stride[default=1]
            p.stride_w = (cp.stride_size() > 1) ? cp.stride(1) : p.stride_h;
        }
        if (cp.has_pad_w() && cp.has_pad_h()) {
            p.pad_left = cp.pad_w();
            p.pad_right = p.pad_left;
            p.pad_top = cp.pad_h();
            p.pad_bottom = p.pad_top;
        } else {
            p.pad_top = (cp.pad_size() > 0) ? cp.pad(0) : 0;
            p.pad_bottom = (cp.pad_size() > 1) ? cp.pad(1) : p.pad_top;
            p.pad_left = (cp.pad_size() > 2) ? cp.pad(2) : p.pad_top;
            p.pad_right = (cp.pad_size() > 3) ? cp.pad(3) : p.pad_top;
        }
        ps.conv_spec = p;
        return ps;
    }

    ParameterSpec adapt_Deconvolution() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        ConvolutionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.convolution_param();
        p.num_outputs = cp.num_output();
        p.num_outputs_origin = p.num_outputs;
        p.kernel_t = 1;
        p.stride_t = 1;
        p.pad_before = 0;
        p.pad_after = 0;
        p.dilatedRate_t = 1;
        p.output_pad_t = 0;
        if (cp.has_kernel_w() && cp.has_kernel_h()) {
            p.kernel_w = cp.kernel_w();
            p.kernel_h = cp.kernel_h();
        } else {
            p.kernel_h = cp.kernel_size(0);
            p.kernel_w = p.kernel_h;
        }

        p.group = (cp.has_group()) ? cp.group() : 1;
        if (1 != p.group) {
            UNI_ERROR_LOG(
                "can not process operator name:%s group != 1.", this->layer.name().c_str());
        }
        p.dilatedRate_h = 1;
        p.dilatedRate_w = 1;
        p.convolution_type = CONVOLUTION_DECONVOLUTION;
        p.dw_activation_type = ACTIVATION_NULL;
        p.pw_activation_type = ACTIVATION_NULL;
        if (cp.has_stride_w() && cp.has_stride_h()) {
            p.stride_w = cp.stride_w();
            p.stride_h = cp.stride_h();
        } else {
            p.stride_h = (cp.stride_size() != 0) ? cp.stride(0) : 1;
            p.stride_w = p.stride_h;
        }
        p.round_mode = ROUND_CEIL;
        if (cp.has_pad_w() && cp.has_pad_h()) {
            p.pad_left = cp.pad_w();
            p.pad_right = p.pad_left;
            p.pad_top = cp.pad_h();
            p.pad_bottom = p.pad_top;
        } else {
            p.pad_top = (cp.pad_size() != 0) ? cp.pad(0) : 0;
            p.pad_bottom = p.pad_top;
            p.pad_left = p.pad_top;
            p.pad_right = p.pad_top;
        }
        p.output_pad_h = 0;
        p.output_pad_w = 0;
        ps.conv_spec = p;
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
        auto cp = layer.pooling_param();
        if (cp.has_kernel_w() && cp.has_kernel_h()) {
            p.kernel_w = cp.kernel_w();
            p.kernel_h = cp.kernel_h();
        } else {
            p.kernel_h = cp.kernel_size();
            p.kernel_w = p.kernel_h;
        }
        if (cp.has_stride_w() && cp.has_stride_h()) {
            p.stride_w = cp.stride_w();
            p.stride_h = cp.stride_h();
        } else {
            p.stride_h = cp.stride();
            p.stride_w = p.stride_h;
        }
        bool global_pooling = cp.global_pooling();
        if (global_pooling) {
            p.kernel_h = 0;
            p.kernel_w = 0;
            p.stride_h = 1;
            p.stride_w = 1;
        } else {
            CHECK_REQUIREMENT(p.kernel_h > 0);
        }
        if (cp.has_pad_w() && cp.has_pad_h()) {
            p.pad_left = cp.pad_w();
            p.pad_right = p.pad_left;
            p.pad_top = cp.pad_h();
            p.pad_bottom = p.pad_top;
        } else {
            p.pad_top = cp.has_pad() ? cp.pad() : 0;
            p.pad_bottom = p.pad_top;
            p.pad_left = p.pad_top;
            p.pad_right = p.pad_top;
        }

        if (cp.has_round_mode() && cp.round_mode() == 1) {
            p.round_mode = ROUND_FLOOR;
        } else {
            p.round_mode = ROUND_CEIL;
        }
        auto op = cp.pool();
        switch (op) {
            case caffe::PoolingParameter_PoolMethod_MAX: {
                p.mode = POOLING_MAX;
                break;
            }
            case caffe::PoolingParameter_PoolMethod_AVE: {
                p.mode = POOLING_MEAN;
                break;
            }
            default: {
                const google::protobuf::EnumDescriptor *descriptor =
                    caffe::PoolingParameter::PoolMethod_descriptor();
                UNI_ERROR_LOG("can not map operator name:%s %s to Pooling.\n",
                    this->layer.name().c_str(), descriptor->FindValueByNumber(op)->name().c_str());
            }
        }
        p.count_include_pad = true;
        ps.pooling_spec = p;
        return ps;
    }

    ParameterSpec adapt_Fc() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        FullyConnectedParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.num_outputs = layer.inner_product_param().num_output();
        p.num_slices = 1;
        p.slice_point[0] = p.num_outputs;
        ps.fc_spec = p;
        return ps;
    }

    ParameterSpec adapt_BatchNorm() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        BatchNormParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.batch_norm_param();
        p.axis = cp.axis();
        p.eps = cp.eps();
        p.gama = 1;
        p.momentum = cp.moving_average_fraction();
        ps.bn_spec = p;
        return ps;
    }

    ParameterSpec adapt_LayerNorm() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        LayerNormParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = -1;   //layer.layer_norm_param().axis();
        p.eps = 1e-6;  //layer.layer_norm_param().eps();
        ps.ln_spec = p;
        return ps;
    }

    EltwiseMode get_eltwise_mode(const caffe::EltwiseOp &op)
    {
        EltwiseMode ret;
        switch (op) {
            case caffe::EltwiseOp::PROD:
                ret = ELTWISE_PROD;
                break;
            case caffe::EltwiseOp::SUM:
                ret = ELTWISE_SUM;
                break;
            case caffe::EltwiseOp::MAX:
                ret = ELTWISE_MAX;
                break;
            case caffe::EltwiseOp::DIV:
                ret = ELTWISE_DIV;
                break;
            default: {
                const google::protobuf::EnumDescriptor *descriptor = caffe::EltwiseOp_descriptor();
                UNI_ERROR_LOG("can not process operator name:%s %s.\n", this->layer.name().c_str(),
                    descriptor->FindValueByNumber(op)->name().c_str());
            }
        }
        return ret;
    }

    ResizeMode get_interp_mode(const caffe::Interp_mode &op)
    {
        ResizeMode ret;
        switch (op) {
            case caffe::Interp_mode::LINEAR:
                ret = RESIZE_LINEAR;
                break;
            case caffe::Interp_mode::NEAREST:
                ret = RESIZE_NEAREST;
                break;
            case caffe::Interp_mode::CUBIC:
                ret = RESIZE_CUBIC;
                break;
            default: {
                const google::protobuf::EnumDescriptor *descriptor = caffe::Interp_mode_descriptor();
                UNI_ERROR_LOG("can not process operator name:%s %s.\n", this->layer.name().c_str(),
                    descriptor->FindValueByNumber(op)->name().c_str());
            }
        }
        return ret;
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec ps;
        EltwiseParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.eltwise_param();
        p.mode = get_eltwise_mode(cp.operation());
        U32 bytes = cp.coeff_size() * sizeof(F32);
        p.sum_spec.num_coeff = cp.coeff_size();
        UNI_MEMCPY(p.sum_spec.coeff, cp.coeff().data(), bytes);
        for (int j = 0; j < cp.coeff_size(); j++) {
            CHECK_REQUIREMENT(p.sum_spec.coeff[j] == 1);
        }
        p.activation_type = ACTIVATION_NULL;
        ps.eltwise_spec = p;
        return ps;
    }

    ParameterSpec adapt_Embedding() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        EmbedParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.embed_param();
        p.num_inputs = cp.input_dim();
        p.num_outputs = cp.num_output();
        p.bias_term = cp.bias_term() == 0 ? false : true;
        p.transpose = cp.transpose() == 0 ? false : true;
        ps.embed_spec = p;
        return ps;
    }

    ParameterSpec adapt_Power() override
    {
        ParameterSpec ps;
        PowerParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.power_param();
        p.scale = cp.scale();
        p.shift = cp.shift();
        p.power = cp.power();
        ps.power_spec = p;
        return ps;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec ps;
        ReshapeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        if (this->op == "Flatten") {
            auto cp = layer.flatten_param();
            CHECK_REQUIREMENT(-1 == cp.end_axis());
            p.num_shape = cp.axis() + 1;
            for (I32 iter = 0; iter < p.num_shape - 1; iter++) {
                p.shape[iter] = 0;
            }
            p.shape[p.num_shape - 1] = -1;
            p.axis = 0;
            p.num_axes = -1;
        } else {
            auto cp = layer.reshape_param();
            p.num_shape = cp.shape().dim_size();
            for (I32 iter = 0; iter < cp.shape().dim_size(); iter++) {
                p.shape[iter] = cp.shape().dim(iter);
            }
            p.axis = cp.axis();
            p.num_axes = cp.num_axes();
        }
        ps.reshape_spec = p;
        return ps;
    }

    ParameterSpec adapt_Slice() override
    {
        ParameterSpec ps;
        SliceParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.slice_param();
        for (I32 iter = 0; iter < cp.slice_point().size(); iter++) {
            p.slice_points[iter] = cp.slice_point(iter);
        }
        p.num_slice = cp.slice_point().size();
        p.axis = cp.axis();
        ps.slice_spec = p;
        return ps;
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec ps;
        TransposeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.permute_param();
        for (I32 iter = 0; iter < cp.order().size(); iter++) {
            p.axes[iter] = cp.order(iter);
        }
        p.num_axes = cp.order().size();
        ps.transpose_spec = p;
        return ps;
    }

    ParameterSpec adapt_Tile() override
    {
        ParameterSpec ps;
        TileParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.tile_param();
        for (int i = 0; i < 8; ++i) {
            p.repeats[i] = 1;
        }
        p.num_repeats = 1;
        p.axis = cp.axis();
        p.repeats[0] = cp.tiles();
        ps.tile_spec = p;
        return ps;
    }

    ParameterSpec adapt_Pad() override
    {
        ParameterSpec ps;
        PadParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.padding_param();
        p.before = 0;
        p.after = 0;
        p.top = cp.shape(0);
        p.bottom = cp.shape(1);
        p.left = cp.shape(2);
        p.right = cp.shape(3);
        p.constant_value = 0;
        p.pad_mode = PAD_CONSTANT;
        ps.pad_spec = p;
        return ps;
    }

    ParameterSpec adapt_Attention() override
    {
        ParameterSpec ps;
        AttentionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.attention_param();
        p.num_heads = cp.num_heads();
        p.from_sequence_length = cp.from_sequence_length();
        p.to_sequence_length = cp.to_sequence_length();
        ps.attention_spec = p;
        return ps;
    }

    ParameterSpec adapt_RNN() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        RNNParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.lstm_param();
        p.mode = RNN_LSTM;
        p.num_outputs = cp.num_output();
        p.steps = cp.steps();
        if (p.steps == -2) {
            p.steps = 0;
            p.bi_direction = true;
        } else {
            p.bi_direction = false;
        }
        p.num_projection = cp.num_proj();
        p.zoneout_cell = cp.zoneout_cell();
        p.zoneout_output = cp.zoneout_output();
        p.forget_bias = 1.0;
        p.activation_type = ACTIVATION_TANH;
        ps.rnn_spec = p;
        return ps;
    }

    ParameterSpec adapt_Scale() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        ScaleParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = layer.scale_param().axis();
        ps.scale_spec = p;
        return ps;
    }

    ParameterSpec adapt_Reduction() override
    {
        ParameterSpec ps;
        ReductionParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.reduction_param();
        p.axes[0] = cp.axis();
        p.num_axes = 1;
        auto op = cp.operation();
        switch (op) {
            case caffe::ReductionParameter_ReductionOp_SUM:
                p.mode = REDUCTION_SUM;
                break;
            case caffe::ReductionParameter_ReductionOp_MEAN:
                p.mode = REDUCTION_MEAN;
                break;
            default: {
                const google::protobuf::EnumDescriptor *descriptor =
                    caffe::ReductionParameter::ReductionOp_descriptor();
                UNI_ERROR_LOG("can not map operator name:%s %s to Reduction.\n",
                    this->layer.name().c_str(), descriptor->FindValueByNumber(op)->name().c_str());
            }
        }
        p.coeff = cp.coeff();
        p.keep_dim = cp.keep_dim();
        ps.reduction_spec = p;
        return ps;
    }

    ParameterSpec adapt_Squeeze() override
    {
        ParameterSpec ps;
        SqueezeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axes[0] = layer.squeeze_param().axis();
        p.num_axes = 1;
        ps.squeeze_spec = p;
        return ps;
    }

    ParameterSpec adapt_Unsqueeze() override
    {
        ParameterSpec ps;
        UnsqueezeParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axes[0] = layer.unsqueeze_param().axis();
        p.num_axes = 1;
        ps.unsqueeze_spec = p;
        return ps;
    }

    ParameterSpec adapt_ArgMax() override
    {
        ParameterSpec ps;
        ArgMaxParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = layer.argmax_param().axis();
        ps.argmax_spec = p;
        return ps;
    }

    ParameterSpec adapt_Repeat() override
    {
        ParameterSpec ps;
        RepeatParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.repeat_param();
        p.loops = cp.loops();
        p.axis = cp.axis();
        ps.repeat_spec = p;
        return ps;
    }

    ParameterSpec adapt_Check() override
    {
        ParameterSpec ps;
        CheckParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.check_param();
        auto op = cp.operation();
        switch (op) {
            case caffe::CheckParameter_CheckOp_EQUAL:
                p.mode = CHECK_EQUAL;
                break;
            case caffe::CheckParameter_CheckOp_GREAT:
                p.mode = CHECK_GREATER;
                break;
            case caffe::CheckParameter_CheckOp_GREATEQUAL:
                p.mode = CHECK_GREATER_EQUAL;
                break;
            default: {
                const google::protobuf::EnumDescriptor *descriptor =
                    caffe::CheckParameter::CheckOp_descriptor();
                UNI_ERROR_LOG("can not map operator name:%s %s to Check.\n",
                    this->layer.name().c_str(), descriptor->FindValueByNumber(op)->name().c_str());
            }
        }
        ps.check_spec = p;
        return ps;
    }

    ParameterSpec adapt_PreAllocatedMemory() override
    {
        ParameterSpec ps;
        PreAllocatedMemoryParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.preallocated_memory_param();
        p.desc.nDims = cp.shape().dim_size();
        for (I32 iter = 0; iter < cp.shape().dim_size(); iter++) {
            p.desc.dims[p.desc.nDims - 1 - iter] = cp.shape().dim(iter);
        }
        p.desc.df = getTensorDefaultDataFormat(p.desc.nDims);
        p.desc.dt = get_type(cp.data_type());
        p.value = cp.value();
        ps.preallocated_memory_spec = p;
        return ps;
    }

    ParameterSpec adapt_SharedWeight() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        SharedWeightParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.shared_weight_param();
        p.desc.nDims = cp.shape().dim_size();
        for (I32 iter = 0; iter < cp.shape().dim_size(); iter++) {
            p.desc.dims[p.desc.nDims - 1 - iter] = cp.shape().dim(iter);
        }
        p.desc.df = getTensorDefaultDataFormat(p.desc.nDims);
        p.desc.dt = get_type(cp.data_type());
        ps.shared_weight_spec = p;
        return ps;
    }

    ParameterSpec adapt_Copy() override
    {
        ParameterSpec ps;
        CopyParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.copy_param();
        p.src_dims[0] = cp.src_batch_stride();
        p.src_dims[1] = cp.src_stride();
        p.src_dims[2] = cp.src_offset();
        p.dst_dims[0] = cp.dst_batch_stride();
        p.dst_dims[1] = cp.dst_stride();
        p.dst_dims[2] = cp.dst_offset();
        p.length = cp.length();
        ps.copy_spec = p;
        return ps;
    }

    ParameterSpec adapt_MatMul() override
    {
        ParameterSpec ps;
        MatMulParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.matmul_param();
        p.transpose_a = cp.transpose_a();
        p.transpose_b = cp.transpose_b();
        ps.matmul_spec = p;
        return ps;
    }

    ParameterSpec adapt_AttentionMask() override
    {
        ParameterSpec ps;
        AttentionMaskParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.attention_mask_param();
        p.attention_length = cp.attention_length();
        p.same_length = cp.same_length();
        p.mask = cp.mask();
        ps.attention_mask_spec = p;
        return ps;
    }

    ParameterSpec adapt_RelativePositionEmbedding() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        EmbedParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.relative_position_embed_param();
        p.num_inputs = cp.input_dim();
        p.num_outputs = cp.num_output();
        p.bias_term = cp.bias_term() == 0 ? false : true;
        p.transpose = cp.transpose() == 0 ? false : true;
        p.axis = cp.axis();
        ps.embed_spec = p;
        return ps;
    }

    ParameterSpec adapt_RelativeShift() override
    {
        ParameterSpec ps;
        RelativeShiftParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.relative_shift_param();
        p.axis = cp.axis();
        p.shift_length = cp.shift_length();
        ps.relative_shift_spec = p;
        return ps;
    }

    ParameterSpec adapt_Concat() override
    {
        ParameterSpec ps;
        ConcatParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = layer.concat_param().axis();
        ps.concat_spec = p;
        return ps;
    }

    ParameterSpec adapt_Softmax() override
    {
        ParameterSpec ps;
        SoftmaxParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = layer.softmax_param().axis();
        ps.softmax_spec = p;
        return ps;
    }

    ParameterSpec adapt_PriorBox() override
    {
        ParameterSpec ps;
        PriorBoxParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.prior_box_param();
        CHECK_REQUIREMENT(cp.min_size_size() <= 2 && cp.max_size_size() <= 2);
        for (int i = 0; i < 2; i++) {
            p.min_sizes[i] = 0;
            if (i < cp.min_size_size()) {
                p.min_sizes[i] = cp.min_size(i);
            }
        }
        for (int i = 0; i < 2; i++) {
            p.max_sizes[i] = 0;
            if (i < cp.max_size_size()) {
                p.max_sizes[i] = cp.max_size(i);
            }
        }
        CHECK_REQUIREMENT(cp.aspect_ratio_size() <= 2);
        for (int i = 0; i < 2; i++) {
            p.aspect_ratios[i] = 0;
            if (i < cp.aspect_ratio_size()) {
                p.aspect_ratios[i] = cp.aspect_ratio(i);
            }
        }
        if (cp.has_flip()) {
            if (cp.flip()) {
                p.flip = 1;
            } else {
                p.flip = 0;
            }
        } else {
            p.flip = 1;
        }
        if (cp.has_clip()) {
            if (cp.clip()) {
                p.clip = 1;
            } else {
                p.clip = 0;
            }
        } else {
            p.clip = 0;
        }
        if (cp.variance_size() == 4) {
            p.variances[0] = cp.variance(0);
            p.variances[1] = cp.variance(1);
            p.variances[2] = cp.variance(2);
            p.variances[3] = cp.variance(3);
        } else if (cp.variance_size() == 1) {
            p.variances[0] = cp.variance(0);
            p.variances[1] = cp.variance(0);
            p.variances[2] = cp.variance(0);
            p.variances[3] = cp.variance(0);
        } else {
            p.variances[0] = 0.1;
            p.variances[1] = 0.1;
            p.variances[2] = 0.1;
            p.variances[3] = 0.1;
        }
        p.image_w = 0;
        p.image_h = 0;
        if (cp.has_img_size()) {
            p.image_w = cp.img_size();
            p.image_h = cp.img_size();
        }
        if (cp.has_img_w() && cp.has_img_h()) {
            p.image_w = cp.img_w();
            p.image_h = cp.img_h();
        }
        p.step_w = 0;
        p.step_h = 0;
        if (cp.has_step()) {
            p.step_w = cp.step();
            p.step_h = cp.step();
        }
        if (cp.has_step_w() && cp.has_step_h()) {
            p.step_w = cp.step_w();
            p.step_h = cp.step_h();
        }
        p.offset = cp.offset();
        ps.prior_box_spec = p;
        return ps;
    }

    ParameterSpec adapt_DetectionOutput() override
    {
        ParameterSpec ps;
        DetectionOutputParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.detection_output_param();
        p.num_class = cp.num_classes();
        CHECK_REQUIREMENT((cp.background_label_id() == 0) && (cp.share_location() == true));
        p.nms_threshold = cp.nms_param().nms_threshold();
        p.nms_top_k = cp.nms_param().top_k();
        p.keep_top_k = cp.keep_top_k();
        p.confidence_threshold = cp.confidence_threshold();
        ps.detection_output_spec = p;
        return ps;
    }

    ParameterSpec adapt_Yolov3DetectionOutput() override
    {
        ParameterSpec ps;
        Yolov3DetectionOutputParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.yolov3_detection_output_param();
        p.num_class = cp.num_classes();
        p.num_box = cp.num_box();
        p.confidence_threshold = cp.confidence_threshold();
        p.nms_threshold = cp.nms_threshold();
        for (int i = 0; i < 18; i++) {
            p.biases[i] = 0;
            if (i < cp.biases_size()) {
                p.biases[i] = cp.biases(i);
            }
        }
        for (int i = 0; i < 3; i++) {
            p.anchors_scale[i] = 0;
            if (i < cp.anchors_scale_size()) {
                p.anchors_scale[i] = cp.anchors_scale(i);
            }
        }
        p.mask_group_num = cp.mask_group_num();
        for (int i = 0; i < 9; i++) {
            p.mask[i] = 0;
            if (i < cp.mask_size()) {
                p.mask[i] = cp.mask(i);
            }
        }
        ps.yolov3_detection_output_spec = p;
        return ps;
    }

    ParameterSpec adapt_Clip() override
    {
        ParameterSpec ps;
        ClipParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        auto cp = layer.clip_param();
        p.min = cp.min();
        p.max = cp.max();
        ps.clip_spec = p;
        return ps;
    }

    ParameterSpec adapt_Relu() override
    {
        ParameterSpec ps;
        ReLUParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.neg_slope = layer.relu_param().negative_slope();
        ps.relu_spec = p;
        return ps;
    }

    ParameterSpec adapt_PRelu() override
    {
        this->weightNumber++;
        ParameterSpec ps;
        return ps;
    }

    ParameterSpec adapt_Exp() override
    {
        ParameterSpec ps;
        auto cp = layer.exp_param();
        if (cp.base() != -1 || cp.scale() != 1 || cp.shift() != 0) {
            UNI_ERROR_LOG("can not process operator name:%s base!=-1(e), scale!=1, shift!=0.\n",
                this->layer.name().c_str());
        }
        return ps;
    }

    ParameterSpec adapt_BilateralSliceApply() override
    {
        ParameterSpec ps;
        BilateralSliceApplyParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        std::string mode = layer.bilateral_slice_apply_param().mode();
        if (mode == "null") {
            p.mode = BILATERAL_SLICE_APPLY_NULL;
            if (layer.bottom_size() != 3) {
                UNI_ERROR_LOG("BilateralSliceApply need 3 inputs(input, grid, guide) under mode == "
                              "'null'. If you want to integrate guide calculation into big "
                              "operator, you can use 'conv' mode and that need 2 inputs.\n");
            } else {
                UNI_WARNING_LOG("We provide BilateralSliceApply big operator by using 'conv' mode, "
                                "It's relatively faster than 'null' mode.\n");
            }
        } else {
            if (layer.bottom_size() != 2) {
                UNI_ERROR_LOG("BilateralSliceApply only need 2 inputs(input, grid) under mode == "
                              "'conv'. If you don't want to integrate guide calculation into big "
                              "operator, you can use 'null' mode and that need 3 inputs.\n");
            } else {
                UNI_WARNING_LOG("BilateralSliceApply will use inner guide calculation function. If "
                                "you want to change implementation, you can modify "
                                "compute/image/src/gpu/mali/cl/bilateral_slice_apply_c12.cl for "
                                "GPU, or compute/image/src/cpu/bilateral_slice_apply.cpp for "
                                "CPU.\n");
            }
            p.mode = BILATERAL_SLICE_APPLY_CONV;
        }
        ps.bilateral_slice_apply_spec = p;
        return ps;
    }

    ParameterSpec adapt_ConvertColor() override
    {
        ParameterSpec ps;
        ConvertColorParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.src = get_color(layer.convert_color_param().src());
        p.dst = get_color(layer.convert_color_param().dst());
        p.dt = get_type(layer.convert_color_param().dt());
        ps.convert_color_spec = p;
        return ps;
    }

    ParameterSpec adapt_Cum() override
    {
        ParameterSpec ps;
        CumParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.mode = get_eltwise_mode(layer.cum_param().operation());
        p.exclusive = layer.cum_param().exclusive();
        p.reverse = layer.cum_param().reverse();
        p.axis = layer.cum_param().axis();
        ps.cum_spec = p;
        return ps;
    }

    ParameterSpec adapt_OneHot() override
    {
        ParameterSpec ps;
        OneHotParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.axis = layer.onehot_param().axis();
        p.depth = layer.onehot_param().depth();
        p.values[0] = layer.onehot_param().off_value();
        p.values[1] = layer.onehot_param().on_value();
        ps.onehot_spec = p;
        return ps;
    }

    ParameterSpec adapt_Lut() override
    {
        ParameterSpec ps;
        LutParamSpec p;
        UNI_MEMSET(&p, 0, sizeof(p));
        p.type = get_color(layer.lut_param().type());
        p.mode = get_interp_mode(layer.lut_param().mode());
        ps.lut_spec = p;
        return ps;
    }

private:
    std::string op;
    caffe::NetParameter proto;
    caffe::NetParameter net;
    caffe::LayerParameter layer;
    int weightNumber;
};
#endif
