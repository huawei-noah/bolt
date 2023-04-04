// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "model_common.h"
#include "file.h"

template <typename T>
inline void serialize_field(U8 **buffer, U32 *position, const T *element, int length = 1)
{
    int size = length * sizeof(T);
    UNI_MEMCPY(*buffer, element, size);
    *buffer += size;
    *position += size;
}

EE serialize_header(const ModelSpec *spec, std::string *tmp)
{
    U32 bufSize = sizeof(I32) * 2 + sizeof(I8) * NAME_LEN + sizeof(DataType) + sizeof(I32) +
        sizeof(I8) * NAME_LEN * spec->num_inputs + sizeof(TensorDesc) * spec->num_inputs +
        sizeof(I32) + sizeof(I8) * NAME_LEN * spec->num_outputs;
    U8 *buf = (U8 *)mt_malloc(bufSize);
    U8* _pointer = buf;
    U32 _pos = 0;
    U8** pointer = &_pointer;
    U32 *pos = &_pos;

    serialize_field<I32>(pointer, pos, &spec->version);
    serialize_field<I32>(pointer, pos, &spec->magic_number);
    serialize_field<I8>(pointer, pos, spec->model_name, NAME_LEN);
    serialize_field<DataType>(pointer, pos, &spec->dt);

    serialize_field<I32>(pointer, pos, &spec->num_inputs);
    for (I32 i = 0; i < spec->num_inputs; i++) {
        serialize_field<I8>(pointer, pos, spec->input_names[i], NAME_LEN);
    }
    for (I32 i = 0; i < spec->num_inputs; i++) {
        serialize_field<TensorDesc>(pointer, pos, spec->input_dims + i, 1);
    }
    serialize_field<I32>(pointer, pos, &spec->num_outputs);
    for (I32 i = 0; i < spec->num_outputs; i++) {
        serialize_field<I8>(pointer, pos, spec->output_names[i], NAME_LEN);
    }
    CHECK_REQUIREMENT(*pos == bufSize);
    CHECK_REQUIREMENT((U32)(*pointer - buf) == bufSize);

    tmp->clear();
    tmp->assign(buf, buf + bufSize);
    mt_free(buf);
    return SUCCESS;
}

U32 get_operator_size(const OperatorSpec &op)
{
    // sizeof(U32) * 4 : type + num_inputs + num_output + num_quant_feature
    U32 allocatedBufferSize = sizeof(I8) * NAME_LEN + sizeof(U32) * 3 + sizeof(OperatorType) + 
        op.num_inputs * NAME_LEN * sizeof(I8) + op.num_outputs * NAME_LEN * sizeof(I8) +
        (op.num_inputs + op.num_outputs) * sizeof(I32) +
        get_operator_parameter_size(sg_boltVersion, op.type);
    for (U32 i = 0; i < op.num_quant_feature; i++) {
        allocatedBufferSize += sizeof(I32);  // num_scale
        allocatedBufferSize += op.feature_scale[i].num_scale * sizeof(F32);
    }
    return allocatedBufferSize;
}

EE serialize_operators(const ModelSpec *spec, std::string *tmp)
{
    I32 removeOpNum = 0;
    U32 bufSize = sizeof(I32);
    auto p = spec->ops;
    for (I32 i = 0; i < spec->num_operator_specs; i++) {
        if (isDeprecatedOp(p[i].type)) {
            removeOpNum++;
        } else {
            bufSize += get_operator_size(p[i]);
        }
    }
    U8 *buf = (U8 *)mt_malloc(bufSize);
    U8* _pointer = buf;
    U32 _pos = 0;
    U8** pointer = &_pointer;
    U32 *pos = &_pos;

    I32 num = spec->num_operator_specs - removeOpNum;
    serialize_field<I32>(pointer, pos, &num);
    for (int i = 0; i < spec->num_operator_specs; i++) {
        if (isDeprecatedOp(p[i].type)) {
            continue;
        }

        serialize_field<I8>(pointer, pos, p[i].name, NAME_LEN);
        serialize_field<OperatorType>(pointer, pos, &(p[i].type));
        serialize_field<U32>(pointer, pos, &(p[i].num_inputs));
        for (U32 j = 0; j < p[i].num_inputs; j++) {
            serialize_field<I8>(pointer, pos, p[i].input_tensors_name[j], NAME_LEN);
        }
        serialize_field<U32>(pointer, pos, &(p[i].num_outputs));
        for (U32 j = 0; j < p[i].num_outputs; j++) {
            serialize_field<I8>(pointer, pos, p[i].output_tensors_name[j], NAME_LEN);
        }

        U32 numTensors = p[i].num_inputs + p[i].num_outputs;
        if (nullptr != p[i].tensor_positions) {
            serialize_field<I32>(pointer, pos, p[i].tensor_positions, numTensors);
        } else {
            for (U32 j = 0; j < numTensors; j++) {
                I32 t = -1;
                serialize_field<I32>(pointer, pos, &t);
            }
        }

        serialize_field<U32>(pointer, pos, &(p[i].num_quant_feature));
        for (U32 j = 0; j < p[i].num_quant_feature; j++) {
            serialize_field<I32>(pointer, pos, &(p[i].feature_scale[j].num_scale));
            serialize_field<F32>(pointer, pos, p[i].feature_scale[j].scale, p[i].feature_scale[j].num_scale);
        }

        serialize_field<U8>(pointer, pos, (U8*)&(p[i].ps), get_operator_parameter_size(sg_boltVersion, p[i].type));
    }

    CHECK_REQUIREMENT(*pos == bufSize);
    CHECK_REQUIREMENT((U32)(*pointer - buf) == bufSize);
    tmp->clear();
    tmp->assign(buf, buf + bufSize);
    mt_free(buf);
    return SUCCESS;
}

EE serialize_weights(const ModelSpec *spec, std::string *tmp)
{
    I32 removeWeightNum = 0;
    U32 bufSize = sizeof(I32);
    auto p = spec->ws;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        if (isDeprecatedOpWeight(spec, i)) {
            removeWeightNum++;
            continue;
        }

        // U32 x 5: length, mdt, bytes_of_weight, bytes_of_vec, num_quant_scale
        bufSize += sizeof(I8) * NAME_LEN + sizeof(DataType) + sizeof(U32) * 4 + p[i].bytes_of_weight +
            p[i].bytes_of_vec;
        for (U32 j = 0; j < p[i].num_quant_scale; j++) {
            bufSize += sizeof(I32);  // num_scale
            bufSize += p[i].weight_scale[j].num_scale * sizeof(F32);
        }
    }
    U8 *buf = (U8 *)mt_malloc(bufSize);
    U8* _pointer = buf;
    U32 _pos = 0;
    U8** pointer = &_pointer;
    U32 *pos = &_pos;

    I32 num = spec->num_weight_specs - removeWeightNum;
    serialize_field<I32>(pointer, pos, &num);
    for (int i = 0; i < spec->num_weight_specs; i++) {
        if (isDeprecatedOpWeight(spec, i)) {
            continue;
        }

        U32 length = p[i].bytes_of_weight + p[i].bytes_of_vec;
        serialize_field<U32>(pointer, pos, &length);

        serialize_field<I8>(pointer, pos, p[i].op_name, NAME_LEN);
        serialize_field<DataType>(pointer, pos, &(p[i].mdt));
        serialize_field<U32>(pointer, pos, &(p[i].bytes_of_weight));
        serialize_field<U8>(pointer, pos, p[i].weight, p[i].bytes_of_weight);
        serialize_field<U32>(pointer, pos, &(p[i].bytes_of_vec));
        serialize_field<U8>(pointer, pos, p[i].vec, p[i].bytes_of_vec);

        serialize_field<U32>(pointer, pos, &(p[i].num_quant_scale));
        for (U32 j = 0; j < p[i].num_quant_scale; j++) {
            serialize_field<I32>(pointer, pos, &(p[i].weight_scale[j].num_scale));
            serialize_field<F32>(pointer, pos, p[i].weight_scale[j].scale, p[i].weight_scale[j].num_scale);
        }
    }

    CHECK_REQUIREMENT(*pos == bufSize);
    CHECK_REQUIREMENT((U32)(*pointer - buf) == bufSize);
    tmp->clear();
    tmp->assign(buf, buf + bufSize);
    mt_free(buf);
    return SUCCESS;
}

EE serialize_model(const ModelSpec *spec, std::string *bytes)
{
    bytes->clear();
    std::string tmp;

    CHECK_STATUS(serialize_header(spec, &tmp));
    *bytes += tmp;

    CHECK_STATUS(serialize_operators(spec, &tmp));
    *bytes += tmp;

    CHECK_STATUS(serialize_weights(spec, &tmp));
    *bytes += tmp;
    return SUCCESS;
}

EE write_to_file(std::string *bytes, const char *filePath)
{
    if (bytes == NULL || filePath == NULL) {
        return NULL_POINTER;
    }
    return save_binary(filePath, bytes->c_str(), bytes->size());
}

EE serialize_model_to_file(const ModelSpec *spec, const char *filePath)
{
    UNI_DEBUG_LOG("Write bolt model to %s...\n", filePath);
    std::string bytes = "";
    EE ret = serialize_model(spec, &bytes);
    if (ret == SUCCESS) {
        ret = write_to_file(&bytes, filePath);
    }
    UNI_DEBUG_LOG("Write bolt model end.\n");
    return SUCCESS;
}
