// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MODEL_SERIALIZE_DESERIALIZE
#define _H_MODEL_SERIALIZE_DESERIALIZE

#include <string>
#include "types.h"

int get_operator_parameter_size(OperatorType operatorType);

#if defined(_BUILD_TEST) || defined(_USE_CAFFE) || defined(_USE_ONNX) || defined(_USE_TFLITE) || \
    defined(_USE_TENSORFLOW)
EE serialize_header(const ModelSpec *spec, std::string *tmp);

EE serialize_operators(const ModelSpec *spec, std::string *tmp);

EE serialize_weights(const ModelSpec *spec, std::string *tmp);

EE serialize_model(const ModelSpec *spec, std::string *bytes);

EE write_to_file(std::string *bytes, const char *fn);

EE serialize_model_to_file(const ModelSpec *spec, const char *fn);

EE ms_datatype_converter(ModelSpec *original_ms,
    ModelSpec *target_ms,
    DataConvertType convert_mode,
    std::string storageMode);
#endif

EE deserialize_header(char *bytes, ModelSpec *spec, U32 *pos);

EE deserialize_operator(char *bytes, ModelSpec *spec, U32 *pos);

EE deserialize_weight(char *bytes, ModelSpec *spec, U32 *pos);

EE operator_relationship(ModelSpec *spec);

EE deserialize_model_from_file(const char *fn, ModelSpec *spec, bool useFileStream = false);

inline std::string concat_dir_file(std::string dir, std::string file)
{
    std::string ret;
    if (!dir.empty()) {
        int len = dir.size();
        char &last = dir.at(len - 1);
        if ('/' != last) {
            ret = dir + '/';
        } else {
            ret = dir;
        }
        ret += file;
    } else {
        ret = file;
    }

    return ret;
}
#endif
