// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_MODELSERIALIZEDESERIALIZE
#define _H_MODELSERIALIZEDESERIALIZE

#include "model_tools.h"
#include <string>


EE serialize_header(const ModelSpec* spec, std::string* tmp);

EE serialize_operators(const ModelSpec* spec, std::string* tmp);

EE serialize_weights(const ModelSpec* spec, std::string* tmp);

EE serialize_model(const ModelSpec* spec, std::string* bytes);

EE write_to_file(std::string* bytes, const char* fn);

EE serialize_model_to_file(const ModelSpec* spec, const char* fn);

EE deserialize_header(char* bytes, ModelSpec* spec, U32* pos);

EE deserialize_operator(char* bytes, ModelSpec* spec, U32* pos);

EE deserialize_weight(char* bytes, ModelSpec* spec, U32* pos);

EE opeator_relationship(ModelSpec* spec);

EE deserialize_model(char* bytes, ModelSpec* spec);

EE read_from_file(const char* fn, std::string* bytes);

EE deserialize_model_from_file(const char* fn, ModelSpec* spec);

EE str_copy(I8 *dst, const I8 *src, I32 src_len);

void* mt_new_storage(size_t size);

EE ms_datatype_converter(ModelSpec* original_ms, ModelSpec* target_ms, DataConvertType convert_mode);

#endif
