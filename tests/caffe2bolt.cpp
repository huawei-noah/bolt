// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <string>
#include "model_tools.h"
#include "model_serialize_deserialize.hpp"
#include "model_optimizer.hpp"
#include "converter.h"
#include "model_print.h"

int main(int argc, char* argv[]) {
    std::string dir = std::string(argv[1]);
    std::string mfn = std::string(argv[2]);
    InferencePrecision ip = FP16;
    DataConvertType converterMode = F32_to_F16;
    if (argc > 3) {
        if (std::string(argv[3]) == std::string("INT8_Q")) {
            ip = INT8_Q;
            converterMode = F32_to_F16;
        } else if (std::string(argv[3]) == std::string("FP32")) {
            ip = FP32;
            converterMode = F32_to_F32;
        }
    }

    ModelSpec originalMs, targetMs, resultMs;
    CHECK_STATUS(mt_create_model(&originalMs));
    CHECK_STATUS(mt_create_model(&targetMs));
    CHECK_STATUS(mt_create_model(&resultMs));

    CHECK_STATUS(caffe_converter(dir, mfn, &originalMs));

     //graph_optimizer
    ModelSpecOptimizer ms_optimizer;
    ms_optimizer.suggest();
    ms_optimizer.optimize(&originalMs);
#ifdef _DEBUG
    print_ms(originalMs);
#endif
    //datatype converter
    CHECK_STATUS(ms_datatype_converter(&originalMs, &targetMs, converterMode));
#ifdef _DEBUG
    print_ms(targetMs);
#endif

    //serialize ms to ./bolt
    std::string modelStorePath = std::string(argv[1]) + "/" + std::string(argv[2]);
    switch (ip) {
        case INT8_Q: {
            modelStorePath += std::string("_int8_q.bolt");
            targetMs.dt = DT_F16_8Q;
            CHECK_STATUS(serialize_model_to_file(&targetMs, modelStorePath.c_str()));
            break;
        }
        case FP16: {
            modelStorePath += std::string("_f16.bolt");
            CHECK_STATUS(serialize_model_to_file(&targetMs, modelStorePath.c_str()));
            break;
        }
        case FP32: {
            modelStorePath += std::string("_f32.bolt");
            CHECK_STATUS(serialize_model_to_file(&targetMs, modelStorePath.c_str()));
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
            return 0;
        }
    }

    //deserialize ./bolt to ms in memory
    CHECK_STATUS(deserialize_model_from_file(modelStorePath.c_str(), &resultMs));
    print_ms(resultMs);

    CHECK_STATUS(mt_destroy_model(&originalMs));
    CHECK_STATUS(mt_destroy_model(&targetMs));
    CHECK_STATUS(mt_destroy_model(&resultMs));

    return 0;
}
