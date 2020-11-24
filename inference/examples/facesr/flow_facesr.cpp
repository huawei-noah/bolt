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
#include "task.h"
#include "flow.h"

DataType inferencePrecision = DT_F16;

std::map<std::string, std::shared_ptr<Tensor>> inputOutput()
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    TensorDesc inputDesc = tensor4df(inferencePrecision, DF_NCHW, 1, 64, 48, 48);
    tensors["geninput"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["geninput"]->resize(inputDesc);
    tensors["geninput"]->alloc();

    switch (inferencePrecision) {
        case DT_F32: {
            F32 *ptr = (F32 *)((CpuMemory *)tensors["geninput"]->get_memory())->get_ptr();
            for (U32 i = 0; i < tensorNumElements(inputDesc); i++) {
                ptr[i] = 1;
            }
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            F16 *ptr = (F16 *)((CpuMemory *)tensors["geninput"]->get_memory())->get_ptr();
            for (U32 i = 0; i < tensorNumElements(inputDesc); i++) {
                ptr[i] = 1;
            }
            break;
        }
#endif
        default:
            UNI_ERROR_LOG("currently not support to init this data type(%d) facesr input data\n",
                inferencePrecision);
            break;
    }

    tensors["pixel_shuffle_final_out"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["pixel_shuffle_final_out"]->resize(
        tensor4df(inferencePrecision, DF_NCHWC8, 1, 8, 384, 384));
    tensors["pixel_shuffle_final_out"]->alloc();
    return tensors;
}

int main(int argc, char *argv[])
{
    int num = 100;
    std::string facesrGraphPath = argv[1];
    std::vector<std::string> graphPath = {facesrGraphPath};
    int threads = atoi(argv[2]);

    Flow flowExample;
    flowExample.init(graphPath, inferencePrecision, AFFINITY_CPU_HIGH_PERFORMANCE, threads, false);
    sleep(10);

    for (int i = 0; i < num; i++) {
        std::map<std::string, std::shared_ptr<Tensor>> data = inputOutput();
        Task task(facesrGraphPath, data);
        flowExample.enqueue(task);
    }

    std::vector<Task> results;
    double start = ut_time_ms();
    UNI_PROFILE(results = flowExample.dequeue(true), std::string("flow_facesr"),
        std::string("flow_facesr"));
    double end = ut_time_ms();
    UNI_CI_LOG("avg_time:%fms/image\n", (end - start) / num);
    return 0;
}
