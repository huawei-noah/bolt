// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "task.h"
#include "flow.h"

DataType inferencePrecision = DT_F16;

std::map<std::string, std::shared_ptr<Tensor>> inputOutput()
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    TensorDesc inputDesc = tensor4df(inferencePrecision, DF_NCHW, 1, 3, 128, 128);
    tensors["input"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["input"]->resize(inputDesc);
    tensors["input"]->alloc();

    switch (inferencePrecision) {
        case DT_F32: {
            F32 *ptr = (F32 *)((CpuMemory *)tensors["input"]->get_memory())->get_ptr();
            for (U32 i = 0; i < tensorNumElements(inputDesc); i++) {
                ptr[i] = 1;
            }
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            F16 *ptr = (F16 *)((CpuMemory *)tensors["input"]->get_memory())->get_ptr();
            for (U32 i = 0; i < tensorNumElements(inputDesc); i++) {
                ptr[i] = 1;
            }
            break;
        }
#endif
        default:
            UNI_ERROR_LOG("currently not support to init this data type(%d) dlaWOdcn input data\n",
                inferencePrecision);
            break;
    }

    tensors["594"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["594"]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 13, 32, 32));
    tensors["594"]->alloc();
    tensors["nms_hm"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["nms_hm"]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 13, 32, 32));
    tensors["nms_hm"]->alloc();
    tensors["598"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["598"]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 62, 32, 32));
    tensors["598"]->alloc();
    tensors["nms_hm_kp"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["nms_hm_kp"]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 62, 32, 32));
    tensors["nms_hm_kp"]->alloc();
    tensors["wh"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["wh"]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 2, 32, 32));
    tensors["wh"]->alloc();
    tensors["kps"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["kps"]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 124, 32, 32));
    tensors["kps"]->alloc();
    tensors["reg"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["reg"]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 2, 32, 32));
    tensors["reg"]->alloc();
    tensors["kp_offset"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["kp_offset"]->resize(tensor4df(inferencePrecision, DF_NCHW, 1, 2, 32, 32));
    tensors["kp_offset"]->alloc();
    return tensors;
}

int main(int argc, char *argv[])
{
    int num = 200;
    std::string dlaWOdcnGraphPath = argv[1];
    std::vector<std::string> graphPath = {dlaWOdcnGraphPath};
    int threads = atoi(argv[2]);

    Flow flowExample;
    flowExample.init(graphPath, inferencePrecision, AFFINITY_CPU_HIGH_PERFORMANCE, threads, false);
    sleep(10);

    for (int i = 0; i < num; i++) {
        std::map<std::string, std::shared_ptr<Tensor>> data = inputOutput();
        Task task(dlaWOdcnGraphPath, data);
        flowExample.enqueue(task);
    }

    std::vector<Task> results;
    double start = ut_time_ms();
    UNI_PROFILE(results = flowExample.dequeue(true), std::string("flow_dlaWOdcn"),
        std::string("flow_dlaWOdcn"));
    double end = ut_time_ms();
    UNI_CI_LOG("avg_time:%fms/image\n", (end - start) / num);
    return 0;
}
