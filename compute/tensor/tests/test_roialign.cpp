// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_computing.h"
#include "ut_util.h"

int roialignTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 16);
    // in0 feature map
    U32 in0 = atoi(argv[1]);
    U32 ic0 = atoi(argv[2]);
    U32 ih0 = atoi(argv[3]);
    U32 iw0 = atoi(argv[4]);
    // in1 rois
    U32 ih1 = atoi(argv[5]);
    U32 iw1 = atoi(argv[6]);
    // in2 batch_indices
    U32 ilens2 = atoi(argv[7]);
    // output
    U32 on0 = atoi(argv[8]);
    U32 oc0 = atoi(argv[9]);
    U32 oh0 = atoi(argv[10]);
    U32 ow0 = atoi(argv[11]);
    // p
    U32 output_h = atoi(argv[12]);
    U32 output_w = atoi(argv[13]);
    U32 sampling_ratio = atoi(argv[14]);
    F32 spatial_scale = (F32)atof(argv[15]);
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    RoiAlignParamSpec p;
    p.output_h = output_h;
    p.output_w = output_w;
    p.sampling_ratio = sampling_ratio;
    p.spatial_scale = spatial_scale;

    std::vector<Tensor> inputTensors(3);
    std::vector<Tensor *> inputTensorPtrs(3);
    TensorDesc inputDesc_feat = tensor4d(dt, in0, ic0, ih0, iw0);
    TensorDesc inputDesc_rois = tensor2d(dt, ih1, iw1);
    TensorDesc inputDesc_batch = tensor1d(dt, ilens2);
    Tensor inputTensor_feat = Tensor::alloc_sized<CPUMem>(inputDesc_feat);
    Tensor inputTensor_rois = Tensor::alloc_sized<CPUMem>(inputDesc_rois);
    Tensor inputTensor_batch = Tensor::alloc_sized<CPUMem>(inputDesc_batch);
    U32 input_len_feat = tensorNumElements(inputDesc_feat);
    U32 input_len_rois = tensorNumElements(inputDesc_rois);
    U32 input_len_batch = tensorNumElements(inputDesc_batch);
    U8 *input_feat = ut_input_v(input_len_feat, dt, UT_INIT_RANDOM);
    U8 *input_rois = ut_input_v(input_len_rois, dt, UT_INIT_RANDOM);
    U8 *input_batch = ut_input_v(input_len_batch, dt, UT_INIT_ZERO);
    memcpy(
        get_ptr_from_tensor(inputTensor_feat, UT_ARCH), input_feat, tensorNumBytes(inputDesc_feat));
    memcpy(
        get_ptr_from_tensor(inputTensor_rois, UT_ARCH), input_rois, tensorNumBytes(inputDesc_rois));
    memcpy(get_ptr_from_tensor(inputTensor_batch, UT_ARCH), input_batch,
        tensorNumBytes(inputDesc_batch));
    inputTensors[0] = inputTensor_feat;
    inputTensors[1] = inputTensor_rois;
    inputTensors[2] = inputTensor_batch;
    inputTensorPtrs[0] = &inputTensors[0];
    inputTensorPtrs[1] = &inputTensors[1];
    inputTensorPtrs[2] = &inputTensors[2];

    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(roialign_infer_output_size(inputTensorPtrs, p, &outputTensor, &archInfo));
    outputTensor.alloc();
    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();
    U32 output_len = outputTensor.length();
    CHECK_REQUIREMENT(ih1 == on0 && ic0 == oc0 && output_h == oh0 && output_w == ow0);
    CHECK_REQUIREMENT(input_len_feat == in0 * ic0 * ih0 * iw0 && input_len_rois == ih1 * iw1 &&
        input_len_batch == ilens2 && output_len == on0 * oc0 * oh0 * ow0);

    if (UT_CHECK) {
        CHECK_STATUS(roialign(inputTensors, p, outputTensor, &archInfo));
        CHECK_STATUS(roialign(inputTensors, p, outputTensorRef, &archInfo_org));
        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), output_len, dt, 0.05, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(roialign(inputTensors, p, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) * (%u %u) * (%u) * (%u %u) = (%u %u %u %u)", in0, ic0, ih0, iw0,
        ih1, iw1, ilens2, output_h, output_w, on0, oc0, oh0, ow0);
    sprintf(buffer, "%20s, %80s", "Roialign", params);
    double ops = 1.0 * output_len;
    ut_log(dt, buffer, ops, time);

    free(input_feat);
    free(input_rois);
    free(input_batch);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    roialignTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    roialignTest(argc, argv, DT_F32);
#endif
    return 0;
}
