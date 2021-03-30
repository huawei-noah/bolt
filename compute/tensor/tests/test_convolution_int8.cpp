// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>
#include "tensor_computing.h"
#include "ut_util.h"

#ifdef _USE_INT8
int int8ConvolutionTest(int argc, char *argv[], DataType dt, DataType filterDataType)
{
    CHECK_REQUIREMENT(argc == 16);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    // weight
    U32 fn = atoi(argv[5]);
    U32 fc = atoi(argv[6]);
    U32 fh = atoi(argv[7]);
    U32 fw = atoi(argv[8]);
    U32 group = atoi(argv[9]);
    // stride & padding
    U32 stride = atoi(argv[10]);
    U32 padding = atoi(argv[11]);
    // output
    U32 on = atoi(argv[12]);
    U32 oc = atoi(argv[13]);
    U32 oh = atoi(argv[14]);
    U32 ow = atoi(argv[15]);
    CHECK_REQUIREMENT(in == 1 && on == 1);

    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;
    ActivationParamSpec activationDesc;
    activationDesc.mode = ACTIVATION_RELU;
    activationDesc.value[0] = 0;

    TensorDesc inputDesc, filterDesc, outputDesc, biasDesc;
    ConvolutionParamSpec p = createConvolutionParamSpec(group, 1, fh, fw, 1, stride, stride, 0, 0,
        padding, padding, padding, padding, 1, 1, 1, fn, Convolution_Depthwise_Pointwise);

    if (ic % 8 != 0) {
        printf("[WARN] can not quantize the first layer\n");
        return 0;
    } else {
        DataType qdt = DT_I8;
        TensorDesc inputDesc_ref = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
        filterDesc = tensor4df(dt, DF_NCHW, oc, ic, fh, fw);
        biasDesc = tensor1d(dt, oc);

        // setup input, filter, bias
        U8 *input_ref = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
        U8 *filter = ut_input_v(fn * fc * fh * fw, dt, UT_INIT_RANDOM);
        U8 *bias = ut_input_v(oc, dt, UT_INIT_RANDOM);

        INT8 *input = (INT8 *)ut_input_v(in * ic * ih * iw, DT_I8, UT_INIT_ZERO);
        F16 scale_i = -1;
        quantize_tensor(inputDesc_ref, input_ref, &inputDesc, input, &scale_i);

        Tensor inputTensor;
        inputTensor.resize(inputDesc);
        inputTensor.alloc();
        memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inputDesc));

        Tensor inputTensorRef;
        inputTensorRef.resize(inputDesc_ref);
        inputTensorRef.alloc();
        memcpy(
            get_ptr_from_tensor(inputTensorRef, UT_ARCH), input_ref, tensorNumBytes(inputDesc_ref));

        Tensor filterTensor;
        filterTensor.resize(filterDesc);
        filterTensor.alloc();
        memcpy(get_ptr_from_tensor(filterTensor, UT_ARCH), filter, tensorNumBytes(filterDesc));

        Tensor filterTensorRef;
        filterTensorRef.resize(filterDesc);
        filterTensorRef.alloc();
        memcpy(get_ptr_from_tensor(filterTensorRef, UT_ARCH), filter, tensorNumBytes(filterDesc));

        Tensor biasTensor;
        biasTensor.resize(biasDesc);
        biasTensor.alloc();
        memcpy(get_ptr_from_tensor(biasTensor, UT_ARCH), bias, tensorNumBytes(biasDesc));

        Tensor outputTensor, outputTensorRef;

        // setup output, bias
        CHECK_STATUS(convolution_infer_output_size(
            &inputTensor, filterTensor, p, &outputTensor, qdt, &archInfo));
        outputTensor.alloc();

        outputDesc = outputTensor.get_desc();
        TensorDesc outputDesc_ref = outputTensor.get_desc();
        outputDesc_ref.dt = dt;
        outputTensorRef.resize(outputDesc_ref);
        outputTensorRef.alloc();

        // setup alg
        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
        CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor, filterTensor, outputTensor, p,
            policy, &alg, qdt, activationDesc, &archInfo));

        F16 *scales;

        // setup filter trans
        U32 ftBytes;
        Tensor ftmTensor, tmpTensor;

        switch (alg) {
            case CONVOLUTION_ALGORITHM_WINOGRAD: {
                CHECK_STATUS(
                    convolution_transform_filter_bytes(filterTensor, p, alg, &ftBytes, &archInfo));

                Tensor tFilter;
                tFilter.resize(tensor1d(DT_U8, ftBytes));
                tFilter.alloc();

                filterDesc.dt = filterDataType;  // To label as int8
                filterTensor.resize(filterDesc);
                CHECK_STATUS(convolution_transform_filter(
                    filterTensor, p, alg, tmpTensor, &tFilter, &archInfo));

                TensorDesc ftmDesc = tFilter.get_desc();
                ftmDesc.dt = DT_I8;
                ftmTensor.resize(ftmDesc);
                ftmTensor.alloc();

                scales = (F16 *)ut_input_v(
                    38, DT_F16, UT_INIT_ZERO);  // 1 for input, 1 for output and 36 for filter
                CHECK_STATUS(
                    quantize_tensor(tFilter.get_desc(), get_ptr_from_tensor(tFilter, UT_ARCH),
                        &ftmDesc, get_ptr_from_tensor(ftmTensor, UT_ARCH), scales + 2));
                break;
            }
            default: {
                Tensor qFilter;
                TensorDesc qDesc = filterDesc;
                qDesc.dt = DT_I8;
                qFilter.resize(qDesc);
                qFilter.alloc();
                scales = (F16 *)ut_input_v(3, DT_F16, UT_INIT_ZERO);
                CHECK_STATUS(quantize_tensor(
                    filterDesc, filter, &qDesc, get_ptr_from_tensor(qFilter, UT_ARCH), scales + 2));

                CHECK_STATUS(
                    convolution_transform_filter_bytes(qFilter, p, alg, &ftBytes, &archInfo));

                ftmTensor.resize(tensor1d(DT_U8, ftBytes));
                ftmTensor.alloc();
                // trans filter
                CHECK_STATUS(
                    convolution_transform_filter(qFilter, p, alg, tmpTensor, &ftmTensor, &archInfo));
                break;
            }
        }

        scales[0] = scale_i;

        // setup tmp
        U32 tmpBytes;
        CHECK_STATUS(convolution_infer_forward_tmp_bytes(
            inputTensor, ftmTensor, outputTensor, p, alg, &tmpBytes, &archInfo));
        tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
        tmpTensor.alloc();

        std::vector<Tensor> inputTensors(1, inputTensor);
        std::vector<Tensor> inputTensorsRef(1, inputTensorRef);
        if (UT_CHECK) {
            CHECK_STATUS(convolution(inputTensors, ftmTensor, p, alg, scales, biasTensor, tmpTensor,
                outputTensor, activationDesc, &archInfo));

            // naive implement
            CHECK_STATUS(convolution(inputTensorsRef, filterTensorRef, p, alg, nullptr, biasTensor,
                tmpTensor, outputTensorRef, activationDesc, &archInfo_org));

            U32 output_size = outputTensor.length();
            U8 *out_d = ut_input_v(output_size, dt, UT_INIT_ZERO);
            INT8 *output = (INT8 *)get_ptr_from_tensor(outputTensor, UT_ARCH);
            for (U32 i = 0; i < output_size; i++) {
                switch (dt) {
#ifdef _USE_FP32
                    case DT_F32:
                        ((F32 *)out_d)[i] = output[i] / scales[1];
                        break;
#endif
#ifdef _USE_FP16
                    case DT_F16:
                        ((F16 *)out_d)[i] = output[i] / scales[1];
                        break;
#endif
                    default:
                        break;
                }
            }
            ut_check_v(out_d, get_ptr_from_tensor(outputTensorRef, UT_ARCH), output_size, dt, 8,
                __FILE__, __LINE__);
            free(out_d);
        }

        // benchmark
        double time_start = ut_time_ms();
        for (int iter = 0; iter < UT_LOOPS; iter++) {
            CHECK_STATUS(convolution(inputTensors, ftmTensor, p, alg, scales, biasTensor, tmpTensor,
                outputTensor, activationDesc, &archInfo));
        }
        double time_end = ut_time_ms();
        double time = (time_end - time_start) / UT_LOOPS;

        // log performance data
        char buffer[150];
        char params[120];
        DataFormat df;
        CHECK_STATUS(tensor4dGet(outputDesc, &dt, &df, &on, &oc, &oh, &ow));
        sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u %u)=(%u %u %u %u)", in, ic, ih, iw, fn,
            fc, fh, fw, group, stride, padding, on, oc, oh, ow);
        sprintf(buffer, "%20s, %80s", "Convolution", params);
        double ops = (1.0 * on * oc * oh * ow) * (2.0 * ic * fh * fw / group + 1);
        ut_log(DT_I8, buffer, ops, time);

        free(input);
        free(filter);
        free(bias);
        free(input_ref);
        free(scales);
    }
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_INT8
    int8ConvolutionTest(argc, argv, DT_F16, DT_F16_8Q);
#endif
    return 0;
}
