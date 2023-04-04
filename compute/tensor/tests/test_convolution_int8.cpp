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

    ActivationParamSpec activationDesc;
    activationDesc.mode = ACTIVATION_RELU;
    activationDesc.value[0] = 0;

    TensorDesc inputDesc, filterDesc, outputDesc, biasDesc;
    ConvolutionParamSpec p = createConvolutionParamSpec(group, 1, fh, fw, 1, stride, stride, 0, 0,
        padding, padding, padding, padding, 1, 1, 1, fn, CONVOLUTION_DEPTHWISE_POINTWISE);

    if (ic % 8 != 0) {
        UNI_WARNING_LOG("can not quantize the first layer\n");
        return 0;
    } else {
#ifdef _USE_X86
        DataType qdt = DT_U8_Q;
#else
        DataType qdt = DT_I8;
#endif
        TensorDesc inputDesc_ref = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
        inputDesc = tensor4df(qdt, DF_NCHWC8, in, ic, ih, iw);
        filterDesc = tensor4df(dt, DF_NCHW, oc, ic, fh, fw);
        biasDesc = tensor1d(dt, oc);

        // setup input, filter, bias
        Tensor inputTensorRef = Tensor::alloc_sized<CPUMem>(inputDesc_ref);
        Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inputDesc);
        ut_init_v((U8 *)get_ptr_from_tensor(inputTensorRef, CPU_GENERAL), inputTensorRef.length(),
            dt, UT_INIT_RANDOM);

        F32 scale_i = -1;
        quantize(inputTensorRef, &inputTensor, &scale_i, &UT_CPU_ARCHINFO);

        Tensor filterTensor = Tensor::alloc_sized<CPUMem>(filterDesc);
        ut_init_v((U8 *)get_ptr_from_tensor(filterTensor, CPU_GENERAL), filterTensor.length(), dt,
            UT_INIT_RANDOM);

        Tensor filterTensorRef = Tensor::alloc_sized<CPUMem>(filterDesc);
        filterTensorRef.copy_from(&filterTensor);

        Tensor biasTensor = Tensor::alloc_sized<CPUMem>(biasDesc);
        ut_init_v((U8 *)get_ptr_from_tensor(biasTensor, CPU_GENERAL), biasTensor.length(), dt,
            UT_INIT_RANDOM);

        Tensor outputTensor, outputTensorRef;

        // setup output, bias
        CHECK_STATUS(convolution_infer_output_size(
            &inputTensor, filterTensor, p, &outputTensor, qdt, &UT_CPU_ARCHINFO));
        outputTensor.alloc();

        outputDesc = outputTensor.get_desc();
        TensorDesc outputDesc_ref = outputTensor.get_desc();
        outputDesc_ref.dt = dt;
        outputDesc_ref.df = DF_NCHWC8;
#ifdef _USE_AVX512_VNNI
        outputDesc_ref.df = DF_NCHW;
#endif
        outputTensorRef.resize(outputDesc_ref);
        outputTensorRef.alloc();

        // setup alg
        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
        CHECK_STATUS(convolution_infer_forward_algorithm(inputTensor, filterTensor, outputTensor, p,
            policy, &alg, qdt, activationDesc, &UT_CPU_ARCHINFO));

        std::vector<F32> scales;

        // setup filter trans
        U32 ftBytes;
        Tensor ftmTensor, tmpTensor;
        switch (alg) {
            case CONVOLUTION_ALGORITHM_WINOGRAD: {
                CHECK_STATUS(convolution_transform_filter_bytes(
                    filterTensor, p, alg, &ftBytes, &UT_CPU_ARCHINFO));

                Tensor tFilter = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, ftBytes));

                filterDesc.dt = filterDataType;  // To label as int8
                filterTensor.resize(filterDesc);
                CHECK_STATUS(convolution_transform_filter(
                    filterTensor, p, alg, tmpTensor, &tFilter, &UT_CPU_ARCHINFO));

                U32 ftmBytes = ftBytes / bytesOf(filterDataType);
                ftmTensor = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, ftmBytes));
                scales = std::vector<F32>(38);
                CHECK_STATUS(quantize(tFilter, &ftmTensor, scales.data() + 2, &UT_CPU_ARCHINFO));
                break;
            }
            default: {
                TensorDesc qDesc = filterDesc;
                qDesc.dt = DT_I8;
                Tensor qFilter = Tensor::alloc_sized<CPUMem>(qDesc);
                scales = std::vector<F32>(3);
                CHECK_STATUS(quantize(filterTensor, &qFilter, scales.data() + 2, &UT_CPU_ARCHINFO));

                CHECK_STATUS(
                    convolution_transform_filter_bytes(qFilter, p, alg, &ftBytes, &UT_CPU_ARCHINFO));

                ftmTensor = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, ftBytes));
                // trans filter
                CHECK_STATUS(convolution_transform_filter(
                    qFilter, p, alg, tmpTensor, &ftmTensor, &UT_CPU_ARCHINFO));
                break;
            }
        }

        scales[0] = scale_i;

        // setup tmp
        U32 tmpBytes;
        CHECK_STATUS(convolution_infer_forward_tmp_bytes(
            inputTensor, ftmTensor, outputTensor, p, alg, &tmpBytes, &UT_CPU_ARCHINFO));
        tmpTensor = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, tmpBytes));

        std::vector<Tensor> inputTensors(1, inputTensor);
        std::vector<Tensor> inputTensorsRef(1, inputTensorRef);
        std::vector<Tensor> tmpTensors(1, tmpTensor);
        if (UT_CHECK) {
            CHECK_STATUS(convolution(inputTensors, ftmTensor, p, alg, scales.data(), biasTensor,
                tmpTensors, outputTensor, activationDesc, &UT_CPU_ARCHINFO));

            // naive implement
            CHECK_STATUS(convolution(inputTensorsRef, filterTensorRef, p, alg, nullptr, biasTensor,
                tmpTensors, outputTensorRef, activationDesc, &UT_SERIAL_ARCHINFO));

            U32 output_size = outputTensor.length();
            U8 *out_d = ut_input_v(output_size, dt, UT_INIT_ZERO);
            INT8 *output = (INT8 *)get_ptr_from_tensor(outputTensor, CPU_GENERAL);
#ifdef _USE_AVX512_VNNI
            TensorDesc destDesc = outputDesc;
            destDesc.df = DF_NCHW;
            U8 *out_c = ut_input_v(output_size, dt, UT_INIT_ZERO);
            transformToNCHW(outputDesc, (void *)output, destDesc, (void *)out_c);
            output = (INT8 *)out_c;
#endif
            for (U32 i = 0; i < output_size; i++) {
                switch (dt) {
#ifdef _USE_FP32
                    case DT_F32:
#ifdef _USE_X86
                        ((F32 *)out_d)[i] = (((UINT8 *)output)[i] - 128) / scales[1];
#else
                        ((F32 *)out_d)[i] = output[i] / scales[1];
#endif
                        break;
#endif
#ifdef _USE_FP16
                    case DT_F16:
#ifdef _USE_X86
                        ((F16 *)out_d)[i] = (((UINT8 *)output)[i] - 128) / scales[1];
#else
                        ((F16 *)out_d)[i] = output[i] / scales[1];
#endif
                        break;
#endif
                    default:
                        break;
                }
            }
            ut_check_v(out_d, get_ptr_from_tensor(outputTensorRef, CPU_GENERAL), output_size, dt, 8);
            free(out_d);
#ifdef _USE_AVX512_VNNI
            free(out_c);
#endif
        }

        // benchmark
        double time_start = ut_time_ms();
        for (int iter = 0; iter < UT_LOOPS; iter++) {
            CHECK_STATUS(convolution(inputTensors, ftmTensor, p, alg, scales.data(), biasTensor,
                tmpTensors, outputTensor, activationDesc, &UT_CPU_ARCHINFO));
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
    }
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_INT8
#ifdef _USE_FP16
    int8ConvolutionTest(argc, argv, DT_F16, DT_F16_8Q);
#elif defined(_USE_X86) || not defined(__aarch64__)
    int8ConvolutionTest(argc, argv, DT_F32, DT_F32_8Q);
#endif
#endif
    return 0;
}
