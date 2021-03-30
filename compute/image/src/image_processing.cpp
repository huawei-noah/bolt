// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "image.h"

template <typename T>
std::shared_ptr<Tensor> get_resize_image(
    Tensor rgbTensor, TensorDesc imageDesc, ImageFormat targetImageFormat, float scaleValue)
{
    ArchInfo archInfo;
    auto arch = CPU_GENERAL;
    archInfo.arch = arch;
    DataType rgbDt = DT_F16, imageDt = DT_F16;
    DataFormat rgbDf = DF_RGB, imageDf = DF_RGB;
    U32 rgbNum = 0, rgbChannel = 0, rgbHeight = 0, rgbWidth = 0;
    U32 imageNum = 0, imageChannel = 0, imageHeight = 0, imageWidth = 0;
    TensorDesc rgbDesc = rgbTensor.get_desc();
    CHECK_STATUS(tensor4dGet(rgbDesc, &rgbDt, &rgbDf, &rgbNum, &rgbChannel, &rgbHeight, &rgbWidth));
    CHECK_REQUIREMENT(rgbDf == DF_RGB);
    CHECK_REQUIREMENT(rgbChannel == 3);
    CHECK_REQUIREMENT(rgbNum == 1);

    CHECK_STATUS(tensor4dGet(
        imageDesc, &imageDt, &imageDf, &imageNum, &imageChannel, &imageHeight, &imageWidth));
    CHECK_REQUIREMENT(imageDf == DF_NCHW);
    CHECK_REQUIREMENT(imageNum == 1);

    U32 height = rgbHeight;
    U32 width = rgbWidth;

    Tensor temp;
    std::shared_ptr<Tensor> transferSpaceTensor(new Tensor());
    transferSpaceTensor->resize(imageDesc);
    transferSpaceTensor->alloc();
    T *transferSpacePtrMov = (T *)get_ptr_from_tensor(*transferSpaceTensor, arch);

    // magic number
    float meanRGB[3] = {122.6789143406786, 116.66876761696767, 104.0069879317889};
    float meanRGBSC[3] = {0.485, 0.456, 0.406};
    float stdRGBSC[3] = {0.229, 0.224, 0.225};

    U32 transform[3];
    switch (targetImageFormat) {
        case RGB:
            transform[0] = 0;
            transform[1] = 1;
            transform[2] = 2;
            break;
        case BGR:
            transform[0] = 2;
            transform[1] = 1;
            transform[2] = 0;
            break;
        case BGR_SC_RAW:
            transform[0] = 2;
            transform[1] = 1;
            transform[2] = 0;
            break;
        case RGB_SC:
            transform[0] = 0;
            transform[1] = 1;
            transform[2] = 2;
            break;
        case RGB_RAW:
            transform[0] = 0;
            transform[1] = 1;
            transform[2] = 2;
            break;
        case RGB_SC_RAW:
            transform[0] = 0;
            transform[1] = 1;
            transform[2] = 2;
            break;
        default:
            UNI_ERROR_LOG("[ERROR] unsupported image format\n");
            return nullptr;
    }

    ResizeParamSpec p;
    p.mode = LINEAR;
    // consider the dataformat
    if (targetImageFormat == RGB_SC) {  // Specific for Birealnet18, scale short edge to 224 first
        F32 scale = 224.0 / UNI_MIN(height, width);
        if (height < width) {
            height = 224;
            width = (U32)(scale * width + 0.5);
        } else {
            height = (U32)(scale * height + 0.5);
            width = 224;
        }
        Tensor scaleTensor;
        TensorDesc scaledDesc = tensor4df(imageDt, imageDf, imageNum, imageChannel, height, width);
        scaleTensor.resize(scaledDesc);
        scaleTensor.alloc();
        resize(rgbTensor, temp, scaleTensor, p, &archInfo);

        U32 h0 = (U32)((height - 224) * 0.5);
        U32 w0 = (U32)((width - 224) * 0.5);

        T *scaled = (T *)get_ptr_from_tensor(scaleTensor, arch);
        for (U32 c : transform) {
            for (U32 h = h0; h < h0 + imageHeight; h++) {
                for (U32 w = w0; w < w0 + imageWidth; w++) {
                    T value = (scaled[c * height * width + h * width + w] / 255 - meanRGBSC[c]) /
                        stdRGBSC[c];
                    CHECK_REQUIREMENT(!UNI_ISNAN(value));
                    *transferSpacePtrMov = value;
                    transferSpacePtrMov++;
                }
            }
        }
    } else if (targetImageFormat == RGB_RAW) {
        resize(rgbTensor, temp, *transferSpaceTensor.get(), p, &archInfo);
    } else if (targetImageFormat == RGB_SC_RAW || targetImageFormat == BGR_SC_RAW) {
        F32 scale = 256.0 / UNI_MIN(height, width);
        if (height < width) {
            height = 256;
            width = (U32)(scale * (F32)width + 0.5);
        } else {
            height = (U32)(scale * (F32)height + 0.5);
            width = 256;
        }
        Tensor scaleTensor;
        TensorDesc scaledDesc = tensor4df(imageDt, imageDf, imageNum, imageChannel, height, width);
        scaleTensor.resize(scaledDesc);
        scaleTensor.alloc();
        resize(rgbTensor, temp, scaleTensor, p, &archInfo);

        U32 h0 = (U32)((height - 224) * 0.5);
        U32 w0 = (U32)((width - 224) * 0.5);

        T *scaled = (T *)get_ptr_from_tensor(scaleTensor, arch);
        for (U32 c : transform) {
            for (U32 h = h0; h < h0 + 224; h++) {
                memcpy(transferSpacePtrMov, scaled + c * height * width + h * width + w0,
                    224 * bytesOf(imageDt));
                transferSpacePtrMov += 224;
            }
        }
    } else {
        Tensor scaleTensor;
        scaleTensor.resize(imageDesc);
        scaleTensor.alloc();
        resize(rgbTensor, temp, scaleTensor, p, &archInfo);

        T *resized = (T *)get_ptr_from_tensor(scaleTensor, arch);
        for (U32 c : transform) {
            for (U32 h = 0; h < imageHeight; h++) {
                for (U32 w = 0; w < imageWidth; w++) {
                    T value = (resized[c * imageHeight * imageWidth + h * imageWidth + w] -
                                  1.0 * meanRGB[c]) *
                        scaleValue;
                    CHECK_REQUIREMENT(!UNI_ISNAN(value));
                    *transferSpacePtrMov = value;
                    transferSpacePtrMov++;
                }
            }
        }
    }
    return transferSpaceTensor;
}

// CImg load image to save in RGB format
// OpenCV load image to save in BGR format
// PIL load image to save in BGR format
// scikit-image load image to save in RGB format
// If you want to use other format, please set targetImageFormat
// numpy use OpenCV to load image

// Assume most networks require 224*224 inputs
std::shared_ptr<Tensor> load_resize_image(
    Tensor rgbTensor, TensorDesc imageDesc, ImageFormat targetImageFormat, float scaleValue)
{
    DataType imageDt = DT_F32;
    DataFormat imageDf;
    U32 imageNum, imageChannel, imageHeight, imageWidth;

    CHECK_STATUS(tensor4dGet(
        imageDesc, &imageDt, &imageDf, &imageNum, &imageChannel, &imageHeight, &imageWidth));

    switch (imageDt) {
#ifdef __aarch64__
        case DT_F16: {
            return get_resize_image<F16>(rgbTensor, imageDesc, targetImageFormat, scaleValue);
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            return get_resize_image<F32>(rgbTensor, imageDesc, targetImageFormat, scaleValue);
        }
#endif
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
            return nullptr;
        }
    }
}

template <typename T>
std::shared_ptr<U8> gen_fake_image(TensorDesc inputDesc)
{
    DataType dt;
    DataFormat df;
    U32 in = 0, ic = 0, ih = 0, iw = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &dt, &df, &in, &ic, &ih, &iw));
    CHECK_REQUIREMENT(df == DF_NCHW);
    CHECK_REQUIREMENT(in == 1);

    U32 totalBytes = tensorNumBytes(inputDesc);

    // upon on the data type, to malloc the corresponding space
    T *transferSpacePtr = (T *)operator new(totalBytes);
    T *transferSpacePtrMov = transferSpacePtr;

    // consider the dataformat
    for (U32 c = 0; c < ic; c++) {
        for (U32 h = 0; h < ih; h++) {
            for (U32 w = 0; w < iw; w++) {
                *transferSpacePtrMov = 1;
                transferSpacePtrMov++;
            }
        }
    }

    std::shared_ptr<U8> val((U8 *)transferSpacePtr);
    return val;
}
