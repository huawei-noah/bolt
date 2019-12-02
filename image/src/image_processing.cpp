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
#include <memory>
#include <assert.h>
#include "CImg.h"
#include "tensor_desc.h"
#include "type.h"
#include "error.h"

EE print_image(std::string imagePath) {
    cimg_library::CImg<unsigned char> img(imagePath.c_str());
    int width = img.width();
    int height = img.height();
    std::cout << width << "x" << height << std::endl;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            std::cout << "(" << r << "," << c << ")="
            << " R" << (float)img(c, r, 0, 0) << " " << (void*)(&img(c, r, 0, 0)) << "/"
            << " G" << (float)img(c, r, 0, 1) << " " << (void*)(&img(c, r, 0, 1)) << "/"
            << " B" << (float)img(c, r, 0, 2) << " " << (void*)(&img(c, r, 0, 2))
            << std::endl;
        } 
    }
    return SUCCESS;
}

// CImg load image to save in RGB format
// OpenCV load image to save in BGR format
// PIL load image to save in BGR format
// scikit-image load image to save in RGB format
// If you want to use other format, please set targetImageType
// numpy use OpenCV to load image
std::shared_ptr<U8> load_resize_image(std::string imagePath, TensorDesc imageDesc, ImageType targetImageType, float scaleValue) {
    DataType imageDt;
    DataFormat imageDf;
    U32 imageNum, imageChannel, imageHeight, imageWidth;
    EE ret = tensor4dGet(imageDesc, &imageDt, &imageDf, &imageNum, &imageChannel, &imageHeight, &imageWidth);
    assert(ret == SUCCESS);
    assert(imageDt == DT_F16);
    assert(imageDf == DF_NCHW);
    assert(imageNum == 1);

    // load
    cimg_library::CImg<unsigned char> img(imagePath.c_str());
    U32 channel = imageChannel;
    U32 depth   = img.depth();
    U32 height  = imageHeight;
    U32 width   = imageWidth;
    assert(depth == 1);

    if (targetImageType == RGB_SC) {
        height = img.height();
        width = img.width();
        U32 smaller = (height < width) ? height : width;
        height = height * imageHeight / smaller;
        width = width * imageHeight / smaller;
        img.resize(width, height, 1, channel);
    } else {
        // resize
        img.resize(width, height, 1, channel);
        //img.save("/data/local/test/0.jpg");
    }
    
    U32 totalBytes = tensorNumBytes(imageDesc);
    F16 *transferSpacePtr = (F16 *)operator new(totalBytes);
    F16 *transferSpacePtrMov = transferSpacePtr;

    // magic number
    float meanRGB[3] = {122.6789143406786, 116.66876761696767, 104.0069879317889};
    float meanRGBSC[3] = {0.485, 0.456, 0.406};
    float stdRGBSC[3] = {0.229, 0.224, 0.225};
    int transform[3];
    switch (targetImageType) {
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
        case RGB_SC:
            transform[0] = 0;
            transform[1] = 1;
            transform[2] = 2;
            break;
        default:
            std::cerr << "[ERROR] unsupported image type" << std::endl;
            exit(1);
            return nullptr;
    }
    
    // consider the dataformat
    if (targetImageType == RGB_SC) {
        U32 hBase = 0;
        U32 wBase = 0;
        if (height > width) {
            hBase = (height - imageHeight) / 2;
        } else {
            wBase = (width - imageWidth) / 2;
        }
        for (U32 c : transform) {
            for (U32 h = hBase; h < hBase + imageHeight; h++) {
                for (U32 w = wBase; w < wBase + imageWidth; w++) {
                    F16 value = ((F16)(img(w, h, 0, c)) / 255 - meanRGBSC[c]) / stdRGBSC[c];
                    assert(!std::isnan(value));
                    *transferSpacePtrMov = value;
                    transferSpacePtrMov++;
                }
            }
        }
    } else {
        for (U32 c : transform) {
            for (U32 h = 0; h < imageHeight; h++) {
                for (U32 w = 0; w < imageWidth; w++) {
                    F16 value = (F16)((img(w, h, 0, c) - 1.0*meanRGB[c]) * scaleValue);
                    assert(!std::isnan(value));
                    *transferSpacePtrMov = value;
                    transferSpacePtrMov++;
                }
            }
        }
    }

    std::shared_ptr<U8> val((U8*)transferSpacePtr);    
    return val;
}

std::shared_ptr<U8> load_fake_image(TensorDesc inputDesc) {
    DataType dt;
    DataFormat df;
    U32 in, ic, ih, iw;
    EE ret = tensor4dGet(inputDesc, &dt, &df, &in, &ic, &ih, &iw);
    assert(ret == SUCCESS);
    assert(df == DF_NCHW);
    assert(dt == DT_F16);
    assert(in == 1);

    U32 totalBytes = tensorNumBytes(inputDesc);

    // upon on the data type, to malloc the corresponding space
    F16 *transferSpacePtr = (F16 *)operator new(totalBytes);
    F16 *transferSpacePtrMov = transferSpacePtr;

    // consider the dataformat
    for (U32 c = 0; c < ic; c++) {
        for (U32 h = 0; h < ih; h++) {
            for (U32 w = 0; w < iw; w++) {
                *transferSpacePtrMov = 1;
                transferSpacePtrMov++;
            }
        }
    }

    std::shared_ptr<U8> val((U8*)transferSpacePtr);
    return val;
}

