// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _IMAGE_MANAGER_H
#define _IMAGE_MANAGER_H
#include <vector>

class ImageManager {
public:
    ImageManager()
    {
        this->imageVecs.clear();
    }

    bool buildImageVecs(U32 slot, U32 width, U32 height, U32 depth)
    {
        std::vector<U32> str(3);
        str[0] = width;
        str[1] = height;
        str[2] = depth;
        bool needExpandStrs = true;
        if (width == 0 && height == 0 && depth == 0) {
            return false;
        } else if (width == 0 || height == 0 || depth == 0) {
            UNI_ERROR_LOG("gpu image tensor parameter is wrong.\n");
        }
        if (imageVecs.count(slot) == 0) {
            std::vector<std::vector<U32>> strs(1, str);
            imageVecs[slot] = strs;
            return needExpandStrs;
        }

        U32 vecSize = imageVecs[slot].size();
        for (U32 i = 0; i < vecSize; i++) {
            std::vector<U32> p = imageVecs[slot][i];
            if (p[0] >= str[0] && p[1] >= str[1] && p[2] >= str[2]) {
                needExpandStrs = false;
                return needExpandStrs;
            }
        }

        U32 addElementNum = str[0] * str[1] * str[2];
        std::vector<U32> updateStr(3, 0);
        U32 vecId = vecSize;
        for (U32 i = 0; i < vecSize; i++) {
            std::vector<U32> p = imageVecs[slot][i];
            U32 orgElementNum = p[0] * p[1] * p[2];
            U32 newElementNum =
                UNI_MAX(p[0], str[0]) * UNI_MAX(p[1], str[1]) * UNI_MAX(p[2], str[2]);
            if (newElementNum - orgElementNum < addElementNum) {
                addElementNum = newElementNum - orgElementNum;
                updateStr[0] = UNI_MAX(p[0], str[0]);
                updateStr[1] = UNI_MAX(p[1], str[1]);
                updateStr[2] = UNI_MAX(p[2], str[2]);
                vecId = i;
            }
        }
        if (vecId == vecSize) {
            imageVecs[slot].push_back(str);
        } else {
            imageVecs[slot][vecId] = updateStr;
        }
        return needExpandStrs;
    }

    I32 getImageVecsId(U32 slot, U32 width, U32 height, U32 depth)
    {
        I32 id = -1;
        for (U32 i = 0; i < imageVecs[slot].size(); i++) {
            std::vector<U32> p = imageVecs[slot][i];
            if (p[0] >= width && p[1] >= height && p[2] >= depth) {
                id = i;
                break;
            }
        }
        return id;
    }

    U32 getNumImageVecs()
    {
        U32 size = 0;
        for (auto &p : imageVecs) {
            size += p.second.size();
        }
        return size;
    }

    U32 getImageVecsSizeSum()
    {
        U32 sum = 0;
#ifdef _USE_FP16
        for (auto &p : imageVecs) {
            std::vector<std::vector<U32>> imageSizes = p.second;
            for (U32 i = 0; i < imageSizes.size(); i++) {
                U32 size = 4;
                for (U32 j = 0; j < imageSizes[i].size(); j++) {
                    size *= imageSizes[i][j];
                }
                sum += size * bytesOf(DT_F16);
            }
        }
#endif
        return sum;
    }

    std::map<I32, std::vector<std::vector<U32>>> getImageVecs()
    {
        return this->imageVecs;
    }

protected:
    std::map<I32, std::vector<std::vector<U32>>> imageVecs;
};
#endif
