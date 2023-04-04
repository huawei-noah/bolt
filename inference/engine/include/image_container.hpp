// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _IMAGE_CONTAINER_H
#define _IMAGE_CONTAINER_H

#include "tensor.hpp"
#include "image_manager.hpp"

class ImageContainer : public ImageManager {
public:
    ImageContainer() : ImageManager()
    {
        this->images.clear();
    }

    void add(U32 slot, U32 width, U32 height, U32 depth)
    {
        ImageManager::buildImageVecs(slot, width, height, depth);
    }

    void alloc()
    {
        for (auto &p : imageVecs) {
            auto slot = p.first;
            auto imageSizes = p.second;
            std::vector<std::shared_ptr<Tensor>> tensors;
            for (auto &str : imageSizes) {
                std::shared_ptr<Tensor> tensor = std::shared_ptr<Tensor>(new Tensor(OCLMemImg));
                auto mem = (OclMemoryImg *)tensor->get_memory();
                mem->alloc(str[0], str[1], str[2]);
                tensors.push_back(tensor);
            }
            images[slot] = tensors;
        }
    }

    Tensor get(U32 slot, U32 width, U32 height, U32 depth)
    {
        I32 vecId = ImageManager::getImageVecsId(slot, width, height, depth);
        if (vecId < 0 || vecId >= (I32)images[slot].size()) {
            UNI_ERROR_LOG("gpu image buffer reuse wrong.\n");
        }
        return *(images[slot][vecId].get());
    }

private:
    ImageManager imageManager;
    std::map<I32, std::vector<std::shared_ptr<Tensor>>> images;
};
#endif
