// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_DATA_LOADER
#define _H_DATA_LOADER

#include <string>
#include <vector>
#include <map>
#include "file.h"
#include "tensor.hpp"

static Tensor read_file_data(FILE *file, TensorDesc desc)
{
    Tensor tensor = Tensor::alloc_sized<CPUMem>(desc);
    U32 size = tensor.length();
    DataType dataType = desc.dt;
    auto ptr = ((CpuMemory *)(tensor.get_memory()))->get_ptr();
    F32 value;
    switch (dataType) {
        case DT_F32: {
            F32 *p = (F32 *)ptr;
            for (U32 i = 0; i < size; i++) {
                UNI_FSCANF(file, "%f", &value);
                p[i] = value;
            }
            break;
        }
        case DT_F16: {
            unsigned short *p = (unsigned short *)ptr;
            for (U32 i = 0; i < size; i++) {
                UNI_FSCANF(file, "%f", &value);
                p[i] = float32ToFloat16(value);
            }
            break;
        }
        case DT_U32: {
            U32 *p = (U32 *)ptr;
            for (U32 i = 0; i < size; i++) {
                UNI_FSCANF(file, "%f", &value);
                p[i] = value;
            }
            break;
        }
        case DT_I32: {
            I32 *p = (I32 *)ptr;
            for (U32 i = 0; i < size; i++) {
                UNI_FSCANF(file, "%f", &value);
                p[i] = value;
            }
            break;
        }
        case DT_I8: {
            INT8 *p = (INT8 *)ptr;
            for (U32 i = 0; i < size; i++) {
                UNI_FSCANF(file, "%f", &value);
                p[i] = value;
            }
            break;
        }
        case DT_U8: {
            UINT8 *p = (UINT8 *)ptr;
            for (U32 i = 0; i < size; i++) {
                UNI_FSCANF(file, "%f", &value);
                p[i] = value;
            }
            break;
        }
        default:
            UNI_ERROR_LOG("not support to read %s type data.\n", DataTypeName()[dataType]);
            break;
    }
    return tensor;
}

std::vector<Tensor> load_fake_data(std::vector<TensorDesc> descs)
{
    std::vector<Tensor> result;
    for (U32 index = 0; index < descs.size(); index++) {
        Tensor tensor = Tensor::alloc_sized<CPUMem>(descs[index]);
        U8 *ptr = (U8 *)((CpuMemory *)(tensor.get_memory()))->get_ptr();
        UNI_INIT(tensorNumElements(descs[index]), descs[index].dt, 1, ptr);
        result.push_back(tensor);
    }
    return result;
}

std::vector<Tensor> load_txt(std::string path, std::vector<TensorDesc> descs)
{
    UNI_DEBUG_LOG("read data from %s.\n", path.c_str());
    std::vector<Tensor> result;
    FILE *file = fopen(path.c_str(), "r");
    if (file == NULL) {
        UNI_ERROR_LOG("can not read %s.\n", path.c_str());
    }
    for (U32 index = 0; index < descs.size(); index++) {
        result.push_back(read_file_data(file, descs[index]));
    }
    fclose(file);
    return result;
}

std::vector<Tensor> load_seq(std::string path, std::vector<TensorDesc> descs)
{
    UNI_DEBUG_LOG("read data from %s.\n", path.c_str());
    std::vector<Tensor> result;
    FILE *file = fopen(path.c_str(), "r");
    if (file == NULL) {
        UNI_ERROR_LOG("can not read %s.\n", path.c_str());
    }
    for (U32 index = 0; index < descs.size(); index++) {
        U32 sequenceLen = 0;
        UNI_FSCANF(file, "%u", &sequenceLen);
        TensorDesc sequenceDesc = descs[index];
        sequenceDesc.dims[0] = sequenceLen;
        for (U32 j = 1; j < sequenceDesc.nDims; j++) {
            sequenceDesc.dims[j] = 1;
        }
        result.push_back(read_file_data(file, sequenceDesc));
    }
    fclose(file);
    return result;
}

std::vector<Tensor> load_bin(
    std::string path, std::vector<DataType> types, std::vector<TensorDesc> descs)
{
    UNI_DEBUG_LOG("read data from %s.\n", path.c_str());
    std::vector<Tensor> result;
    FILE *f = fopen(path.c_str(), "rb");
    if (nullptr == f) {
        result = load_fake_data(descs);
    } else {
        for (U32 index = 0; index < descs.size(); index++) {
            TensorDesc desc = descs[index];
            desc.dt = types[index];
            Tensor tensor = Tensor::alloc_sized<CPUMem>(desc);
            U32 len = tensor.length();
            auto ptr = ((CpuMemory *)(tensor.get_memory()))->get_ptr();
            CHECK_REQUIREMENT(fread(ptr, bytesOf(types[index]), len, f) == len);
            if (types[index] != descs[index].dt) {
                Tensor transform_tensor = Tensor::alloc_sized<CPUMem>(descs[index]);
                if (types[index] == DT_F32) {
                    transformFromFloat(descs[index].dt, (const float *)ptr,
                        ((CpuMemory *)(transform_tensor.get_memory()))->get_ptr(), len);
                } else {
                    UNI_ERROR_LOG(
                        "not support to read+transform %s data.\n", DataTypeName()[types[index]]);
                }
                result.push_back(transform_tensor);
            } else {
                result.push_back(tensor);
            }
        }
        fclose(f);
    }
    return result;
}

#ifdef _BUILD_EXAMPLE
#include <jpeglib.h>
#include <jerror.h>
#include "image_processing.hpp"

static std::vector<Tensor> load_jpeg(
    std::string path, std::vector<TensorDesc> imageDesc, ImageFormat ImageFormat, F32 scaleValue)
{
    FILE *file = fopen(path.c_str(), "rb");
    CHECK_REQUIREMENT(NULL != file);

    struct jpeg_decompress_struct info;
    struct jpeg_error_mgr err;

    info.err = jpeg_std_error(&err);
    jpeg_create_decompress(&info);

    jpeg_stdio_src(&info, file);
    jpeg_read_header(&info, TRUE);

    jpeg_start_decompress(&info);

    U32 width = info.output_width;
    U32 height = info.output_height;
    U32 numChannels = info.output_components;
    U32 dataSize = numChannels * height * width;

    UNI_DEBUG_LOG(
        "%s: channels %u , out color space %d\n", path.c_str(), numChannels, info.out_color_space);
    CHECK_REQUIREMENT(2 == info.out_color_space);  // Support RGB for now

    U8 *data = (U8 *)UNI_MALLOC(dataSize);
    JSAMPROW row_pointer[1];
    while (info.output_scanline < info.output_height) {
        row_pointer[0] = data + info.output_scanline * width * numChannels;
        int ret = jpeg_read_scanlines(&info, row_pointer, 1);
        CHECK_REQUIREMENT(ret == 1);
    }

    jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    fclose(file);

    TensorDesc rgbDesc = tensor4df(DT_U8, DF_RGB, 1, 3, height, width);
    Tensor rgbTensor = Tensor::alloc_sized<CPUMem>(rgbDesc);
    U8 *rgb = (U8 *)((CpuMemory *)(rgbTensor.get_memory()))->get_ptr();
    U8 *r = rgb;
    U8 *g = r + height * width;
    U8 *b = g + height * width;

    U8 *dataMov = data;
    for (U32 i = 0; i < height * width; i++) {
        r[i] = dataMov[0];
        g[i] = dataMov[1];
        b[i] = dataMov[2];
        dataMov += numChannels;
    }
    UNI_FREE(data);

    std::shared_ptr<Tensor> imageTensor =
        load_resize_image(rgbTensor, imageDesc[0], ImageFormat, scaleValue);
    std::vector<Tensor> result;
    imageTensor->resize(imageDesc[0]);
    result.push_back(*imageTensor.get());
    return result;
}

std::vector<std::string> load_image_with_scale(std::string directory,
    std::vector<TensorDesc> descs,
    std::vector<std::vector<Tensor>> *datas,
    ImageFormat ImageFormat,
    F32 scaleValue)
{
    std::vector<std::string> paths;
    if (datas == NULL) {
        return paths;
    }
    std::vector<Tensor> data;
    if (directory == "") {
        data = load_fake_data(descs);
        datas->push_back(data);
        paths.push_back("fake data");
        return paths;
    }
    std::vector<std::string> names = search_files(directory);
    for (U32 i = 0; i < names.size(); i++) {
        std::string path = directory + "/" + names[i];
        if (endswith(path, ".jpg") || endswith(path, ".jpeg")) {
            data = load_jpeg(path, descs, ImageFormat, scaleValue);
        } else if (endswith(path, ".txt")) {
            data = load_txt(path, descs);
        } else {
            UNI_ERROR_LOG("can not load image data %s.\n", path.c_str());
        }
        datas->push_back(data);
        paths.push_back(path);
    }
    return paths;
}
#endif

std::vector<std::string> load_data(
    std::string directory, std::vector<TensorDesc> desc, std::vector<std::vector<Tensor>> *datas)
{
    std::vector<std::string> paths;
    if (datas == NULL) {
        return paths;
    }
    std::vector<Tensor> data;
    if (directory == "") {
        data = load_fake_data(desc);
        datas->push_back(data);
        paths.push_back("fake data");
        return paths;
    }
    std::vector<std::string> names = search_files(directory);
    for (U32 i = 0; i < names.size(); i++) {
        std::string path = directory + "/" + names[i];
        if (endswith(path, ".txt")) {
            data = load_txt(path, desc);
        } else if (endswith(path, ".seq")) {
            data = load_seq(path, desc);
        } else {
            UNI_ERROR_LOG("can not load data %s.\n", path.c_str());
        }
        datas->push_back(data);
        paths.push_back(path);
    }
    return paths;
}

std::map<std::string, TensorDesc> load_shape(
    std::string path, std::map<std::string, TensorDesc> descs)
{
    std::map<std::string, TensorDesc> ret;
    if (path == "") {
        return ret;
    }
    FILE *fp = fopen(path.c_str(), "r");
    if (!fp) {
        return ret;
    }
    UNI_DEBUG_LOG(
        "read shape information from %s with cache(%d).\n", path.c_str(), descs.size() != 0);
    char line[512];
    while ((fgets(line, 512, fp)) != NULL) {
        auto items = split(line, " ");
        if (items.size() <= 1) {
            continue;
        }
        std::string name = items[0];
        std::vector<int> shape;
        for (U32 i = 1; i < items.size(); i++) {
            shape.push_back(atoi(items[i].c_str()));
        }
        if (name != "" && shape.size() > 0) {
            TensorDesc desc;
            if (descs.size() > 0) {
                if (descs.find(name) != descs.end()) {
                    desc = descs[name];
                } else {
                    UNI_ERROR_LOG("invalid shape name: %s\n", name.c_str());
                }
            }
            desc.nDims = shape.size();
            for (int i = shape.size() - 1; i >= 0; i--) {
                desc.dims[i] = shape[shape.size() - 1 - i];
            }
            ret[name] = desc;
            UNI_DEBUG_LOG("    name:%s desc:%s\n", name.c_str(), tensorDesc2Str(desc).c_str());
        } else {
            UNI_ERROR_LOG("invalid shape representation: %s\n", line);
        }
    }
    fclose(fp);
    return ret;
}
#endif
