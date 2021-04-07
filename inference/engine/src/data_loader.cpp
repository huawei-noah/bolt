// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <stddef.h>
#include <stdlib.h>
#include "data_loader.hpp"

#ifdef _BUILD_TEST
#include <jpeglib.h>
#include <jerror.h>
#include "image_processing.hpp"

std::vector<Tensor> load_jpeg(
    std::string dataPath, std::vector<TensorDesc> imageDesc, ImageFormat ImageFormat, F32 scaleValue)
{
    FILE *file = fopen(dataPath.c_str(), "rb");
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

    UNI_DEBUG_LOG("%s: channels %u , out color space %d\n", dataPath.c_str(), numChannels,
        info.out_color_space);
    CHECK_REQUIREMENT(2 == info.out_color_space);  // Support RGB for now

    U8 *data = (U8 *)malloc(dataSize);
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
    free(data);

    std::shared_ptr<Tensor> imageTensor =
        load_resize_image(rgbTensor, imageDesc[0], ImageFormat, scaleValue);
    std::vector<Tensor> result;
    imageTensor->resize(imageDesc[0]);
    result.push_back(*imageTensor.get());
    return result;
}

std::vector<std::string> load_image_with_scale(std::string directoryPath,
    std::vector<TensorDesc> dataDesc,
    std::vector<std::vector<Tensor>> *datas,
    ImageFormat ImageFormat,
    F32 scaleValue)
{
    std::vector<std::string> dataPaths;
    if (directoryPath == "") {
        std::vector<Tensor> data = load_fake_data(dataDesc);
        (*datas).push_back(data);
        dataPaths.push_back("fake data");
        return dataPaths;
    }

    std::vector<std::string> paths;
    get_files(directoryPath, paths);
    std::vector<Tensor> data;
    for (U32 i = 0; i < paths.size(); i++) {
        std::string dataPath = paths[i];
        if (string_end_with(dataPath, ".jpg") || string_end_with(dataPath, ".jpeg")) {
            data = load_jpeg(dataPath, dataDesc, ImageFormat, scaleValue);
        } else if (string_end_with(dataPath, ".txt")) {
            data = load_txt(dataPath, dataDesc);
        } else {
            UNI_ERROR_LOG("can not load jpeg data %s\n", dataPath.c_str());
        }
        (*datas).push_back(data);
        dataPaths.push_back(dataPath);
    }
    return dataPaths;
}
#endif

void get_files(std::string directoryName, std::vector<std::string> &files)
{
    if (directoryName.empty()) {
        UNI_ERROR_LOG("null data\n");
    }
    DIR *directory = opendir(directoryName.c_str());
    if (NULL == directory) {
        UNI_ERROR_LOG("permission denied to access %s\n", directoryName.c_str());
    }
    struct dirent *file;
    while ((file = readdir(directory)) != NULL) {
        if (strcmp(file->d_name, ".") == 0 || strcmp(file->d_name, "..") == 0) {
            continue;
        }
        struct stat st;
        stat(file->d_name, &st);
        if (S_ISDIR(st.st_mode)) {
            continue;
        } else {
            files.push_back(directoryName + "/" + file->d_name);
        }
    }
    closedir(directory);
}

Tensor fscanfReadData(FILE *f, TensorDesc desc)
{
    Tensor tensor = Tensor::alloc_sized<CPUMem>(desc);
    U32 size = tensor.length();
    DataType dataType = desc.dt;
    auto ptr = ((CpuMemory *)(tensor.get_memory()))->get_ptr();
    switch (dataType) {
        case DT_F32: {
            F32 *dataPtr = (F32 *)ptr;
            for (U32 i = 0; i < size; i++) {
                fscanf(f, "%f", dataPtr + i);
            }
            break;
        }
#ifdef __aarch64__
        case DT_F16: {
            F16 *dataPtr = (F16 *)ptr;
            F32 value;
            for (U32 i = 0; i < size; i++) {
                fscanf(f, "%f", &value);
                dataPtr[i] = (F16)value;
            }
            break;
        }
#endif
        case DT_U32: {
            U32 *dataPtr = (U32 *)ptr;
            for (U32 i = 0; i < size; i++) {
                fscanf(f, "%u", dataPtr + i);
            }
            break;
        }
        case DT_I32: {
            I32 *dataPtr = (I32 *)ptr;
            for (U32 i = 0; i < size; i++) {
                fscanf(f, "%d", dataPtr + i);
            }
            break;
        }
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return tensor;
}

std::vector<Tensor> load_fake_data(std::vector<TensorDesc> dataDesc)
{
    std::vector<Tensor> result;
    for (U32 index = 0; index < dataDesc.size(); index++) {
        Tensor tensor = Tensor::alloc_sized<CPUMem>(dataDesc[index]);
        U8 *ptr = (U8 *)((CpuMemory *)(tensor.get_memory()))->get_ptr();
        UNI_INIT(tensorNumElements(dataDesc[index]), dataDesc[index].dt, 1, ptr);
        result.push_back(tensor);
    }
    return result;
}

std::vector<Tensor> load_txt(std::string dataPath, std::vector<TensorDesc> dataDesc)
{
    std::vector<Tensor> result;
    FILE *f = fopen(dataPath.c_str(), "r");
    CHECK_REQUIREMENT(f != nullptr);
    for (U32 index = 0; index < dataDesc.size(); index++) {
        result.push_back(fscanfReadData(f, dataDesc[index]));
    }
    fclose(f);
    return result;
}

std::vector<Tensor> load_seq(std::string dataPath, std::vector<TensorDesc> dataDesc)
{
    std::vector<Tensor> result;
    FILE *f = fopen(dataPath.c_str(), "r");
    CHECK_REQUIREMENT(f != nullptr);
    for (U32 index = 0; index < dataDesc.size(); index++) {
        U32 sequenceLen = 0;
        fscanf(f, "%u", &sequenceLen);
        TensorDesc sequenceDesc = dataDesc[index];
        sequenceDesc.dims[0] = sequenceLen;
        for (U32 j = 1; j < sequenceDesc.nDims; j++) {
            sequenceDesc.dims[j] = 1;
        }
        result.push_back(fscanfReadData(f, sequenceDesc));
    }
    fclose(f);
    return result;
}

std::vector<Tensor> load_bin(
    std::string dataPath, std::vector<DataType> sourceDataType, std::vector<TensorDesc> dataDesc)
{
    std::vector<Tensor> result;
    FILE *f = fopen(dataPath.c_str(), "r");
    if (nullptr == f) {
        result = load_fake_data(dataDesc);
    } else {
        for (U32 index = 0; index < dataDesc.size(); index++) {
            TensorDesc sourceDesc = dataDesc[index];
            sourceDesc.dt = sourceDataType[index];
            Tensor tensor = Tensor::alloc_sized<CPUMem>(sourceDesc);
            U32 len = tensor.length();
            auto ptr = ((CpuMemory *)(tensor.get_memory()))->get_ptr();
            U32 readLength = fread(ptr, bytesOf(sourceDataType[index]), len, f);
            CHECK_REQUIREMENT(len == readLength);
            if (sourceDataType[index] != dataDesc[index].dt) {
                Tensor transform_tensor = Tensor::alloc_sized<CPUMem>(dataDesc[index]);
                if (0) {
#ifdef __aarch64__
                } else if (sourceDataType[index] == DT_F32 && dataDesc[index].dt == DT_F16) {
                    F32 *ptr1 = (F32 *)ptr;
                    F16 *ptr2 = (F16 *)((CpuMemory *)(transform_tensor.get_memory()))->get_ptr();
                    for (U32 i = 0; i < len; i++) {
                        ptr2[i] = (F16)ptr1[i];
                    }
#endif
                } else {
                    CHECK_STATUS(NOT_SUPPORTED);
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

int string_end_with(std::string s, std::string sub)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);
    return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

std::vector<std::string> load_data(std::string directoryPath,
    std::vector<TensorDesc> dataDesc,
    std::vector<std::vector<Tensor>> *datas)
{
    std::vector<std::string> dataPaths;
    if (directoryPath == "") {
        std::vector<Tensor> data = load_fake_data(dataDesc);
        (*datas).push_back(data);
        dataPaths.push_back("fake data");
        return dataPaths;
    }

    std::vector<std::string> paths;
    get_files(directoryPath, paths);
    std::vector<Tensor> data;
    for (U32 i = 0; i < paths.size(); i++) {
        std::string dataPath = paths[i];
        if (string_end_with(dataPath, ".txt")) {
            data = load_txt(dataPath, dataDesc);
        } else if (string_end_with(dataPath, ".seq")) {
            data = load_seq(dataPath, dataDesc);
        } else {
            UNI_ERROR_LOG("can not load data %s\n", dataPath.c_str());
        }
        (*datas).push_back(data);
        dataPaths.push_back(dataPath);
    }
    return dataPaths;
}
