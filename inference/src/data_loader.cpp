// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _BUILD_TEST

#include <algorithm>
#include <string>
#include <dirent.h>
#include <cstddef>
#include <jpeglib.h>
#include <jerror.h>

#include "image_processing.hpp"
#include "data_loader.hpp"

template<typename T>
void init_one(T* data, U32 len) {
    for (U32 i = 0; i < len; i++) {
        data[i] = 1;
    }
}

void get_files(std::string directoryName, Vec<std::string> &files) {
    if (directoryName.empty()) {
        std::cerr << "[ERROR] null data" << std::endl;
        exit(1);
    }
    DIR *directory = opendir(directoryName.c_str());
    if (NULL == directory) {
        std::cerr << "[ERROR] permission denied to access " << directoryName << std::endl;
        exit(1);
    }
    struct dirent *file;
    while ((file = readdir(directory)) != NULL) {
        if (strcmp(file->d_name, ".") == 0 || strcmp(file->d_name, "..") == 0) {
            continue;
        }
        if (file->d_type == DT_DIR) {
            //std::string fileName = directoryName + "/" + file->d_name;
            //get_files(fileName, files);
            continue;
        } else {
            files.push_back(directoryName + "/" + file->d_name);
        }
    }
    closedir(directory);
}

Vec<Tensor> load_jpeg(std::string dataPath, Vec<TensorDesc> imageDesc, ImageFormat ImageFormat, F32 scaleValue) {
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
    U32 dataSize =  numChannels * height * width;

    DEBUG_info("[INFO] " << dataPath << ": channels " << numChannels << ", out color space " << info.out_color_space);
    CHECK_REQUIREMENT(2 == info.out_color_space);  // Support RGB for now

    U8 *data = (U8*)malloc(dataSize);
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
    U8 *rgb = (U8*)malloc(tensorNumBytes(rgbDesc));

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

    std::shared_ptr<U8> imageData = load_resize_image(rgbDesc, rgb, imageDesc[0], ImageFormat, scaleValue);
    free(rgb);

    Vec<Tensor> result;
    std::shared_ptr<Tensor> tensorData(new Tensor());
    tensorData->set_desc(imageDesc[0]);
    tensorData->set_shared_ptr(imageData);
    result.push_back(*tensorData.get());
    return result;
}

Vec<Tensor> load_fake_data(Vec<TensorDesc> dataDesc) {
    Vec<Tensor> result;
    for (U32 index = 0; index < dataDesc.size(); index++) {
        U8 *ptr = nullptr;
        switch (dataDesc[index].dt) {
            case DT_F32: {
                F32* dataPtr = (F32 *)operator new(tensorNumBytes(dataDesc[index]));
                init_one<F32>(dataPtr, tensorNumElements(dataDesc[index]));
                ptr = (U8 *)dataPtr;
                break;
            }
#ifdef _USE_FP16
            case DT_F16: {
                F16* dataPtr = (F16 *)operator new(tensorNumBytes(dataDesc[index]));
                init_one<F16>(dataPtr, tensorNumElements(dataDesc[index]));
                ptr = (U8 *)dataPtr;
                break;
            }
#endif
            case DT_U32: {
                U32* dataPtr = (U32 *)operator new(tensorNumBytes(dataDesc[index]));
                init_one<U32>(dataPtr, tensorNumElements(dataDesc[index]));
                ptr = (U8 *)dataPtr;
                break;
            }
            case DT_I32: {
                I32* dataPtr = (I32 *)operator new(tensorNumBytes(dataDesc[index]));
                init_one<I32>(dataPtr, tensorNumElements(dataDesc[index]));
                ptr = (U8 *)dataPtr;
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
                break;
        }
        std::shared_ptr<Tensor> data(new Tensor());
        data->set_desc(dataDesc[index]);
        data->set_val(ptr);
        result.push_back(*data.get());
    }
    return result;
}

Vec<Tensor> load_txt(std::string dataPath, Vec<TensorDesc> dataDesc) {
    Vec<Tensor> result;
    FILE *f = fopen(dataPath.c_str(), "r");
    CHECK_REQUIREMENT(f != nullptr);
    for (U32 index = 0; index < dataDesc.size(); index++) {
        U8 *ptr = nullptr;
        switch (dataDesc[index].dt) {
            case DT_F32: {
                F32* dataPtr = (F32 *)operator new(tensorNumBytes(dataDesc[index]));
                for (U32 i = 0; i < tensorNumElements(dataDesc[index]); i++) {
                    fscanf(f, "%f", dataPtr+i);
                }
                ptr = (U8 *)dataPtr;
                break;
            }
#ifdef _USE_FP16
            case DT_F16: {
                F16* dataPtr = (F16 *)operator new(tensorNumBytes(dataDesc[index]));
                F32 value;
                for (U32 i = 0; i < tensorNumElements(dataDesc[index]); i++) {
                    fscanf(f, "%f", &value);
                    dataPtr[i] = (F16)value;
                }
                ptr = (U8 *)dataPtr;
                break;
            }
#endif
            case DT_U32: {
                U32* dataPtr = (U32 *)operator new(tensorNumBytes(dataDesc[index]));
                for (U32 i = 0; i < tensorNumElements(dataDesc[index]); i++) {
                    fscanf(f, "%u", dataPtr+i);
                }
                ptr = (U8 *)dataPtr;
                break;
            }
            case DT_I32: {
                I32* dataPtr = (I32 *)operator new(tensorNumBytes(dataDesc[index]));
                for (U32 i = 0; i < tensorNumElements(dataDesc[index]); i++) {
                    fscanf(f, "%d", dataPtr+i);
                }
                ptr = (U8 *)dataPtr;
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
                break;
        }
        std::shared_ptr<Tensor> data(new Tensor());
        data->set_desc(dataDesc[index]);
        data->set_val(ptr);
        result.push_back(*data.get());
    }
    fclose(f);
    return result;
}

Vec<Tensor> load_seq(std::string dataPath, Vec<TensorDesc> dataDesc) {
    Vec<Tensor> result;
    FILE *f = fopen(dataPath.c_str(), "r");
    CHECK_REQUIREMENT(f != nullptr);
    for (U32 index = 0; index < dataDesc.size(); index++) {
        U32 sequenceLen = 0;
        fscanf(f, "%u", &sequenceLen);
        TensorDesc sequenceDesc = dataDesc[index];
        sequenceDesc.dims[0] = sequenceLen;

        U8 *ptr = nullptr;
        switch (dataDesc[index].dt) {
            case DT_F32: {
                F32* dataPtr = (F32 *)operator new(tensorNumBytes(sequenceDesc));
                for (U32 i = 0; i < tensorNumElements(sequenceDesc); i++) {
                    fscanf(f, "%f", dataPtr+i);
                }
                ptr = (U8 *)dataPtr;
                break;
            }
#ifdef _USE_FP16
            case DT_F16: {
                F16* dataPtr = (F16 *)operator new(tensorNumBytes(sequenceDesc));
                F32 value;
                for (U32 i = 0; i < tensorNumElements(sequenceDesc); i++) {
                    fscanf(f, "%f", &value);
                    dataPtr[i] = (F16)value;
                }
                ptr = (U8 *)dataPtr;
                break;
            }
#endif
            case DT_U32: {
                U32* dataPtr = (U32 *)operator new(tensorNumBytes(sequenceDesc));
                for (U32 i = 0; i < tensorNumElements(sequenceDesc); i++) {
                    fscanf(f, "%u", dataPtr+i);
                }
                ptr = (U8 *)dataPtr;
                break;
            }
            case DT_I32: {
                I32* dataPtr = (I32 *)operator new(tensorNumBytes(sequenceDesc));
                for (U32 i = 0; i < tensorNumElements(sequenceDesc); i++) {
                    fscanf(f, "%d", dataPtr+i);
                }
                ptr = (U8 *)dataPtr;
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
                break;
        }

        std::shared_ptr<Tensor> data(new Tensor());
        data->set_desc(sequenceDesc);
        data->set_val(ptr);
        result.push_back(*data.get());
    }
    fclose(f);
    return result;
}

Vec<Tensor> load_bin(std::string dataPath, Vec<DataType> sourceDataType, Vec<TensorDesc> dataDesc) {
    Vec<Tensor> result;
#ifdef _USE_FP16
    FILE *f = fopen(dataPath.c_str(), "r");
    CHECK_REQUIREMENT(f != nullptr);
    for (U32 index = 0; index < dataDesc.size(); index++) {
        U32 len = tensorNumElements(dataDesc[index]);
        U8* ptr = (U8 *)operator new(len * bytesOf(sourceDataType[index]));
        U32 readLength = fread(ptr, bytesOf(sourceDataType[index]), len, f);
        CHECK_REQUIREMENT(len == readLength);

        U8 *ptrNew = nullptr;
        if (sourceDataType[index] != dataDesc[index].dt) {
            ptrNew = (U8 *)operator new(len * bytesOf(dataDesc[index].dt));
            if (sourceDataType[index] == DT_F32 && dataDesc[index].dt == DT_F16) {
                F32* ptr1 = (F32*)ptr;
                F16* ptr2 = (F16*)ptrNew;
                for (U32 i = 0; i < len; i++)
                    ptr2[i] = (F16)ptr1[i];
            }
            else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        else {
            ptrNew = ptr;
        }
        std::shared_ptr<Tensor> data(new Tensor());
        data->set_desc(dataDesc[index]);
        data->set_val(ptrNew);
        result.push_back(*data.get());
    }
    fclose(f);
#endif
    return result;
}

int string_end_with(std::string s, std::string sub){
    std::transform(s.begin(),   s.end(),   s.begin(),   ::tolower);
    std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);
    return s.rfind(sub)==(s.length() - sub.length()) ? 1 : 0;
}

Vec<std::string> load_data(std::string directoryPath,
    Vec<TensorDesc> dataDesc,
    Vec<Vec<Tensor>>* datas)
{
    Vec<std::string> dataPaths;
    if (directoryPath == "") {
        Vec<Tensor> data = load_fake_data(dataDesc);
        (*datas).push_back(data);
        dataPaths.push_back("fake data");
        return dataPaths;
    }

    Vec<std::string> paths;
    get_files(directoryPath, paths);
    Vec<Tensor> data;
    for (U32 i = 0; i < paths.size(); i++) {
        std::string dataPath = paths[i];
        if (string_end_with(dataPath, ".txt"))
            data = load_txt(dataPath, dataDesc);
        else if (string_end_with(dataPath, ".seq"))
            data = load_seq(dataPath, dataDesc);
        else {
            std::cerr << "[ERROR] can not load data " << dataPath << std::endl;
            exit(1);
        }
        (*datas).push_back(data);
        dataPaths.push_back(dataPath);
    }
    return dataPaths;
}

Vec<std::string> load_image_with_scale(std::string directoryPath,
    Vec<TensorDesc> dataDesc,
    Vec<Vec<Tensor>>* datas,
    ImageFormat ImageFormat,
    F32 scaleValue)
{
    Vec<std::string> dataPaths;
    if (directoryPath == "") {
        Vec<Tensor> data = load_fake_data(dataDesc);
        (*datas).push_back(data);
        dataPaths.push_back("fake data");
        return dataPaths;
    }

    Vec<std::string> paths;
    get_files(directoryPath, paths);
    Vec<Tensor> data;
    for (U32 i = 0; i < paths.size(); i++) {
        std::string dataPath = paths[i];
        if (string_end_with(dataPath, ".jpg") || string_end_with(dataPath, ".jpeg"))
            data = load_jpeg(dataPath, dataDesc, ImageFormat, scaleValue);
        else if (string_end_with(dataPath, ".txt"))
            data = load_txt(dataPath, dataDesc);
        else {
            std::cerr << "[ERROR] can not load jpeg data " << dataPath << std::endl;
            exit(1);
        }
        (*datas).push_back(data);
        dataPaths.push_back(dataPath);
    }
    return dataPaths;
}

Vec<std::string> load_bin_with_type(std::string directoryPath,
    Vec<TensorDesc> dataDesc,
    Vec<Vec<Tensor>>* datas,
    Vec<DataType> sourceDataType)
{
    Vec<std::string> dataPaths;
    if (directoryPath == "") {
        Vec<Tensor> data = load_fake_data(dataDesc);
        (*datas).push_back(data);
        dataPaths.push_back("fake data");
        return dataPaths;
    }

    Vec<std::string> paths;
    get_files(directoryPath, paths);
    Vec<Tensor> data;
    for (U32 i = 0; i < paths.size(); i++) {
        std::string dataPath = paths[i];
        if (string_end_with(dataPath, ".bin"))
            data = load_bin(dataPath, sourceDataType, dataDesc);
        else {
            std::cerr << "[ERROR] can not load binary data " << dataPath << std::endl;
            exit(1);
        }
        (*datas).push_back(data);
        dataPaths.push_back(dataPath);
    }
    return dataPaths;
}
#endif
