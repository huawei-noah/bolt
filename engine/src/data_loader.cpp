// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string>
#include <glob.h>

#include "data_loader.hpp"
#include "image_processing.hpp"


Vec<std::string> load_images(std::string directoryPath, TensorDesc imageDesc, Vec<Tensor>* images, ImageType imageType, float scaleValue) {
    Vec<std::string> imagePaths;
    if (directoryPath == "") {
        std::shared_ptr<U8> imageData = load_fake_image(imageDesc);
        Tensor image(imageDesc, imageData);
        (*images).push_back(image);
        imagePaths.push_back("");
        return imagePaths;
    }

    if (directoryPath.back() == '/') {
        directoryPath = directoryPath + "*";
    } else {
        directoryPath = directoryPath + "/*";
    }

    glob_t globResult;
    glob(directoryPath.c_str(), GLOB_TILDE, NULL, &globResult);

    for (U32 i=0; i < globResult.gl_pathc; i++) {
        std::string imagePath = globResult.gl_pathv[i];
        std::shared_ptr<U8> imageData = load_resize_image(imagePath, imageDesc, imageType, scaleValue);
        Tensor image(imageDesc, imageData);
        (*images).push_back(image);
        imagePaths.push_back(imagePath);
    }
    return imagePaths;
}


template<typename T>
void init_one(T* data, U32 len) {
    for (U32 i = 0; i < len; i++) {
        data[i] = 1;
    }
        
}


Vec<Tensor> load_fake_sequence(TensorDesc sequenceDesc) {
    Vec<Tensor> result;

    // word
    TensorDesc wordDesc = sequenceDesc;
    wordDesc.dt = DT_U32;
    U32* wordPtr = (U32*)operator new(tensorNumBytes(wordDesc));
    init_one<U32>(wordPtr, tensorNumElements(wordDesc));
    std::shared_ptr<U8> wordVal((U8*)wordPtr);
    Tensor word(wordDesc, wordVal);
    result.push_back(word);

    // position
    TensorDesc positionDesc = sequenceDesc;
    positionDesc.dt = DT_U32;
    U32* positionPtr = (U32 *)operator new(tensorNumBytes(positionDesc));
    init_one<U32>(positionPtr, tensorNumElements(positionDesc));
    std::shared_ptr<U8> positionVal((U8*)positionPtr);
    Tensor position(positionDesc, positionVal);
    result.push_back(position);

    // token type
    TensorDesc tokenTypeDesc = sequenceDesc;
    tokenTypeDesc.dt = DT_U32;
    U32* tokenTypePtr = (U32 *)operator new(tensorNumBytes(tokenTypeDesc));
    init_one<U32>(tokenTypePtr, tensorNumElements(tokenTypeDesc));
    std::shared_ptr<U8> tokenTypeVal((U8*)tokenTypePtr);
    Tensor tokenType(tokenTypeDesc, tokenTypeVal);
    result.push_back(tokenType);

    // input mask
    TensorDesc inputMaskDesc = sequenceDesc;
    inputMaskDesc.dt = DT_F16;
    F16* inputMaskPtr = (F16 *)operator new(tensorNumBytes(inputMaskDesc));
    init_one<F16>(inputMaskPtr, tensorNumElements(inputMaskDesc));
    std::shared_ptr<U8> inputMaskVal((U8*)inputMaskPtr);
    Tensor inputMask(inputMaskDesc, inputMaskVal);
    result.push_back(inputMask);

    return result;
}


Vec<Tensor> load_sequence(std::string sequencePath, TensorDesc sequenceDesc) {
    Vec<Tensor> result;
    FILE *f = fopen(sequencePath.c_str(), "r");

    // word
    TensorDesc wordDesc = sequenceDesc;
    wordDesc.dt = DT_U32;
    U32* wordPtr = (U32 *)operator new(tensorNumBytes(wordDesc));
    for (U32 i = 0; i < tensorNumElements(wordDesc); i++) {
        fscanf(f, "%u", wordPtr+i);
    }
    std::shared_ptr<U8> wordVal((U8*)wordPtr);
    Tensor word(wordDesc, wordVal);
    result.push_back(word);

    // position
    TensorDesc positionDesc = sequenceDesc;
    positionDesc.dt = DT_U32;
    U32* positionPtr = (U32*)operator new(tensorNumBytes(positionDesc));
    for (U32 i = 0; i < tensorNumElements(positionDesc); i++) {
        fscanf(f, "%u", positionPtr + i);
    }
    std::shared_ptr<U8> positionVal((U8*)positionPtr);
    Tensor position(positionDesc, positionVal);
    result.push_back(position);

    // token type
    TensorDesc tokenTypeDesc = sequenceDesc;
    tokenTypeDesc.dt = DT_U32;
    U32 *tokenTypePtr = (U32 *)operator new(tensorNumBytes(tokenTypeDesc));
    for (U32 i = 0; i < tensorNumElements(tokenTypeDesc); i++) {
        fscanf(f, "%u", tokenTypePtr + i);
    }
    std::shared_ptr<U8> tokenTypeVal((U8*)tokenTypePtr);
    Tensor tokenType(tokenTypeDesc, tokenTypeVal);
    result.push_back(tokenType);

    // input mask
    TensorDesc inputMaskDesc = sequenceDesc;
    inputMaskDesc.dt = DT_F16;
    F16* inputMaskPtr = (F16*)operator new(tensorNumBytes(inputMaskDesc));
    for (U32 i = 0; i < tensorNumElements(positionDesc); i++) {
        F32 value = 0;
        fscanf(f, "%f", &value);
        inputMaskPtr[i] = value;
    }
    std::shared_ptr<U8> inputMaskVal((U8*)inputMaskPtr);
    Tensor inputMask(inputMaskDesc, inputMaskVal);
    result.push_back(inputMask);
    fclose(f);
    return result;
}



Vec<std::string> load_sequences(std::string directoryPath, TensorDesc sequenceDesc, Vec<Vec<Tensor>>* sequences) {
    Vec<std::string> sequencePaths;
    if (directoryPath == "") {
        Vec<Tensor> sequence = load_fake_sequence(sequenceDesc);
        (*sequences).push_back(sequence);
        sequencePaths.push_back("");
        return sequencePaths;
    }

    if (directoryPath.back() == '/') {
        directoryPath = directoryPath + "*";
    } else {
        directoryPath = directoryPath + "/*";
    }

    glob_t globResult;
    glob(directoryPath.c_str(), GLOB_TILDE, NULL, &globResult);

    for (U32 i = 0; i < globResult.gl_pathc; i++) {
        std::string sequencePath = globResult.gl_pathv[i];
        Vec<Tensor> sequence = load_sequence(sequencePath, sequenceDesc);
        (*sequences).push_back(sequence);
        sequencePaths.push_back(sequencePath);
    }
    return sequencePaths;
}

