// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstdint>
#include <cstdio>

#include "CIFARParser.h"

namespace raul
{

void CIFARPArser::LoadData(std::unique_ptr<TensorU8>& images, std::unique_ptr<TensorU8>& labels, std::istream& stream)
{
    const long packetSize = 3073;
    const long imageSize = packetSize - 1;
    uint8_t label;
    // bool readCorrected = true;
    stream.seekg(0, stream.end);
    size_t res = static_cast<size_t>(stream.tellg());
    stream.seekg(0, stream.beg);
    try
    {
        std::unique_ptr<TensorU8> newLabels = std::make_unique<TensorU8>(labels->size() + res / packetSize);
        if (!labels->empty()) std::copy(labels->begin(), labels->end(), newLabels->begin());

        std::unique_ptr<TensorU8> newImages = std::make_unique<TensorU8>(images->size() + (res / packetSize) * imageSize);
        if (!images->empty()) std::copy(images->begin(), images->end(), newImages->begin());

        size_t cur_pos = images->size();

        for (size_t q = 0; q < res / packetSize; ++q)
        {
            stream.read((char*)&label, 1);
            (*newLabels)[labels->size() + q] = label;
            stream.read((char*)&(*newImages)[cur_pos], imageSize);

            cur_pos += imageSize;
        }

        labels = std::move(newLabels);
        images = std::move(newImages);
    }
    catch (...)
    {
        THROW_NONAME("CIFARPArser", "problems with loading");
    }
}

void CIFARPArser::LoadData(std::unique_ptr<TensorU8>& images, std::unique_ptr<TensorU8>& labels, const std::filesystem::path& path)
{
    const std::string filename = path.string();
    FILE* file = fopen(filename.c_str(), "rb");

    const long packetSize = 3073;
    const long imageSize = packetSize - 1;

    if (!file) THROW_NONAME("CIFARPArser", "File \"" + filename + "\" with datasets doesn't exist");

    long size_file;
    if (fseek(file, 0, SEEK_END) == 0 && ((size_file = ftell(file)) > 0) && fseek(file, 0, SEEK_SET) == 0 && size_file % packetSize == 0)
    {
        std::unique_ptr<TensorU8> newLabels = std::make_unique<TensorU8>(labels->size() + size_file / packetSize);
        if (!labels->empty()) std::copy(labels->begin(), labels->end(), newLabels->begin());

        std::unique_ptr<TensorU8> newImages = std::make_unique<TensorU8>(images->size() + (size_file / packetSize) * imageSize);
        if (!images->empty()) std::copy(images->begin(), images->end(), newImages->begin());

        size_t cur_pos = images->size();

        for (long i = 0; i < size_file / packetSize; ++i)
        {
            uint8_t label;

            if (fread(&label, sizeof(uint8_t), 1, file) != 1)
            {
                THROW_NONAME("CIFARPArser", "Label in file is not correct");
            }

            (*newLabels)[labels->size() + i] = label;

            if (fread(&(*newImages)[cur_pos], sizeof(uint8_t), imageSize, file) != imageSize)
            {
                THROW_NONAME("CIFARPArser", "Images in file is not correct");
            }

            cur_pos += imageSize;
        }

        labels = std::move(newLabels);
        images = std::move(newImages);
    }

    fclose(file);
}
} // namespace raul