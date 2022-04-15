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

#include "MNISTParser.h"

namespace
{
uint32_t reverse32(uint32_t v)
{
    return (v >> 24) | ((v & 0xFF0000) >> 8) | ((v & 0xFF00) << 8) | ((v & 0xFF) << 24);
}
} // anonymous namespace

namespace raul
{

void MNISTParser::LoadData(std::unique_ptr<TensorU8>& labels, std::unique_ptr<TensorU8>& images, const std::filesystem::path& labels_path, const std::filesystem::path& images_path)
{
    const std::string labels_path_str = labels_path.string();
    const std::string images_path_str = images_path.string();

    FILE* file = fopen(labels_path_str.c_str(), "rb");

    if (!file)
    {
        THROW_NONAME("MNISTParser", "File with labels doesn't exist: " + labels_path_str);
    }

    uint32_t magic, labelsCount;

    if (fread(&magic, 4, 1, file) == 1 && fread(&labelsCount, 4, 1, file) == 1)
    {
        magic = reverse32(magic);
        labelsCount = reverse32(labelsCount);

        if (magic == 0x801)
        {
            std::unique_ptr<TensorU8> newLabels = std::make_unique<TensorU8>(labels->size() + labelsCount);
            if (!labels->empty()) std::copy(labels->begin(), labels->end(), newLabels->begin());

            size_t read = fread(&(*newLabels)[labels->size()], 1, labelsCount, file);
            if (read == 0)
            {
                THROW_NONAME("MNISTParser", "Error reading labels file: " + labels_path_str);
            }
            labels = std::move(newLabels);
        }
    }
    fclose(file);

    file = fopen(images_path_str.c_str(), "rb");
    if (!file)
    {
        THROW_NONAME("MNISTParser", "File with images doesn't exist: " + images_path_str);
    }

    uint32_t imageCount;
    uint32_t width;
    uint32_t height;

    if (fread(&magic, 4, 1, file) == 1 && fread(&imageCount, 4, 1, file) == 1 && fread(&height, 4, 1, file) == 1 && fread(&width, 4, 1, file) == 1)
    {
        magic = reverse32(magic);
        imageCount = reverse32(imageCount);
        width = reverse32(width);
        height = reverse32(height);

        if (magic == 0x803)
        {
            size_t step = width * height;

            std::unique_ptr<TensorU8> newImages = std::make_unique<TensorU8>(images->size() + static_cast<size_t>(imageCount * step));
            if (!images->empty()) std::copy(images->begin(), images->end(), newImages->begin());

            size_t cur_pos = images->size();

            for (size_t i = 0; i < imageCount; ++i)
            {
                if (fread(&(*newImages)[cur_pos], 1, step, file) != step)
                {
                    THROW_NONAME("MNISTParser", "Error reading images file: " + images_path_str);
                }
                cur_pos += step;
            }
            images = std::move(newImages);
        }
    }

    fclose(file);
}
} // namespace raul