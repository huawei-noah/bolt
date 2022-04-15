// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>
#include <turbojpeg.h>

#include <filesystem>

enum PixelFormat
{
    RGB = 0,
    BGR,
    RGBX,
    BGRX,
    XBGR,
    XRGB,
    GRAY,
    RGBA,
    BGRA,
    ABGR,
    ARGB,
    CMYK,
    UNKNOWN = -1
};

/**
 * @brief Jpeg Image
 *
 * A turbojpeg wrapper
 *
 */
class JpegImage
{
  private:
    size_t mWidth, mHeight;
    mutable std::vector<uint8_t> mBuffer;
    PixelFormat mPixelFormat;

  public:
    JpegImage()
    {
        mPixelFormat = PixelFormat::RGB;
        mWidth = 0;
        mHeight = 0;
    }
    bool OpenJpeg(const std::filesystem::path& filename);
    bool SaveJpeg(const std::filesystem::path& filename) const;
    std::vector<uint8_t>& getBuffer();
    const std::vector<uint8_t>& getBuffer() const;
    size_t getWidth() const;
    size_t getHeight() const;
    size_t getPixelFormat() const;
};

bool JpegImage::OpenJpeg(const std::filesystem::path& filename)
{
    int inSubsamp = -1, inColorSpace = -1;
    tjscalingfactor scalingFactor = { 1, 1 };
    unsigned long jpegSize = 0;
    int flags = 0;
    tjhandle tjInstance = NULL;
    FILE* jpegFile = NULL;
    std::vector<unsigned char> buffer;
    if ((jpegFile = fopen(filename.string().c_str(), "rb")) == NULL) return false;
    if (fseek(jpegFile, 0, SEEK_END) < 0 || (!(jpegSize = ftell(jpegFile))) || fseek(jpegFile, 0, SEEK_SET) < 0) return false;
    if (jpegSize == 0) return false;

    buffer = std::vector<unsigned char>(jpegSize);
    if ((tjInstance = tjInitDecompress()) == NULL) return false;
    if (fread(buffer.data(), jpegSize, 1, jpegFile) < 1) return false;
    fclose(jpegFile);
    jpegFile = NULL;
    if (tjDecompressHeader3(tjInstance, buffer.data(), jpegSize, (int*)&mWidth, (int*)&mHeight, &inSubsamp, &inColorSpace) < 0) return false;
    mWidth = TJSCALED(mWidth, scalingFactor);
    mHeight = TJSCALED(mHeight, scalingFactor);
    mBuffer = std::vector<unsigned char>(mHeight * mWidth * tjPixelSize[mPixelFormat]);
    if (tjDecompress2(tjInstance, buffer.data(), jpegSize, mBuffer.data(), static_cast<int>(mWidth), 0, static_cast<int>(mHeight), mPixelFormat, flags) < 0) return false;
    tjDestroy(tjInstance);
    tjInstance = NULL;
    return true;
}

bool JpegImage::SaveJpeg(const std::filesystem::path& filename) const
{
    const auto cond = tjSaveImage(filename.string().c_str(), mBuffer.data(), static_cast<int>(mWidth), 0, static_cast<int>(mHeight), mPixelFormat, 0);
    return cond >= 0;
}

const std::vector<uint8_t>& JpegImage::getBuffer() const
{
    return mBuffer;
}
std::vector<uint8_t>& JpegImage::getBuffer()
{
    return mBuffer;
}

size_t JpegImage::getWidth() const
{
    return mWidth;
}

size_t JpegImage::getHeight() const
{
    return mHeight;
}

size_t JpegImage::getPixelFormat() const
{
    return mPixelFormat;
}