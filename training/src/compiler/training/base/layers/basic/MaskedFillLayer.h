// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef MASKED_FILL_LAYER_H
#define MASKED_FILL_LAYER_H

#include <training/base/layers/BasicLayer.h>

#include <training/base/common/Common.h>

namespace raul
{
/**
 * @brief Masked fill
 *
 *  Fills input tensor elements corresponding to ones in mask with fill value.
 *  Mask can have separate entries for each channel or single entry for all channels.
 *
 */
class MaskedFillLayer : public BasicLayer
{
  public:
    MaskedFillLayer(const Name& name, const MaskedFillParams& params, NetworkParameters& networkParameters);

    MaskedFillLayer(MaskedFillLayer&&) = default;
    MaskedFillLayer(const MaskedFillLayer&) = delete;
    MaskedFillLayer& operator=(const MaskedFillLayer&) = delete;

  private:
    size_t mWidth;
    size_t mHeight;
    size_t mDepth;

    dtype mFillValue;
    bool mInverted;

    Name mInputName;
    std::string mMaskName;
    Name mOutputName;

    template<typename MM>
    friend class MaskedFillLayerCPU;
};
}
#endif