// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef HSWISH_ACTIVATION_H
#define HSWISH_ACTIVATION_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{
/**
 * @brief H-swish (Hard Swish) activation function
 *
 *  Function:
 *  \f[
 *      \mathrm{h-swish}(x) = x \frac{\mathrm{ReLU6}(x+3)}{6},\\
 *      \mathrm{ReLU6}(x) = \min(\max(0,x),6)
 *  \f]
 *
 *  Derivative:
 *  \f[
 *  \mathrm{h-swish}^'(x) =
 *  \left\{
 *      \begin{array}{ll}
 *          1   &,  x > 3 \\
 *          1/6 (2x +3) &, -3 < x < 3 \\
 *          0 &, x < -3 \\
 *          \mathrm{indeterminate} &, \mathrm{otherwise}
 *      \end{array}
 *  \right.
 *  \f]
 *
 *  @see
 *  - A. Howard et al., “Searching for MobileNetV3” arXiv:1905.02244 [cs], Nov. 2019.
 *  - R. Avenash and P. Viswanath, “Semantic Segmentation of Satellite Images using a Modified CNN with Hard-Swish Activation Function” in VISIGRAPP, 2019, doi: 10.5220/0007469604130420.
 */
class HSwishActivation : public BasicLayer
{
  public:
    HSwishActivation(const Name& name, const HSwishActivationParams& params, NetworkParameters& networkParameters);

    HSwishActivation(HSwishActivation&&) = default;
    HSwishActivation(const HSwishActivation&) = delete;
    HSwishActivation& operator=(const HSwishActivation&) = delete;

  private:
    Name mInputName;
    Name mOutputName;

    Limit m3PointVal;
    Limit p3PointVal;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    template<typename MM>
    friend class HSwishActivationCPU;
};
} // raul namespace
#endif