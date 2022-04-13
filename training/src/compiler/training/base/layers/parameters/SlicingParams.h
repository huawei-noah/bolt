// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SLICING_PARAMS_H
#define SLICING_PARAMS_H

#include <string>
#include <vector>

#include "BasicParameters.h"

namespace raul
{

/** Parameters for splitting input tensor along specified axis
 * @param paramDimStr axis for splitting (possible values: "width", "height", "depth", "default" = "width")
 *
 * Usage examples:
 * SlicingParams(input, outputs, "width") - outputs.size() equal slices
 * SlicingParams(input, outputs, "width", {1, 2}) - two slices with size 1 and 2 (remaining part will be omitted)
 * SlicingParams(input, outputs, "width", {1, -1, 2}) - three slices with. first - 1, third - 2, second - remaining part
 * SlicingParams(input, outputs, "width", {-1, 1}) - two slices with. second - 1, first - remaining part
 * SlicingParams(input, outputs, "width", {1, -1}) - two slices with. first - 1, second - remaining part
 */
struct SlicingParams : public BasicParamsWithDim
{
    SlicingParams() = delete;

    SlicingParams(const Name& input, const Names& outputs, const std::string& paramDimStr = "width", std::vector<int> slice_sizes = std::vector<int>());

    SlicingParams(const Name& input, const Names& outputs, Dimension paramDim, std::vector<int> slice_sizes = std::vector<int>());

    std::vector<int> sliceSize;

    void print(std::ostream& stream) const override;
};

} // raul namespace
#endif // SLICING_PARAMS_H
