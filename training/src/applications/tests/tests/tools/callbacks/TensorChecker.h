// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TENSOR_CHECKER_H
#define TENSOR_CHECKER_H

#include <tests/tools/TestTools.h>

#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include <training/common/Common.h>
#include <training/layers/BasicLayer.h>

namespace
{
using namespace std;
using namespace raul;

[[maybe_unused]] void save3(const Tensor& t, string file)
{
    auto r = t.reshape(t.getBatchSize(), t.getHeight() * t.getDepth(), t.getWidth());
    ofstream f(file);
    f << "[" << r.size(0) << " " << r.size(1) << " " << r.size(2) << "]" << endl;
    f << "[";
    for (size_t i = 0; i < r.size(0); ++i)
    {
        f << "[";
        for (size_t j = 0; j < r.size(1); ++j)
        {
            f << "[";
            for (size_t k = 0; k < r.size(2); ++k)
            {
                if (k == 0)
                {
                    f << r[i][j][k];
                }
                else
                {
                    f << " " << r[i][j][k];
                }
            }
            f << "]" << endl;
        }
        f << "]" << endl;
    }
    f << "]" << endl;
}
}

namespace UT::tools::callbacks
{
using namespace std;
using namespace raul;

class TensorChecker
{
  public:
    TensorChecker(const vector<pair<Name, Name>>& tensors, dtype epsAbs, dtype epsRel = -1_dt, bool stopOnFirstError = false)
        : TensorChecker(tensors, {}, epsAbs, epsRel, stopOnFirstError)
    {
    }

    TensorChecker(const vector<pair<Name, Name>>& tensors, const vector<pair<Name, Name>>& gradients, dtype epsAbs, dtype epsRel = -1_dt, bool stopOnFirstError = false)
        : mTensors(tensors)
        , mGradients(gradients)
        , mEpsAbsolute(epsAbs)
        , mEpsRelative(epsRel)
        , mStopOnFirstError(stopOnFirstError)
    {
    }

    void operator()(BasicLayer* layer, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place)
    {
        vector<pair<Name, Name>>* tensors = nullptr;
        Names searchNames;
        if (place == NetworkParameters::CallbackPlace::Before_Forward)
        {
            tensors = &mTensors;
            searchNames = layer->getInputs();
        }
        if (place == NetworkParameters::CallbackPlace::After_Forward)
        {
            tensors = &mTensors;
            searchNames = layer->getOutputs();
        }
        else if (place == NetworkParameters::CallbackPlace::After_Backward)
        {
            tensors = &mGradients;
            searchNames = layer->getInputs();
            for (auto& s : searchNames)
            {
                s = s.grad();
            }
        }

        if (!tensors || tensors->empty())
        {
            return;
        }

        for (const auto& outp : searchNames)
        {
            for (const auto& p : *tensors)
            {
                if (outp == p.first)
                {
                    const auto& output = memory_manager[p.first];
                    const auto& target = memory_manager[p.second];

                    if (output.empty())
                    {
                        return;
                    }

                    ASSERT_EQ(output.getShape(), target.getShape())
                        << " inconsistent shapes: " + p.first + " " + Conversions::toString(output.getShape()) + " vs " + p.second + " " + Conversions::toString(target.getShape());

                    ASSERT_EQ(output.size(), target.size()) << " inconsistent sizes: " + p.first + " " + Conversions::toString(output.size()) + " vs " + p.second + " " +
                                                                   Conversions::toString(target.size());

                    for (size_t i = 0; i < output.size(); ++i)
                    {
                        if (mEpsRelative < 0_dt)
                        {
                            if (mStopOnFirstError)
                            {
                                ASSERT_NEAR(output[i], target[i], mEpsAbsolute) << "at " << i << ", output('" << p.first << "'): " << output[i] << ", target('" << p.second << "'): " << target[i];
                            }
                            else
                            {
                                EXPECT_NEAR(output[i], target[i], mEpsAbsolute) << "at " << i << ", output('" << p.first << "'): " << output[i] << ", target('" << p.second << "'): " << target[i];
                            }
                        }
                        else if (mEpsAbsolute < 0_dt)
                        {
                            if (mStopOnFirstError)
                            {
                                ASSERT_TRUE(UT::tools::expect_near_relative(output[i], target[i], mEpsRelative)) << "at " << i << ", output('" << p.first << "'): " << output[i] << ", target('" << p.second << "'): " << target[i];
                            }
                            else
                            {
                                EXPECT_TRUE(UT::tools::expect_near_relative(output[i], target[i], mEpsRelative)) << "at " << i << ", output('" << p.first << "'): " << output[i] << ", target('" << p.second << "'): " << target[i];
                            }
                        }
                        else
                        {
                            if (std::fabs(output[i] - target[i]) < mEpsAbsolute)
                            {
                                continue;
                            }
                            if (mStopOnFirstError)
                            {
                                ASSERT_TRUE(UT::tools::expect_near_relative(output[i], target[i], mEpsRelative)) << "at " << i << ", output('" << p.first << "'): " << output[i] << ", target('" << p.second << "'): " << target[i];
                            }
                            else
                            {
                                EXPECT_TRUE(UT::tools::expect_near_relative(output[i], target[i], mEpsRelative)) << "at " << i << ", output('" << p.first << "'): " << output[i] << ", target('" << p.second << "'): " << target[i];
                            }
                        }
                    }
                    cout << "Tensor \"" + p.first + "\" checked" << endl;
                }
            }
        }
    }

  private:
    vector<pair<Name, Name>> mTensors;
    vector<pair<Name, Name>> mGradients;
    dtype mEpsAbsolute;
    dtype mEpsRelative;
    bool mStopOnFirstError;
};
} // UT::tools::callbacks

#endif
