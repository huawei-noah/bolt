// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LAYER_TYPE_STATISTICS_H
#define LAYER_TYPE_STATISTICS_H

#include <tests/tools/TestTools.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <vector>

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace UT::tools::callbacks
{
using namespace std;
using namespace raul;

struct LayerTypeStat
{
    LayerTypeStat(Name name = "")
        : mTypeName(name)
    {
    }
    Name mTypeName;
    size_t mCallCount = 0;
    float mLongestForwardCall = 0;
    float mLongestBackwardCall = 0;
    float mTotalForwardTime = 0;
    float mTotalBackwardTime = 0;

    void print(ostream& o, float epochTime) const
    {
        o << mTypeName << endl;
        o << "  calls: " << mCallCount << endl;
        o << "  total forward: " << mTotalForwardTime << " (" << 100.f * mTotalForwardTime / epochTime << "%)" << endl;
        o << "  total backward: " << mTotalBackwardTime << " (" << 100.f * mTotalBackwardTime / epochTime << "%)" << endl;
        o << "  longest forward: " << mLongestForwardCall << " (" << 100.f * mLongestForwardCall / epochTime << "%)" << endl;
        o << "  longest backward: " << mLongestBackwardCall << " (" << 100.f * mLongestBackwardCall / epochTime << "%)" << endl;
        o << "  average forward: " << mTotalForwardTime / static_cast<float>(mCallCount) << endl;
        o << "  average backward: " << mTotalBackwardTime / static_cast<float>(mCallCount) << endl;
    }

    void operator+=(LayerTypeStat s) 
    { 
        mCallCount += s.mCallCount;
        mLongestForwardCall += s.mLongestForwardCall;
        mLongestBackwardCall += s.mLongestBackwardCall;
        mTotalForwardTime += s.mTotalForwardTime;
        mTotalBackwardTime += s.mTotalBackwardTime;
    }
};

class LayerTypeStatistics
{
  public:
    LayerTypeStatistics() {}
    ~LayerTypeStatistics(){}
    void operator()(BasicLayer* layer, const MemoryManager&, NetworkParameters::CallbackPlace place)
    {
        auto name = layer->getName();
        if (place == NetworkParameters::CallbackPlace::Before_Forward || place == NetworkParameters::CallbackPlace::Before_Backward)
        {
            if (mLayerStarts.find(name) != mLayerStarts.end())
            {
                throw runtime_error("LayerTypeStatistics: layer \"" + name + "\" started twice");
            }
            mLayerStarts[name] = chrono::steady_clock::now();
        }
        else
        {
            if (mLayerStarts.find(name) == mLayerStarts.end())
            {
                throw runtime_error("LayerTypeStatistics: layer \"" + name + "\" started twice");
            }
            float duration = static_cast<float>(chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - mLayerStarts[name]).count()) / 1000000.f;
            mLayerStarts.erase(name);
            auto typeName = layer->getTypeName();
            auto& s = mStatMap[typeName];
            s.mTypeName = typeName;
            if (place == NetworkParameters::CallbackPlace::After_Forward)
            {
                ++s.mCallCount;
                s.mLongestForwardCall = max(s.mLongestForwardCall, duration);
                s.mTotalForwardTime += duration;
            }
            else
            {
                s.mLongestBackwardCall = max(s.mLongestBackwardCall, duration);
                s.mTotalBackwardTime += duration;
            }
        }
    }

    vector<LayerTypeStat> getStat() const
    {
        vector<LayerTypeStat> v;
        for (const auto& it : mStatMap)
        {
            v.push_back(it.second);
        }
        sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.mTotalForwardTime + a.mTotalBackwardTime > b.mTotalForwardTime + b.mTotalBackwardTime; });
        return v;
    }

    void print(float forwardTime, float backwardTime, float minPercent = 1.f) const
    {
        auto epochTime = forwardTime + backwardTime;

        float totalLayerForwardTime = 0;
        float totalLayerBackwardTime = 0;

        auto stat = getStat();

        for (const auto& s : stat)
        {
            totalLayerForwardTime += s.mTotalForwardTime;
            totalLayerBackwardTime += s.mTotalBackwardTime;
        }

        cout << "Calculations take " << totalLayerForwardTime + totalLayerBackwardTime << "ms (" << 100.f * (totalLayerForwardTime + totalLayerBackwardTime) / epochTime << "% of " << epochTime
             << "ms epoch time)" << endl;

        cout << "  forward: " << totalLayerForwardTime << "ms (" << 100.f * totalLayerForwardTime / forwardTime << "% of " << forwardTime << "ms forward time)" << endl;
        cout << "  backward: " << totalLayerBackwardTime << "ms (" << 100.f * totalLayerBackwardTime / backwardTime << "% of " << backwardTime << "ms backward time)" << endl;

        LayerTypeStat other("Other");

        for (const auto& s : stat)
        {
            if (s.mTotalForwardTime >= totalLayerForwardTime * minPercent / 100.f || s.mTotalBackwardTime >= totalLayerBackwardTime * minPercent / 100.f)
            {
                s.print(cout, epochTime);
            }
            else
            {
                other += s;
            }
        }
        other.print(cout, epochTime);
    }

  protected:
    map<Name, LayerTypeStat> mStatMap;                        // type name
    map<Name, chrono::steady_clock::time_point> mLayerStarts; // layer name
};

} // UT::tools::callbacks

#endif
