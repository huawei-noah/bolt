// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TENSORTRACER_H
#define TENSORTRACER_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace UT::tools::callbacks
{

using namespace raul;

template<class Os, class U, class V>
Os& operator<<(Os& os, const std::pair<U, V>& p)
{
    return os << '{' << p.first << ": " << p.second << '}';
}

template<class Os, class K, class V>
Os& operator<<(Os& os, const std::unordered_map<K, V>& v)
{
    std::map<int, size_t> ordered(v.begin(), v.end());
    os << '[' << ordered.size() << "] { ";
    bool o{};
    for (const auto& e : ordered)
        os << (o ? ", " : (o = 1, "")) << e;
    return os << " }\n";
}

class TensorTracer
{
  public:
    explicit TensorTracer(NetworkParameters::CallbackPlace place, std::optional<std::regex> filter = std::nullopt, std::string filename = "data.trace")
        : mPlace{ place }
        , mLayer{ std::nullopt }
        , mRange{ -126, 127 }
        , mFileName{ std::move(filename) }
        , mFilter{ std::move(filter) }
    {
        write_header();
    }

    TensorTracer(NetworkParameters::CallbackPlace place, const raul::Name& layer, std::optional<std::regex> filter = std::nullopt, std::string filename = "data.trace")
        : mPlace{ place }
        , mLayer{ layer }
        , mRange{ -126, 127 }
        , mFileName{ std::move(filename) }
        , mFilter{ std::move(filter) }
    {
        write_header();
    }
    ~TensorTracer(){}
    void operator()(BasicLayer* layer, const MemoryManager& memory_manager, NetworkParameters::CallbackPlace place)
    {
        if (mPlace && *mPlace != place)
        {
            return;
        }
        if (mLayer && *mLayer != layer->getName())
        {
            return;
        }
        trace_tensors(layer, memory_manager, callbackPlaceToName(place));
    }
 
  private:
    template<typename T>
    int get_exp(T value)
    {
        int exp;
        std::frexp(static_cast<double>(value), &exp);
        return exp;
    }

    static std::string callbackPlaceToName(NetworkParameters::CallbackPlace place)
    {
        switch (place)
        {

            case NetworkParameters::CallbackPlace::Before_Forward:
                return "Before_Forward";
            case NetworkParameters::CallbackPlace::After_Forward:
                return "After_Forward";
            case NetworkParameters::CallbackPlace::Before_Backward:
                return "Before_Backward";
            case NetworkParameters::CallbackPlace::After_Backward:
                return "After_Backward";
            default:
                return "Unknown";
        }
    }

    static void insert_or_add(std::unordered_map<int, size_t>& dict, int key, size_t value = 1)
    {
        auto it = dict.find(key);
        if (it != dict.end())
        {
            it->second += value;
        }
        else
        {
            dict.insert(std::make_pair(key, value));
        }
    }

    auto get_exp_distr(raul::Tensor tensor)
    {
        std::unordered_map<int, size_t> dict;

        for (const auto& value : tensor)
        {
            const int exp = get_exp(value);
            insert_or_add(dict, exp);
        }
        return dict;
    }

    struct AccumulatingDict
    {
        void append(const std::unordered_map<int, size_t>& new_dict)
        {
            for (const auto& [key, value] : new_dict)
            {
                insert_or_add(dict, key, value);
            }
        }

        auto get_dict() const { return dict; }

      private:
        std::unordered_map<int, size_t> dict;
    };

    void trace_tensors(BasicLayer* layer, const MemoryManager& memory_manager, const std::string& placeName)
    {
        AccumulatingDict accumulator;
        const auto& tensors = memory_manager.getTensorCollection().tensors;

        for (const auto& [name, tensor] : tensors)
        {
            if (mFilter && !std::regex_match(name.str(), *mFilter))
            {
                continue;
            }
            const auto distr = get_exp_distr(*tensor);
            accumulator.append(distr);
        }
        write_record(layer->getName(), placeName, accumulator.get_dict());
    }

    void write_header()
    {
        std::fstream mFile(mFileName, std::fstream::out | std::fstream::app);
        if (mFile.is_open())
        {
            for (int i = mRange.first; i < mRange.second; ++i)
            {
                mFile << i << ",";
            }
            mFile << "layer,place" << std::endl;
        }
    }

    void write_record(const std::string& layer, const std::string& place, const std::unordered_map<int, size_t>& dict)
    {
        std::fstream mFile(mFileName, std::fstream::out | std::fstream::app);
        if (mFile.is_open())
        {
            for (int i = mRange.first; i < mRange.second; ++i)
            {
                const auto it = dict.find(i);
                const auto res = (it == dict.end()) ? 0 : it->second;
                mFile << res << ",";
            }
            mFile << layer << "," << place << std::endl;
        }
    }

    std::optional<NetworkParameters::CallbackPlace> mPlace;
    std::optional<raul::Name> mLayer;
    std::pair<int, int> mRange;
    std::string mFileName;
    std::optional<std::regex> mFilter;
};
} // UT::tools::callbacks

#endif // TENSORTRACER_H
