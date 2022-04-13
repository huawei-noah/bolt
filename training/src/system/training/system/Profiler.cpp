// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Profiler.h"

namespace raul
{

void Profiler::initialize(std::ostream* out, bool nesting, bool enable, bool dontWork, bool toJson)
{
    mLogFile = out;
    mNesting = nesting;
    mEnable = enable;
    mFullyDisabled = dontWork;
    mUseJsonFormat = toJson;
    // For JSON
    mPrevIndent = 0;
    mFirstWrite = true;
}

Profiler& Profiler::getInstance()
{
    static Profiler instance;
    return instance;
}

bool Profiler::isDisabled() const
{
    return mFullyDisabled;
}

void Profiler::enableProfiler()
{
    mEnable = true;
}

void Profiler::disableProfiler()
{
    mEnable = false;
}

void Profiler::increasePrefix(const std::string& prefix)
{
    mPrefix.push_back(prefix);
}

void Profiler::decreasePrefix(size_t num)
{
    for (size_t i = 0; i < num; ++i)
    {
        mPrefix.pop_back();
    }
}

std::string Profiler::generatePrefix() const
{
    std::string prefix;
    for (size_t i = 0; i < mPrefix.size() - 1; ++i)
    {
        prefix += mPrefix[i] + ":";
    }
    prefix += mPrefix.back();
    return prefix;
}

void Profiler::clearPrefix()
{
    mPrefix.clear();
}

void Profiler::tic(const std::string& opName)
{
    // Return if disabled
    if (!mEnable)
    {
        return;
    }

    if (mNesting)
    {
        mOpOrder.emplace_back(mOpStartPoints.size(), opName);
    }
    mOpStartPoints.push(std::chrono::steady_clock::now());
}

void Profiler::toc()
{
    // Nothing to print
    if (mOpStartPoints.empty())
    {
        return;
    }
    // Fix end time
    const auto endTime = std::chrono::steady_clock::now();
    // Get top elements
    const auto startTime = mOpStartPoints.top();
    // Pop these elements
    mOpStartPoints.pop();

    // Find duration
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    if (mNesting)
    {
        mOpDurations.emplace_back(mOpStartPoints.size(), duration);

        // Log
        if (mOpStartPoints.empty() && mLogFile != nullptr)
        {
            std::string prefix = generatePrefix();
            mPrevIndent = mPrefix.size() - 1;
            for (size_t i = 0; i < mOpOrder.size(); ++i)
            {
                for (size_t j = 0; j < mOpDurations.size(); ++j)
                {
                    if (mOpOrder[i].first == mOpDurations[j].first)
                    {
                        if (mUseJsonFormat)
                        {
                            // Prefix
                            if (mOpOrder[i].first > mPrevIndent)
                            {
                                increasePrefix(mOpOrder[i - 1].second);
                                prefix += ":" + mPrefix.back();
                            }
                            else if (mOpOrder[i].first < mPrevIndent)
                            {
                                decreasePrefix(mPrevIndent - mOpOrder[i].first);
                                prefix = generatePrefix();
                            }
                            mPrevIndent = mOpOrder[i].first;
                            // Begin
                            if (!mFirstWrite)
                            {
                                *mLogFile << "},\n";
                            }
                            else
                            {
                                mFirstWrite = false;
                            }
                            // Parse incoming name
                            std::string fullOpName = mOpOrder[i].second;
                            size_t pos1 = fullOpName.find("[");
                            size_t pos2 = fullOpName.rfind("::");
                            std::string layerType = fullOpName.substr(0, pos1);
                            std::string layerName = fullOpName.substr(pos1 + 1, pos2 - pos1 - 1);
                            std::string funcName = fullOpName.substr(pos2 + 2, fullOpName.length() - pos2 - 3);
                            *mLogFile << "{\n";
                            *mLogFile << "    \"cat\": \"PROFILING\",\n";
                            *mLogFile << "    \"pid\": 0,\n";
                            *mLogFile << "    \"tid\": 0,\n";
                            *mLogFile << "    \"ts\": 0,\n";
                            *mLogFile << "    \"ph\": \"B\",\n";
                            *mLogFile << "    \"name\": \"" + fullOpName << "\",\n";
                            *mLogFile << "    \"args\": { \"layer type\": \"" + layerType + "\", ";
                            *mLogFile << "\"layer name\": \"" + layerName + "\", \"function\": \"" + funcName + "\", ";
                            *mLogFile << "\"nesting order\": \"" + prefix + "\" }\n";
                            *mLogFile << "},\n";

                            // End
                            *mLogFile << "{\n";
                            *mLogFile << "    \"cat\": \"PROFILING\",\n";
                            *mLogFile << "    \"pid\": 0,\n";
                            *mLogFile << "    \"tid\": 0,\n";
                            *mLogFile << "    \"ts\": " << mOpDurations[j].second.count() << ",\n";
                            *mLogFile << "    \"ph\": \"E\",\n";
                            *mLogFile << "    \"name\": \"" + fullOpName << "\",\n";
                            *mLogFile << "    \"args\": { \"layer type\": \"" + layerType + "\", ";
                            *mLogFile << "\"layer name\": \"" + layerName + "\", \"function\": \"" + funcName + "\", ";
                            *mLogFile << "\"nesting order\": \"" + prefix + "\" }\n";
                        }
                        else
                        {
                            *mLogFile << std::string((mOpOrder[i].first + 1) * 4, ' ') << mOpOrder[i].second << ": " << mOpDurations[j].second.count() << "us\n";
                            mOpDurations.erase(mOpDurations.begin() + j);
                        }
                        break;
                    }
                }
            }
            mOpDurations.clear();
            mOpOrder.clear();
        }
    }
    else
    {
        if (mLogFile != nullptr)
        {
            *mLogFile << mOpOrder.back().second << ": " << duration.count() << "us\n";
            mOpOrder.pop_back();
        }
    }
}

} // namespace raul
