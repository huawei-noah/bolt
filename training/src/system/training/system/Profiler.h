// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <stack>
#include <string>
#include <vector>

#include <training/system/Name.h>

namespace raul
{

class Profiler
{

  public:
    // Restrict copy semantic
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    // Get instance
    static Profiler& getInstance();

    // Initialize
    void initialize(std::ostream* out, bool nesting = true, bool enable = false, bool dontWork = true, bool toJson = false);

    // Main functions
    void tic(const std::string& opName);
    void toc();

    // Is globally disabled
    bool isDisabled() const;

    // Enable or disable
    void enableProfiler();
    void disableProfiler();

    // Get internal ostream
    std::ostream* getOstream() const { return mLogFile; }

    // Get state
    bool getState() const { return mEnable; }

    // Is JSON
    bool useJsonFormat() const { return mUseJsonFormat; }

    // Create or add/delete part of a prefix
    std::string generatePrefix() const;
    void clearPrefix();
    void increasePrefix(const std::string& prefix);
    void decreasePrefix(size_t numToRemove);

  private:
    Profiler() {}
    std::ostream* mLogFile{ nullptr };
    bool mNesting{ false };
    // Use to follow nesting and print
    std::stack<std::chrono::steady_clock::time_point> mOpStartPoints;
    std::vector<std::pair<size_t, std::chrono::microseconds>> mOpDurations;
    std::vector<std::pair<size_t, std::string>> mOpOrder;
    // On or off
    bool mEnable{ false };

    // Global off
    bool mFullyDisabled{ true };

    // For JSON
    size_t mPrevIndent;
    bool mUseJsonFormat;
    std::vector<std::string> mPrefix;
    bool mFirstWrite;
};

namespace helpers
{

class ProfilerMeasurer
{
  public:
    ProfilerMeasurer(const raul::Name& name) { raul::Profiler::getInstance().tic(name); }
    ~ProfilerMeasurer() { raul::Profiler::getInstance().toc(); }
};

} // namespace helpers

#define PROFILER_TIC(name) raul::Profiler::getInstance().tic(name);
#define PROFILER_TOC raul::Profiler::getInstance().toc();

#define MEASURE_BLOCK(name) raul::helpers::ProfilerMeasurer instance{ name };
} // namespace raul

#endif
