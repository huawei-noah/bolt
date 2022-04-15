// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <gtest/gtest.h>

#include <map>

#include <training/api/API.h>
#include <training/common/Train.h>
#include <training/layers/parameters/trainable/TacotronParams.h>
#include <training/optimizers/schedulers/LrScheduler.h>
#include <training/tools/Dataset.h>

#include "TacotronTrainingParams.h"

namespace UT
{
using namespace raul;
using namespace std;

dtype TensorDiff(const Tensor& a, const Tensor& b);
dtype TensorNorm(const Tensor& a);

map<string, float> loadNamedValues(const filesystem::path& path, const string& name, const TacotronParams& params, bool spbModel = false);

void tacotronParamNamesMaps(const Name& name, const TacotronParams& params, map<Name, string>& oplain, map<Name, string>& otransposed, bool loadOptimizerParams = true, bool spbModel = false);

template<typename MM>
void createAdamTensors(const Names& params, MM& m);

size_t loadTacotronParams(const string& pathPrefix,
                          MemoryManagerGPU& m,
                          const Name& name,
                          const TacotronParams& params,
                          bool allParamsShouldExist = true,
                          bool loadOptimizerParams = true,
                          bool spbModel = false);

template<typename MM>
size_t loadTacotronParams(const string& pathPrefix, MM& m, const Name& name, const TacotronParams& params, bool allParamsShouldExist = true, bool loadOptimizerParams = true, bool spbModel = false);

template<typename T = Tensor>
bool loadTFData(const filesystem::path& p, const vector<T*>& tensors);

bool loadTFData(const filesystem::path& p, MemoryManager& m, const Names& tensors);
bool loadTFData(const filesystem::path& p, MemoryManagerGPU& m, const Names& tensors);

template<typename T>
bool loadTFData(const filesystem::path& p, T& tensor);

bool loadTFData(const filesystem::path& p, TensorGPUHelper tensor);

size_t saveTacotronParams(const filesystem::path& dir, const string& prefix, MemoryManager& m, const Name& name, const TacotronParams& params, bool withAdam = true, bool spbModel = false);
size_t saveTacotronParams(const filesystem::path& dir, const string& prefix, MemoryManagerGPU& m, const Name& name, const TacotronParams& params, bool withAdam = true, bool spbModel = false);

// returns trainable and all params count
pair<size_t, size_t> tacotronParamsCount(const TacotronParams& params);

unique_ptr<optimizers::Scheduler::LrScheduler> createOptimizer(const TacotronTrainingParams& trainParams, dtype ADAM_BETA1, dtype ADAM_BETA2, dtype ADAM_EPSILON);

}
