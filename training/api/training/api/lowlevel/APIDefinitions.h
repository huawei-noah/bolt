// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef API_DEFINITIONS_H
#define API_DEFINITIONS_H

#include <memory>

#include <training/compiler/Workflow.h>
#include <training/compiler/WorkflowEager.h>

#ifdef __cplusplus
extern "C"
{
#endif

    struct Graph_Description_t
    {
        Graph_Description_t()
            : mDef(std::make_shared<raul::Workflow>())
        {
        }

        Graph_Description_t(raul::ExecutionTarget target)
            : mDef(std::make_shared<raul::Workflow>(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, target))
        {
        }

        Graph_Description_t(bool)
            : mDef(std::make_shared<raul::WorkflowEager>())
        {
        }

        Graph_Description_t(raul::ExecutionTarget target, bool enable_compiler)
            : mDef(std::make_shared<raul::Workflow>(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, target, enable_compiler))
        {
        }

        std::shared_ptr<raul::Workflow> mDef;
    };

#ifdef __cplusplus
}
#endif

#endif