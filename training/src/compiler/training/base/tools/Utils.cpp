// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Utils.h"
#include <fstream>
#include <iostream>
#include <training/compiler/Workflow.h>

namespace raul::utils
{

template<typename MM>
void printTensorNamesImpl(std::ostream& stream, const raul::Name& name, const Names& a, const MM& memoryManager)
{
    if (!a.empty())
    {
        stream << "\t" << name << ":";
        for (const auto& s : a)
        {
            stream << " " << memoryManager.getTensor(s);
        }
        stream << std::endl;
    }
}

void printTensorNames(std::ostream& stream, const raul::Name& name, const Names& a, const MemoryManager& memoryManager)
{
    printTensorNamesImpl(stream, name, a, memoryManager);
}

void printTensorNames(std::ostream& stream, const raul::Name& name, const Names& a, const MemoryManagerFP16& memoryManager)
{
    printTensorNamesImpl(stream, name, a, memoryManager);
}

[[maybe_unused]] void printList(std::ostream& stream, const raul::Name& name, const Names& a)
{
    if (!a.empty())
    {
        stream << "\t" << name << ":";
        for (const auto& s : a)
        {
            stream << " " << s;
        }
        stream << std::endl;
    }
}

void traceTensor(const raul::Name& tensorName, const raul::NetworkParameters& networkParameters, const raul::Name& prefix)
{
    std::ofstream outfile;
    const auto& workflow = networkParameters.mWorkflow;
    const auto& target = workflow.getExecutionTarget();
    if (target == ExecutionTarget::CPU)
    {
        outfile.open(prefix + tensorName + ".cpu.trace", std::ios_base::app);
        const auto& memoryManager = workflow.getMemoryManager();
        const auto& tensor = memoryManager[tensorName];
        outfile << tensor.getDescription() << "[CPU]" << std::endl;
        for (const auto& val : tensor)
        {
            outfile << static_cast<double>(val) << std::endl;
        }
    }
    else if (target == ExecutionTarget::CPUFP16)
    {
        outfile.open(prefix + tensorName + ".cpu.fp16.trace", std::ios_base::app);
        const auto& memoryManager = workflow.getMemoryManager<MemoryManagerFP16>();
        const auto& tensor = memoryManager[tensorName];
        outfile << tensor.getDescription() << "[CPU FP16]" << std::endl;
        for (const auto& val : tensor)
        {
            outfile << static_cast<double>(val) << std::endl;
        }
    }
    else
    {
        outfile << tensorName << "[unsupported target]" << std::endl;
    }
}

} // namespace raul::utils