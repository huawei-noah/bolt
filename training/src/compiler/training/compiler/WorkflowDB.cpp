// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "WorkflowDB.h"

namespace raul
{

bool WorkflowDB::isTensorExistsInTable(const Name& tensorName, const Name& layerName, Workflow::Usage usage) const
{
    bool ret = false;

    TLTable::const_iterator i = mTensorsTable.find(tensorName);
    if (i != mTensorsTable.end())
    {
        std::unordered_map<Name, DBCell>::const_iterator j = (*i).second.find(layerName);
        if (j != (*i).second.end())
        {
            DBCell indexes = (*j).second;

            if (usage == Workflow::Usage::Forward)
            {
                if (!isCellElementEmpty(indexes, Workflow::Usage::Forward)) ret = true;
            }

            if (usage == Workflow::Usage::Backward)
            {
                if (!isCellElementEmpty(indexes, Workflow::Usage::Backward)) ret = true;
            }

            if (usage == Workflow::Usage::ForwardAndBackward)
            {
                if (!isCellElementEmpty(indexes, Workflow::Usage::Forward) || !isCellElementEmpty(indexes, Workflow::Usage::Backward)) ret = true;
            }
        }
    }

    return ret;
}

void WorkflowDB::addTensorToTable(const TensorUsage& tensorUsage, TLTable& table, const Name& keyA, const Name& keyB)
{
    auto i = table.find(keyA);
    if (i != table.end())
    {
        auto j = (*i).second.find(keyB);
        if (j != (*i).second.end())
        {
            assignIndexes(tensorUsage, (*j).second);
        }
        else
        {
            auto k = (*i).second.insert({ keyB, { std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max() } });
            assignIndexes(tensorUsage, (*k.first).second);
        }
    }
    else
    {
        auto j = table.insert({ keyA, {} });
        auto k = (*j.first).second.insert({ keyB, { std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max() } });
        assignIndexes(tensorUsage, (*k.first).second);
    }
}

void WorkflowDB::chooseMaxShape(const Name& tensorName, WShape& shape)
{
    TLTable::const_iterator i = mTensorsTable.find(tensorName);
    if (i != mTensorsTable.end())
    {
        for (auto j = (*i).second.begin(); j != (*i).second.end(); ++j)
        {
            std::vector<size_t> tensorUsage = (*j).second;

            if (!isCellElementEmpty(tensorUsage, Workflow::Usage::Forward))
            {
                TensorUsage& usg = mTensorNeeded[tensorUsage[static_cast<size_t>(Workflow::Usage::Forward)]];
                usg.shape.selectMaxShape(shape);
            }

            if (!isCellElementEmpty(tensorUsage, Workflow::Usage::Backward))
            {
                TensorUsage& usg = mTensorNeeded[tensorUsage[static_cast<size_t>(Workflow::Usage::Backward)]];
                usg.shape.selectMaxShape(shape);
            }
        }
    }
}

void WorkflowDB::assignIndexes(const TensorUsage& tensorUsage, DBCell& indexes)
{
    if (tensorUsage.usage == Workflow::Usage::Forward)
    {
        indexes[static_cast<size_t>(Workflow::Usage::Forward)] = mTensorNeeded.size() - 1;
    }

    if (tensorUsage.usage == Workflow::Usage::Backward)
    {
        indexes[static_cast<size_t>(Workflow::Usage::Backward)] = mTensorNeeded.size() - 1;
    }

    if (tensorUsage.usage == Workflow::Usage::ForwardAndBackward)
    {
        indexes[static_cast<size_t>(Workflow::Usage::Forward)] = mTensorNeeded.size() - 1;
        indexes[static_cast<size_t>(Workflow::Usage::Backward)] = mTensorNeeded.size() - 1;
    }
}

std::vector<Name> WorkflowDB::getSlice(const TLTable& table, const Name& keyA) const
{
    std::vector<Name> ret;

    auto i = table.find(keyA);
    if (i != table.end())
    {
        for (auto j = (*i).second.begin(); j != (*i).second.end(); ++j)
        {
            ret.push_back((*j).first);
        }
    }

    return ret;
}

WorkflowDB::DBCell WorkflowDB::getCell(const TLTable& table, const Name& keyA, const Name& keyB) const
{
    DBCell ret;

    auto i = table.find(keyA);
    if (i != table.end())
    {
        auto j = (*i).second.find(keyB);
        if (j != (*i).second.end())
        {
            ret = (*j).second;
        }
    }

    return ret;
}

WorkflowDB::TensorUsage WorkflowDB::findFirstTensor(const Name& tensorName) const
{
    TensorUsage usg;
    bool found = false;

    TLTable::const_iterator i = mTensorsTable.find(tensorName);
    if (i != mTensorsTable.end())
    {
        if (!(*i).second.empty())
        {
            found = true;

            DBCell tensorUsage = (*(*i).second.begin()).second;
            if (!isCellElementEmpty(tensorUsage, Workflow::Usage::Forward))
            {
                const auto stageIdx = static_cast<size_t>(Workflow::Usage::Forward);
                const auto usgIdx = tensorUsage[stageIdx];
                usg = mTensorNeeded[usgIdx];
            }
            else if (!isCellElementEmpty(tensorUsage, Workflow::Usage::Backward))
            {
                const auto stageIdx = static_cast<size_t>(Workflow::Usage::Backward);
                const auto usgIdx = tensorUsage[stageIdx];
                usg = mTensorNeeded[usgIdx];
            }
        }
    }

    if (!found)
    {
        THROW_NONAME("WorkflowDB", "tensor [" + tensorName + "] hasn`t been declared");
    }

    return usg;
}

bool WorkflowDB::isCellElementEmpty(const DBCell& cell, Workflow::Usage usage) const
{
    bool ret = true;

#ifdef _DEBUG
    if (usage != Workflow::Usage::Forward && usage != Workflow::Usage::Backward)
    {
        THROW_NONAME("WorkflowDB", "incorrect parameter");
    }
#endif

    if (!cell.empty())
    {
        ret = !(cell[static_cast<size_t>(usage)] < mTensorNeeded.size());
    }

    return ret;
}

bool WorkflowDB::isTensorDeclared(const Name& tensorName) const
{
    bool ret = false;

    TLTable::const_iterator i = mTensorsTable.find(tensorName);
    if (i != mTensorsTable.end())
    {
        ret = true;
    }

    return ret;
}

const WorkflowDB::TensorUsage& WorkflowDB::getUsage(size_t index) const
{
    return mTensorNeeded.at(index);
}

WorkflowDB::TensorUsage& WorkflowDB::getUsage(size_t index)
{
    return mTensorNeeded.at(index);
}

} // namespace raul