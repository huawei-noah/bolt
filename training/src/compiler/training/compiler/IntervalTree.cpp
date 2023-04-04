// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "IntervalTree.h"

namespace raul
{

IntervalTree::IntervalTree(size_t totalNodes)
    : mRoot(nullptr)
    , mNodes(totalNodes)
    , mNodesCount(0)
{
}

IntervalTree::Node* IntervalTree::insert(IntervalTree::Node* root, const Interval* interval)
{
    IntervalTree::Node* ret;

    if (root == nullptr)
    {
        ret = newNode(interval);
    }
    else
    {
        size_t rootStart = root->interval->start;

        if (interval->start < rootStart)
        {
            root->left = insert(root->left, interval);
        }
        else
        {
            root->right = insert(root->right, interval);
        }

        root->max = std::max(root->max, interval->finish);

        ret = root;
    }

    return ret;
}

IntervalTree::Node* IntervalTree::newNode(const Interval* interval)
{
    if (mNodesCount == mNodes.size())
    {
        THROW_NONAME("IntervalTree", "no insertion possible");
    }

    IntervalTree::Node* ret = &mNodes[mNodesCount];

    ret->interval = interval;
    ret->left = nullptr;
    ret->right = nullptr;
    ret->max = interval->finish;

    ++mNodesCount;

    return ret;
}

void IntervalTree::insert(const Interval* interval)
{
    mRoot = insert(mRoot, interval);
}

bool IntervalTree::isOverlap(const Interval* intervalA, const Interval* intervalB) const
{
    if (intervalA->start <= intervalB->finish && intervalA->finish >= intervalB->start)
    {
        return true;
    }

    return false;
}

const IntervalTree::Interval* IntervalTree::find(const IntervalTree::Node* root, const Interval* interval) const
{
    const Interval* ret = nullptr;

    if (root != nullptr)
    {
        if (isOverlap(root->interval, interval))
        {
            ret = root->interval;
        }
        else
        {
            if (root->left != nullptr && root->left->max >= interval->start)
            {
                ret = find(root->left, interval);
            }
            else
            {
                ret = find(root->right, interval);
            }
        }
    }

    return ret;
}

const IntervalTree::Interval* IntervalTree::find(const Interval* interval) const
{
    return find(mRoot, interval);
}

void IntervalTree::findAll(const IntervalTree::Node* root, const Interval* interval, std::vector<const Interval*>& res) const
{
    if (root != nullptr)
    {
        if (root->left != nullptr && root->left->max >= interval->start)
        {
            findAll(root->left, interval, res);
        }

        if (root->right != nullptr && root->right->max >= interval->start)
        {
            findAll(root->right, interval, res);
        }

        if (isOverlap(root->interval, interval))
        {
            res.push_back(root->interval);
        }
    }
}

std::vector<const IntervalTree::Interval*> IntervalTree::findAll(const Interval* interval) const
{
    std::vector<const Interval*> ret;

    findAll(mRoot, interval, ret);

    return ret;
}
} // namespace raul
