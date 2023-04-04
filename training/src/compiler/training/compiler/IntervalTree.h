// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef INTERVALTREE_H
#define INTERVALTREE_H

#include "WorkflowPool.h"

namespace raul
{
/**
 * @brief Class to calculate intervals overlap
 *
 */
class IntervalTree
{
  public:
    IntervalTree(size_t totalNodes);
    ~IntervalTree(){}
    typedef WorkflowPool<MemoryManager>::Interval Interval;

    void insert(const Interval* interval);

    const Interval* find(const Interval* interval) const;
    std::vector<const Interval*> findAll(const Interval* interval) const;

  private:
    struct Node
    {
        const Interval* interval;
        size_t max;
        Node *left, *right;
    };

    Node* insert(Node* root, const Interval* interval);

    Node* newNode(const Interval* interval);

    bool isOverlap(const Interval* intervalA, const Interval* intervalB) const;

    const Interval* find(const Node* root, const Interval* interval) const;
    void findAll(const Node* root, const Interval* interval, std::vector<const Interval*>& res) const;

    Node* mRoot;

    std::vector<Node> mNodes;
    size_t mNodesCount;
};
} // raul namespace

#endif
