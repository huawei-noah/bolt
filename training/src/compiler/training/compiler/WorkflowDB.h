// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef WORKFLOW_DB_H
#define WORKFLOW_DB_H

#include <training/system/Name.h>
#include <unordered_map>

#include "Workflow.h"

namespace raul
{

/**
 * @brief Class to store information of tensor`s declaration
 *
 * Each tensor declaration is bound to layer and execution stage (forward, backward).
 * This information is stored into plain 1D array of TensorUsage and structured 3D table tensor name vs layer name vs `usage`.
 * `Usage` is an std::vector of size == 2 to store index from TensorUsage array.
 *
 * For example if tensor A from layer L used only in backward pass then [A][L][0] == std::numeric_limits<size_t>::max(), [A][L][1] == `index`.
 */
class WorkflowDB
{
  public:
    struct TensorUsage
    {
        Name layerName;
        Name tensorName;

        WShape shape;          ///< set for whole workflow (should be same)
        Workflow::Usage usage; ///< set for each usage location (might be different)
        Workflow::Mode mode;   ///< set for each usage location (might be different)

        bool isOptimizeGraph; ///< set for whole workflow (should be same)
        bool isOptimizeMem;   ///< set for whole workflow (should be same)
        bool isTrainable;     ///< set for whole workflow (should be same)
        bool isZero;          ///< set for each usage location (might be different)
        bool isCompress;      ///< set for each usage location (might be different)

        LayerExecutionTarget layerExecutionTarget;
    };

    WorkflowDB()
        : mTensorsTable{}
        , mLayersTable{}
        , mTensorNeeded{}
    {
    }
    ~WorkflowDB(){}
    /**
     * Cell of the 3d table
     *
     */
    using DBCell = std::vector<size_t>;

    /**
     * 3d table of [tensor, layer, usage (forward | backward)]
     */
    using TLTable = std::unordered_map<Name, std::unordered_map<Name, DBCell>>;

    /**
     * Check whether provided tensor with context is in database
     *
     * @param tensorName name of a tensor
     * @param layerName layer with the tensor
     * @param usage Training stage where tensor is used
     * @return true/false
     */
    [[nodiscard]] bool isTensorExistsInTable(const Name& tensorName, const Name& layerName, Workflow::Usage usage) const;

    /**
     * Get table slice by key
     *
     * @param table target table
     * @param keyA search key
     * @return vector of names of the keys in the slice
     *
     * @tod(ck): is it really a slice?
     */
    [[nodiscard]] Names getSlice(const TLTable& table, const Name& keyA) const;

    /**
     * Get cell of the search table (pair of indexes for forward and backward)
     *
     * @param table target table
     * @param keyA the first search key
     * @param keyB the second search key
     * @return Cell
     */
    [[nodiscard]] DBCell getCell(const TLTable& table, const Name& keyA, const Name& keyB) const;

    /**
     * Return the first meet usage information of the tensor
     *
     * @param tensorName
     * @return
     */
    [[nodiscard]] TensorUsage findFirstTensor(const Name& tensorName) const;

    /**
     * Check whether provided cell empty or not
     *
     * Function checks index stored in dictionary. If it goes out of array length
     * it means cell is empty.
     *
     * @param cell
     * @param usage
     * @return true/false
     *
     */
    [[nodiscard]] bool isCellElementEmpty(const DBCell& cell, Workflow::Usage usage) const;

    /**
     * Check whether provided tensor declared in the database
     *
     * @param tensorName
     * @return true/false
     */
    [[nodiscard]] bool isTensorDeclared(const Name& tensorName) const;

    /**
     * Getter to search table by tensor
     * @return Search table (read-only)
     *
     */
    [[nodiscard]] const TLTable& getTensorsTable() const { return mTensorsTable; }

    /**
     * Getter to search table by tensor
     * @return Search table
     *
     */
    TLTable& getTensorsTable() { return mTensorsTable; }

    /**
     * Getter to search table by layer
     * @return Search table (read-only)
     *
     */
    [[nodiscard]] const TLTable& getLayersTable() const { return mLayersTable; }

    /**
     * Getter to search table by layer
     * @return Search table
     *
     */
    TLTable& getLayersTable() { return mLayersTable; }

    /**
     * Getter to usage
     *
     * @param index Plain index
     * @return Usage (read-only)
     *
     */
    [[nodiscard]] const TensorUsage& getUsage(size_t index) const;

    /**
     * Getter to usage
     *
     * @param index Plain index
     * @return Usage
     */
    TensorUsage& getUsage(size_t index);

    /**
     * Add usage information to the workflow database
     *
     * @param usage Usage
     */
    void addUsage(const TensorUsage& usage)
    {
        mTensorNeeded.push_back(usage);
        addTensorToTable(usage, mTensorsTable, usage.tensorName, usage.layerName);
        addTensorToTable(usage, mLayersTable, usage.layerName, usage.tensorName);
    }

    void chooseMaxShape(const Name& tensorName, WShape& shape);

  private:
    /**
     * Add usage information of the tensor to a search table
     *
     * @param tensorUsage usage of the tensor
     * @param table quick access table with structure [Tensor, Layer, Usage] or [Layer, Tensor, Usage]
     * @param keyA the first key in table (Tensor or Layer)
     * @param keyB the second key in table (Tensor or Layer)
     */
    void addTensorToTable(const TensorUsage& tensorUsage, TLTable& table, const Name& keyA, const Name& keyB);
    void assignIndexes(const TensorUsage& tensorUsage, DBCell& indexes);

  private:
    TLTable mTensorsTable; ///< Search table: [Tensor, Layer, Usage] --> Index in storage aka mTensorNeeded
    TLTable mLayersTable;  ///< Search table: [Layer, Tensor, Usage] --> Index in storage aka mTensorNeeded

    std::vector<TensorUsage> mTensorNeeded; ///< Storage of all workflow tensors
};

} // raul namespace

#endif // WORKFLOW_DB_H
