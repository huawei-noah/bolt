// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TESTWORKFLOWTOOLS_H
#define TESTWORKFLOWTOOLS_H

namespace
{

class TestInitLayer : public raul::BasicLayer
{
  public:
    TestInitLayer(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "TestInit", params, networkParameters, { false, false })
        , mBS(0)
    {
    }
    ~TestInitLayer(){}
    void onBatchSizeChanged(size_t size) override { mBS = size; }

    void forwardComputeImpl(raul::NetworkMode) override {}
    void backwardComputeImpl() override {}

    size_t getBatchSize() const { return mBS; }

  private:
    size_t mBS;
};

class TestLayer : public raul::BasicLayer
{
  public:
    TestLayer(const raul::Name& name, const raul::BasicParams& params, bool performGradChecks, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "test", params, networkParameters, { false, false })
        , mForwardCountTest(0)
        , mForwardCountTrain(0)
        , mBackwardCount(0)
        , mExpectGrad(0.0_dt)
        , mPerformGradChecks(performGradChecks)
        , mPerformGradWeightsChecks(true)
        , mPerformSizeChecks(true)
    {
        for (auto& input : params.getInputs())
        {
            mNetworkParams.mWorkflow.tensorNeeded(
                name, input, raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false);

            mNetworkParams.mWorkflow.tensorNeeded(
                name, input.grad(), raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Write, true, true, false, true, false);
        }

        for (auto& output : params.getOutputs())
        {
            mNetworkParams.mWorkflow.tensorNeeded(name, output, raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Write, true, true, false, false, false);

            mNetworkParams.mWorkflow.tensorNeeded(
                name, output.grad(), raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, true, true, false, false, false);
        }

        if (mSharedWeights.empty())
        {
            mNetworkParams.mWorkflow.tensorNeeded(
                name, name / "Weights", raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, true, false, false);

            mNetworkParams.mWorkflow.tensorNeeded(
                name, (name / "Weights").grad(), raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Write, false, false, false, true, false);
        }
        else
        {
            for (auto& weights : params.getSharedWeights())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, weights, raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, true, false, false);

                mNetworkParams.mWorkflow.tensorNeeded(
                    name, weights.grad(), raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Write, false, false, false, true, false);
            }
        }
    }

    void forwardComputeImpl(raul::NetworkMode mode) override
    {
        if (mode == raul::NetworkMode::Test)
        {
            ++mForwardCountTest;
        }

        if (mode == raul::NetworkMode::Train)
        {
            ++mForwardCountTrain;
        }

        for (auto& input : mInputs)
        {
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(input));
            ASSERT_EQ(mNetworkParams.mMemoryManager[input].size(), 1u * mNetworkParams.mWorkflow.getBatchSize());

            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(input.grad()));

            if (mPerformSizeChecks)
            {
                ASSERT_EQ(mNetworkParams.mMemoryManager[input.grad()].size(), 0u);
            }
        }

        for (auto& output : mOutputs)
        {
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(output));
            ASSERT_EQ(mNetworkParams.mMemoryManager[output].size(), 1u * mNetworkParams.mWorkflow.getBatchSize());

            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(output.grad()));

            if (mPerformSizeChecks)
            {
                ASSERT_EQ(mNetworkParams.mMemoryManager[output.grad()].size(), 0u);
            }
        }

        if (mSharedWeights.empty())
        {
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(mName / "Weights"));
            ASSERT_EQ(mNetworkParams.mMemoryManager[mName / "Weights"].size(), 1u);

            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists((mName / "Weights").grad()));
            ASSERT_EQ(mNetworkParams.mMemoryManager[(mName / "Weights").grad()].size(), 1u);
        }
        else
        {
            for (auto& weights : mSharedWeights)
            {
                ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(weights));
                ASSERT_EQ(mNetworkParams.mMemoryManager[weights].size(), 1u);

                ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(weights.grad()));
                ASSERT_EQ(mNetworkParams.mMemoryManager[weights.grad()].size(), 1u);
            }
        }
    }

    void backwardComputeImpl() override
    {
        ++mBackwardCount;

        for (auto& input : mInputs)
        {
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(input));

            if (mPerformSizeChecks)
            {
                ASSERT_EQ(mNetworkParams.mMemoryManager[input].size(), 1u * mNetworkParams.mWorkflow.getBatchSize());
            }

            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(input.grad()));

            if (mPerformSizeChecks)
            {
                ASSERT_EQ(mNetworkParams.mMemoryManager[input.grad()].size(), 1u * mNetworkParams.mWorkflow.getBatchSize());
            }
        }

        for (auto& output : mOutputs)
        {
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(output));

            if (mPerformSizeChecks)
            {
                ASSERT_EQ(mNetworkParams.mMemoryManager[output].size(), 0u * mNetworkParams.mWorkflow.getBatchSize());
            }

            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(output.grad()));

            if (mPerformSizeChecks)
            {
                ASSERT_EQ(mNetworkParams.mMemoryManager[output.grad()].size(), 1u * mNetworkParams.mWorkflow.getBatchSize());
            }
        }

        if (mSharedWeights.empty())
        {
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(mName / "Weights"));
            ASSERT_EQ(mNetworkParams.mMemoryManager[mName / "Weights"].size(), 1u);

            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists((mName / "Weights").grad()));
            ASSERT_EQ(mNetworkParams.mMemoryManager[(mName / "Weights").grad()].size(), 1u);

            auto& gradients = mNetworkParams.mMemoryManager[(mName / "Weights").grad()];

            for (auto& grad : gradients)
            {
                if (mPerformGradWeightsChecks)
                {
                    ASSERT_EQ(grad, 0.0_dt);
                }
                grad += 1.0_dt;
            }

            for (auto& grad : gradients)
            {
                if (mPerformGradWeightsChecks)
                {
                    ASSERT_EQ(grad, 1.0_dt);
                }
            }
        }
        else
        {
            for (auto& weights : mSharedWeights)
            {
                ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(weights));
                ASSERT_EQ(mNetworkParams.mMemoryManager[weights].size(), 1u);

                ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(weights.grad()));
                ASSERT_EQ(mNetworkParams.mMemoryManager[weights.grad()].size(), 1u);

                auto& gradients = mNetworkParams.mMemoryManager[weights.grad()];

                for (auto& grad : gradients)
                {
                    grad += 1.0_dt;
                }
            }

            auto& gradients = mNetworkParams.mMemoryManager[mSharedWeights[0].grad()];
            for (auto& grad : gradients)
            {
                ASSERT_EQ(grad, TODTYPE(mSharedWeights.size()));
            }
        }

        for (size_t q = 0; q < mOutputs.size(); ++q)
        {
            const auto& delta = mNetworkParams.mMemoryManager[mOutputs[q].grad()];

            for (size_t w = 0; w < mInputs.size(); ++w)
            {
                auto& prevLayerDelta = mNetworkParams.mMemoryManager[mInputs[w].grad()];

                if (prevLayerDelta.size() == delta.size())
                {
                    for (size_t e = 0; e < delta.size(); ++e)
                    {
                        prevLayerDelta[e] += delta[e] + 1.0_dt;
                    }
                }
            }
        }

        if (!mInputs.empty())
        {
            auto& prevLayerDelta = mNetworkParams.mMemoryManager[mInputs[0].grad()];
            for (size_t q = 0; q < prevLayerDelta.size(); ++q)
            {
                if (mPerformGradChecks)
                {
                    ASSERT_EQ(prevLayerDelta[q], mExpectGrad);
                }
            }
        }
    }

    size_t getForwardCountTest() const { return mForwardCountTest; }
    size_t getForwardCountTrain() const { return mForwardCountTrain; }
    size_t getBackwardCount() const { return mBackwardCount; }

    void setExpectGrad(raul::dtype expectGrad) { mExpectGrad = expectGrad; }

    void setPerformGradWeightsChecks(bool param) { mPerformGradWeightsChecks = param; }
    void setPerformSizeChecks(bool param) { mPerformSizeChecks = param; }

  private:
    size_t mForwardCountTest;
    size_t mForwardCountTrain;
    size_t mBackwardCount;

    raul::dtype mExpectGrad;

    bool mPerformGradChecks;
    bool mPerformGradWeightsChecks;
    bool mPerformSizeChecks;
};

[[maybe_unused]] const raul::Name getName(const raul::Workflow::Action* action)
{
    if (dynamic_cast<const raul::TensorAction<raul::MemoryManager>*>(action))
    {
        return static_cast<const raul::TensorAction<raul::MemoryManager>*>(action)->mName;
    }
    if (dynamic_cast<const raul::LayerAction*>(action))
    {
        return static_cast<const raul::LayerAction*>(action)->mLayer->getName();
    }

    return "";
}

[[maybe_unused]] bool checkName(const raul::Workflow::Action* action, const raul::Name& name)
{
    return getName(action) == name;
}

[[maybe_unused]] bool checkGroupedName(const raul::Names& names, const raul::Names& check)
{
    std::unordered_set<raul::Name> tNames;

    for (const auto& tName : names)
    {
        tNames.insert(tName);
    }

    for (const auto& tName : check)
    {
        if (tNames.find(tName) == tNames.end())
        {
            return false;
        }
    }

    return true;
}

[[maybe_unused]] raul::Name getGradName(const raul::Name& name)
{
    return name.grad();
}

[[maybe_unused]] bool checkBlock(const raul::Names& plain, size_t offset, const raul::Names& block)
{
    bool ret = true;

    if (block.size() + offset > plain.size()) return false;

    for (size_t q = 0; q < block.size(); ++q)
    {
        if (block[q] != plain[q + offset])
        {
            ret = false;
            break;
        }
    }

    return ret;
}

[[maybe_unused]] void permutate(std::vector<size_t>& indexes, size_t index, std::vector<std::vector<size_t>>& order)
{
    if (index == indexes.size())
    {
        order.push_back(indexes);
        return;
    }

    for (size_t q = index; q < indexes.size(); ++q)
    {
        std::swap(indexes[index], indexes[q]);
        permutate(indexes, index + 1, order);
        std::swap(indexes[index], indexes[q]);
    }
}

[[maybe_unused]] bool checkBlocks(const raul::Names& plain, const std::vector<raul::Names>& blocks)
{
    size_t sumBlocksLen = 0;

    for (const auto& block : blocks)
    {
        sumBlocksLen += block.size();
    }

    if (sumBlocksLen != plain.size())
    {
        return false;
    }

    std::vector<size_t> indexes(blocks.size());
    for (size_t q = 0; q < blocks.size(); ++q)
    {
        indexes[q] = q;
    }

    std::vector<std::vector<size_t>> order;

    permutate(indexes, 0, order);

    bool ret = false;

    for (size_t q = 0; q < order.size(); ++q)
    {
        bool found = true;
        size_t offset = 0;
        for (size_t w = 0; w < order[q].size(); ++w)
        {
            if (!checkBlock(plain, offset, blocks[order[q][w]]))
            {
                found = false;
                break;
            }
            offset += blocks[order[q][w]].size();
        }

        if (found)
        {
            ret = true;
            break;
        }
    }

    return ret;
}

[[maybe_unused]] bool checkBlocksName(const raul::Workflow::Pipeline& pipe, size_t indexStart, size_t indexFinish, const std::vector<raul::Names>& blocks)
{
    if (indexStart >= indexFinish)
    {
        return false;
    }

    if (indexStart >= pipe.size() || indexFinish >= pipe.size())
    {
        return false;
    }

    raul::Names plain;

    for (size_t q = indexStart; q <= indexFinish; ++q)
    {
        plain.push_back(getName(pipe[q].get()));
    }

    return checkBlocks(plain, blocks);
}

[[maybe_unused]] bool checkBlocksType(const raul::Workflow::Pipeline& pipe, size_t indexStart, size_t indexFinish, const std::vector<raul::Names>& blocks)
{
    if (indexStart >= indexFinish)
    {
        return false;
    }

    if (indexStart >= pipe.size() || indexFinish >= pipe.size())
    {
        return false;
    }

    raul::Names plain;

    for (size_t q = indexStart; q <= indexFinish; ++q)
    {
        plain.push_back(pipe[q]->type());
    }

    return checkBlocks(plain, blocks);
}

} // anonymous namespace

#endif
