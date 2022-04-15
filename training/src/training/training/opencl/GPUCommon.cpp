// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GPUCommon.h"
#include <training/common/OpenclInitializer.h>

namespace
{
using namespace raul;

cl::Kernel getReLUKernel(OpenCLKernelManager& manager, const Name& caller)
{
    Name name = "ReLUActivation";

    if (!manager.hasKernel(name, "activation"))
    {
        const std::string source =
#include <training/opencl/kernels/activation.cl>
            ;
        manager.registerProgram(name, source, "-DUSE_RELU");
    }

    return manager.getKernel(name, "activation", caller);
}

cl::Kernel getReLU6Kernel(OpenCLKernelManager& manager, const Name& caller)
{
    Name name = "ReLU6Activation";

    if (!manager.hasKernel(name, "activation"))
    {
        const std::string source =
#include <training/opencl/kernels/activation.cl>
            ;
        manager.registerProgram(name, source, "-DUSE_RELU6");
    }

    return manager.getKernel(name, "activation", caller);
}

cl::Kernel getReLUBackwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    Name name = "ReLUActivation";
    if (!manager.hasKernel(name, "activation_backward"))
    {
        const std::string source =
#include <training/opencl/kernels/activation.cl>
            ;
        manager.registerProgram(name, source, "-DUSE_RELU");
    }

    return manager.getKernel(name, "activation_backward", caller);
}

cl::Kernel getReLU6BackwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    Name name = "ReLU6Activation";
    if (!manager.hasKernel(name, "activation_backward"))
    {
        const std::string source =
#include <training/opencl/kernels/activation.cl>
            ;
        manager.registerProgram(name, source, "-DUSE_RELU6");
    }

    return manager.getKernel(name, "activation_backward", caller);
}

cl::Kernel getClampKernel(OpenCLKernelManager& manager, const Name& direction, const Name& caller)
{
    Name name = "Clamp";

    if (!manager.hasKernel(name, "clip" + direction))
    {
        const std::string source =
#include <training/opencl/kernels/clip.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, "clip" + direction, caller);
}

cl::NDRange getBatchNormWorkSize(raul::Dimension dim, size_t depth, size_t height, size_t width)
{
    switch (dim)
    {
        case raul::Dimension::Depth:
            return cl::NDRange{ depth, 1, 1 };
        case raul::Dimension::Height:
            return cl::NDRange{ height, 1, 1 };
        case raul::Dimension::Width:
            return cl::NDRange{ width, 1, 1 };
        default:
            THROW_NONAME("GPU", "getBatchNormWorkSize: unsupported dimension");
    }
}
cl::Kernel getBackwardBatchNormKernel(raul::OpenCLKernelManager& manager, raul::Dimension dim, bool frozen)
{
    string pname = "batchnorm";
    string name;
    switch (dim)
    {
        case raul::Dimension::Depth:
            name = "bn_backward_depth";
            break;
        case raul::Dimension::Height:
            name = "bn_backward_height";
            break;
        case raul::Dimension::Width:
            name = "bn_backward_width";
            break;
        default:
            THROW_NONAME("GPU", "getBackwardBatchNormKernel: unsupported dimension");
    }

    if (frozen)
    {
        name += "_frozen";
    }

    if (!manager.hasKernel(pname, name))
    {
        string source =
#include "kernels/batchnorm.cl"
            ;
        manager.registerProgram(pname, source);
    }

    return manager.getKernel(pname, name, "getBackwardBatchNormKernel");
}

cl::Kernel getForwardBatchNormKernel(raul::OpenCLKernelManager& manager, raul::Dimension dim)
{
    string pname = "batchnorm";
    string name;
    switch (dim)
    {
        case raul::Dimension::Depth:
            name = "bn_forward_depth";
            break;
        case raul::Dimension::Height:
            name = "bn_forward_height";
            break;
        case raul::Dimension::Width:
            name = "bn_forward_width";
            break;
        default:
            THROW_NONAME("GPU", "getForwardBatchNormKernel: unsupported dimension");
    }

    if (!manager.hasKernel(pname, name))
    {
        string source =
#include "kernels/batchnorm.cl"
            ;
        manager.registerProgram(pname, source);
    }

    return manager.getKernel(pname, name, "getForwardBatchNormKernel");
}

cl::Kernel getTestBatchNormKernel(raul::OpenCLKernelManager& manager, raul::Dimension dim)
{
    string pname = "batchnorm";
    string name;
    switch (dim)
    {
        case raul::Dimension::Depth:
            name = "bn_test_depth";
            break;
        case raul::Dimension::Height:
            name = "bn_test_height";
            break;
        case raul::Dimension::Width:
            name = "bn_test_width";
            break;
        default:
            THROW_NONAME("GPU", "getTestBatchNormKernel: unsupported dimension");
    }

    if (!manager.hasKernel(pname, name))
    {
        string source =
#include "kernels/batchnorm.cl"
            ;
        manager.registerProgram(pname, source);
    }

    return manager.getKernel(pname, name, "getTestBatchNormKernel");
}

cl::Kernel getSwap_ijk2kjiKernel(OpenCLKernelManager& manager, const Name& caller)
{
    if (!manager.hasKernel("swap_indices", "swap_ijk2kji"))
    {
        const std::string source =
#include <training/opencl/kernels/swap_indices.cl>
            ;
        manager.registerProgram("swap_indices", source);
    }

    return manager.getKernel("swap_indices", "swap_ijk2kji", caller);
}

cl::Kernel getSliceKernel(OpenCLKernelManager& manager, const size_t axisNum, const size_t numOfSlices, const Name& caller)
{
    const Name name = "slice" / std::to_string(axisNum) + std::to_string(numOfSlices);
    const Name kernelName = "slice_nchw_" + std::to_string(axisNum) + std::to_string(numOfSlices);

    if (!manager.hasKernel(name, kernelName))
    {
        const std::string source =
#include <training/opencl/kernels/slice.cl>
            ;
        manager.registerProgram(name, source, "-DUSE_NCHW -DAXIS_NUM=" + std::to_string(axisNum) + " -DON=" + std::to_string(numOfSlices));
    }

    return manager.getKernel(name, kernelName, caller);
}

cl::Kernel getConcatKernel(OpenCLKernelManager& manager, const size_t axisNum, const size_t numOfSlices, const Name& caller)
{
    const Name name = "concat" / std::to_string(axisNum) + std::to_string(numOfSlices);

    Name kernelName = "concat_nchw_";
    Name options = "-DN=" + std::to_string(numOfSlices);
    switch (axisNum)
    {
        case 0:
            kernelName += "w_";
            options += " -DAXIS_W";
            break;
        case 1:
            kernelName += "h_";
            options += " -DAXIS_H";
            break;
        case 2:
            kernelName += "c_";
            options += " -DAXIS_C";
            break;
        default:
            THROW_NONAME("GPU", "getConcatKernel: unsupported dim");
    }
    kernelName += std::to_string(numOfSlices);

    if (!manager.hasKernel(name, kernelName))
    {
        const std::string source =
#include <training/opencl/kernels/concat_nchw.cl>
            ;
        manager.registerProgram(name, source, options);
    }

    return manager.getKernel(name, kernelName, caller);
}

cl::Kernel getIotaKernel(OpenCLKernelManager& manager, const Name& caller)
{
    Name name = "iota";

    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/iota.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

void callEltwiseKernel(OpenCLKernelManager& manager,
                       cl::Kernel kernel,
                       const Name& caller,
                       size_t inputNum,
                       size_t batch,
                       size_t depth,
                       size_t height,
                       size_t width,
                       const std::array<cl::Buffer, 4> in,
                       cl::Buffer& out)
{
    switch (inputNum)
    {
        case 1:
            manager.callKernel(kernel,
                               cl::NDRange{ (width + 3) / 4, height, depth * batch },
                               caller,
                               (cl_int)height,
                               (cl_int)width,
                               (cl_int)(depth * batch),
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               (cl_int)((width + 3) / 4),
                               (cl_int)height,
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[0],
                               out);
            break;
        case 2:
            manager.callKernel(kernel,
                               cl::NDRange{ (width + 3) / 4, height, depth * batch },
                               caller,
                               (cl_int)height,
                               (cl_int)width,
                               (cl_int)(depth * batch),
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               (cl_int)((width + 3) / 4),
                               (cl_int)height,
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[0],
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[1],
                               out);
            break;
        case 3:
            manager.callKernel(kernel,
                               cl::NDRange{ (width + 3) / 4, height, depth * batch },
                               caller,
                               (cl_int)height,
                               (cl_int)width,
                               (cl_int)(depth * batch),
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               (cl_int)((width + 3) / 4),
                               (cl_int)height,
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[0],
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[1],
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[2],
                               out);
            break;
        case 4:
            manager.callKernel(kernel,
                               cl::NDRange{ (width + 3) / 4, height, depth * batch },
                               caller,
                               (cl_int)height,
                               (cl_int)width,
                               (cl_int)(depth * batch),
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               (cl_int)((width + 3) / 4),
                               (cl_int)height,
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[0],
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[1],
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[2],
                               (cl_int)height,
                               (cl_int)width,
                               0,
                               0,
                               in[3],
                               out);
            break;
        default:
            THROW_NONAME("GPU", "Unsupported format");
    }
}

void callEltwiseKernel(OpenCLKernelManager& manager,
                       cl::Kernel kernel,
                       const Name& caller,
                       size_t inputNum,
                       size_t oBatch,
                       size_t oDepth,
                       size_t oHeight,
                       size_t oWidth,
                       const std::array<size_t, 4> iBatches,
                       const std::array<size_t, 4> iDepths,
                       const std::array<size_t, 4> iHeights,
                       const std::array<size_t, 4> iWidths,
                       const std::array<cl::Buffer, 4> in,
                       cl::Buffer& out)
{
    for (size_t i = 0; i < oBatch; ++i)
    {
        switch (inputNum)
        {
            case 1:
                manager.callKernel(kernel,
                                   cl::NDRange{ (oWidth + 3) / 4, oHeight, oDepth },
                                   caller,
                                   (cl_int)oWidth,
                                   (cl_int)oDepth,
                                   (cl_int)oHeight,
                                   (cl_int)oWidth,
                                   (cl_int)(i * oDepth * oWidth * oHeight),
                                   (cl_int)((oWidth + 3) / 4),
                                   (cl_int)oHeight,
                                   (cl_int)iDepths[0],
                                   (cl_int)iHeights[0],
                                   (cl_int)iWidths[0],
                                   (cl_int)(i % iBatches[0] * iDepths[0] * iHeights[0] * iWidths[0]),
                                   in[0],
                                   out);
                break;
            case 2:
                manager.callKernel(kernel,
                                   cl::NDRange{ (oWidth + 3) / 4, oHeight, oDepth },
                                   caller,
                                   (cl_int)oWidth,
                                   (cl_int)oDepth,
                                   (cl_int)oHeight,
                                   (cl_int)oWidth,
                                   (cl_int)(i * oDepth * oWidth * oHeight),
                                   (cl_int)((oWidth + 3) / 4),
                                   (cl_int)oHeight,
                                   (cl_int)iDepths[0],
                                   (cl_int)iHeights[0],
                                   (cl_int)iWidths[0],
                                   (cl_int)(i % iBatches[0] * iDepths[0] * iHeights[0] * iWidths[0]),
                                   in[0],
                                   (cl_int)iDepths[1],
                                   (cl_int)iHeights[1],
                                   (cl_int)iWidths[1],
                                   (cl_int)(i % iBatches[1] * iDepths[1] * iHeights[1] * iWidths[1]),
                                   in[1],
                                   out);
                break;
            case 3:
                manager.callKernel(kernel,
                                   cl::NDRange{ (oWidth + 3) / 4, oHeight, oDepth },
                                   caller,
                                   (cl_int)oWidth,
                                   (cl_int)oDepth,
                                   (cl_int)oHeight,
                                   (cl_int)oWidth,
                                   (cl_int)(i * oDepth * oWidth * oHeight),
                                   (cl_int)((oWidth + 3) / 4),
                                   (cl_int)oHeight,
                                   (cl_int)iDepths[0],
                                   (cl_int)iHeights[0],
                                   (cl_int)iWidths[0],
                                   (cl_int)(i % iBatches[0] * iDepths[0] * iHeights[0] * iWidths[0]),
                                   in[0],
                                   (cl_int)iDepths[1],
                                   (cl_int)iHeights[1],
                                   (cl_int)iWidths[1],
                                   (cl_int)(i % iBatches[1] * iDepths[1] * iHeights[1] * iWidths[1]),
                                   in[1],
                                   (cl_int)iDepths[2],
                                   (cl_int)iHeights[2],
                                   (cl_int)iWidths[2],
                                   (cl_int)(i % iBatches[2] * iDepths[2] * iHeights[2] * iWidths[2]),
                                   in[2],
                                   out);
                break;
            case 4:
                manager.callKernel(kernel,
                                   cl::NDRange{ (oWidth + 3) / 4, oHeight, oDepth },
                                   caller,
                                   (cl_int)oWidth,
                                   (cl_int)oDepth,
                                   (cl_int)oHeight,
                                   (cl_int)oWidth,
                                   (cl_int)(i * oDepth * oWidth * oHeight),
                                   (cl_int)((oWidth + 3) / 4),
                                   (cl_int)oHeight,
                                   (cl_int)iDepths[0],
                                   (cl_int)iHeights[0],
                                   (cl_int)iWidths[0],
                                   (cl_int)(i % iBatches[0] * iDepths[0] * iHeights[0] * iWidths[0]),
                                   in[0],
                                   (cl_int)iDepths[1],
                                   (cl_int)iHeights[1],
                                   (cl_int)iWidths[1],
                                   (cl_int)(i % iBatches[1] * iDepths[1] * iHeights[1] * iWidths[1]),
                                   in[1],
                                   (cl_int)iDepths[2],
                                   (cl_int)iHeights[2],
                                   (cl_int)iWidths[2],
                                   (cl_int)(i % iBatches[2] * iDepths[2] * iHeights[2] * iWidths[2]),
                                   in[2],
                                   (cl_int)iDepths[3],
                                   (cl_int)iHeights[3],
                                   (cl_int)iWidths[3],
                                   (cl_int)(i % iBatches[3] * iDepths[3] * iHeights[3] * iWidths[3]),
                                   in[3],
                                   out);
                break;
            default:
                throw std::runtime_error("Unsupported format");
        }
    }
}

cl::Kernel getSelectKernel(OpenCLKernelManager& manager, const Name& programName, const Name& kernelName, const Name& caller)
{
    if (!manager.hasKernel(programName, kernelName))
    {
        const std::string source =
#include <training/opencl/kernels/select.cl>
            ;
        manager.registerProgram(programName, source);
    }

    return manager.getKernel(programName, kernelName, caller);
}

cl::Kernel getTransposeKernel(OpenCLKernelManager& manager, const Name& caller)
{
    Name name = "transpose";

    if (!manager.hasKernel(name, "transpose_3d_nchw"))
    {
        const std::string source =
#include <training/opencl/kernels/transpose_3d_nchw.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, "transpose_3d_nchw", caller);
}

cl::Kernel getCumsumKernel(OpenCLKernelManager& manager, const Name& caller, bool isForward = true)
{
    Name name = isForward ? "cumsumForward" : "cumsumBackward";

    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/cumsum.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getRoundKernel(OpenCLKernelManager& manager, const Name& caller)
{
    Name name = "Round";

    if (!manager.hasKernel(name, "activation"))
    {
        const std::string source =
#include <training/opencl/kernels/activation.cl>
            ;
        manager.registerProgram(name, source, "-DUSE_ROUND");
    }

    return manager.getKernel(name, "activation", caller);
}

cl::Kernel getZeroOutputKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name programName = "zeroOutput";
    if (!manager.hasKernel(programName))
    {
        const std::string source =
#include <training/opencl/kernels/zero_output.cl>
            ;
        manager.registerProgram(programName, source);
    }

    return manager.getKernel(programName, caller);
}

cl::Kernel getEltwiseExtremumKernel(OpenCLKernelManager& manager, const Name& programName, const Name& options, const Name& caller)
{
    if (!manager.hasKernel(programName, "eltwise_extremum"))
    {
        const std::string source =
#include <training/opencl/kernels/eltwise_extremum.cl>
            ;
        manager.registerProgram(programName, source, options);
    }

    return manager.getKernel(programName, "eltwise_extremum", caller);
}

cl::Kernel getSeqMaskKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name programName = "seqMask";
    if (!manager.hasKernel(programName))
    {
        const std::string source =
#include <training/opencl/kernels/seq_mask.cl>
            ;
        manager.registerProgram(programName, source);
    }

    return manager.getKernel(programName, caller);
}

cl::Kernel getTileKernel(OpenCLKernelManager& manager, const Name& programName, const Name& options, const Name& caller)
{
    if (!manager.hasKernel(programName))
    {
        const std::string source =
#include <training/opencl/kernels/tile.cl>
            ;
        manager.registerProgram(programName, source, options);
    }

    return manager.getKernel(programName, caller);
}

cl::Kernel getGaussianUpsamplingDistributionForwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "gaussian_upsampling_distribution_forward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/gaussian_upsampling_distribution.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getGaussianUpsamplingDistributionBackwardKernel(OpenCLKernelManager& manager, bool backwardForLoc, const Name& caller)
{
    Name name = "gaussian_upsampling_distribution_backward_";
    Name options = "-DCALCULATE_";
    if (backwardForLoc)
    {
        name += "loc";
        options += "LOC_GRAD";
    }
    else
    {
        name += "scale";
        options += "SCALE_GRAD";
    }
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/gaussian_upsampling_distribution.cl>
            ;
        manager.registerProgram(name, source, options);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getReverseKernel(OpenCLKernelManager& manager, const Name& options, const Name& caller)
{
    Name programName = "reverse";
    if (!options.empty())
    {
        programName += "_only";
    }
    if (!manager.hasKernel(programName))
    {
        const std::string source =
#include <training/opencl/kernels/reverse.cl>
            ;
        manager.registerProgram(programName, source, options);
    }

    return manager.getKernel(programName, caller);
}

cl::Kernel getCopyKernel(OpenCLKernelManager& manager, const Name& options, const Name& caller)
{
    Name name = "copy_";
    if (!options.empty())
    {
        name += "and_add_";
    }
    name += "DT";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/copy.cl>
            ;
        manager.registerProgram(name, source, options);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getTargetsReductionKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "reduce_targets";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/targets_reduction.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getInitAlignmentKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "init_alignment";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/init_alignment.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

void callEltwiseExtremumKernel(OpenCLKernelManager& manager,
                               cl::Kernel kernel,
                               const Name& caller,
                               size_t batch,
                               size_t depth,
                               size_t height,
                               size_t width,
                               size_t index,
                               const cl::Buffer& in,
                               cl::Buffer& indexes,
                               cl::Buffer& out)
{
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, depth * batch },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)(depth * batch),
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)((width + 3) / 4),
                       (cl_int)height,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       static_cast<dtype>(index),
                       in,
                       indexes,
                       out);
}

cl::Kernel getGlobalL2SquareNormKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "global_l2_square_norm";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/global_l2_square_norm.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}
cl::Kernel getAdamKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "adam";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/adam.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getDynamicDepthwiseConv2DKernel(OpenCLKernelManager& manager, const Name& options, const Name& path, const Name& outputName, const Name& caller)
{
    const Name name = "dynamicDepthwiseConv2D" + path + outputName;
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/dynamic_depthwise_conv_2d.cl>
            ;
        manager.registerProgram(name, source, options + " -DPATH=" + path + " -DOUTPUT=" + outputName);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getReduceForwardKernel(OpenCLKernelManager& manager, size_t dimension, const Name& options, const Name& caller)
{
    const Name name = "reduction_nchw_TP" + std::to_string(dimension);

    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/reduction_nchw.cl>
            ;
        manager.registerProgram(name, source, options);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getReduceBackwardKernel(OpenCLKernelManager& manager, size_t dimension, const Name& options, const Name& caller)
{
    const Name name = "reduction_nchw_backward_TP" + std::to_string(dimension);

    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/reduction_nchw_backward.cl>
            ;
        manager.registerProgram(name, source, options);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getReduceDefaultForwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "reduction_default";

    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/reduction_default.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getReduceDefaultBackwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "reduction_default_backward";

    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/reduction_default_backward.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getGradientClippingKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "gradient_clipping";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/gradient_clipping.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getEltwiseKernel(OpenCLKernelManager& manager, size_t inputNum, const Name& programName, const Name& options, const Name& caller)
{
    if (!manager.hasKernel(programName + std::to_string(inputNum), "eltwise_AMEMnchw_" + std::to_string(inputNum)))
    {
        const std::string source =
#include <training/opencl/kernels/eltwise.cl>
            ;
        manager.registerProgram(programName + std::to_string(inputNum), source, options);
    }

    return manager.getKernel(programName + std::to_string(inputNum), "eltwise_AMEMnchw_" + std::to_string(inputNum), caller);
}

cl::Kernel getBrEltWiseKernel(OpenCLKernelManager& manager, size_t inputNum, const Name& programName, const Name& options, const Name& caller)
{
    if (!manager.hasKernel(programName + std::to_string(inputNum), "eltwise_br_AMEMnchw_" + std::to_string(inputNum)))
    {
        const std::string source =
#include <training/opencl/kernels/eltwise_br.cl>
            ;
        manager.registerProgram(programName + std::to_string(inputNum), source, options);
    }

    return manager.getKernel(programName + std::to_string(inputNum), "eltwise_br_AMEMnchw_" + std::to_string(inputNum), caller);
}

cl::Kernel getActivationForwardKernel(OpenCLKernelManager& manager, const Name& programName, const Name& options, const Name& caller)
{
    if (!manager.hasKernel(programName, "activation"))
    {
        const std::string source =
#include <training/opencl/kernels/activation.cl>
            ;
        manager.registerProgram(programName, source, options);
    }

    return manager.getKernel(programName, "activation", caller);
}

cl::Kernel getActivationBackwardKernel(OpenCLKernelManager& manager, const Name& programName, const Name& options, const Name& caller)
{
    if (!manager.hasKernel(programName, "activation_backward"))
    {
        const std::string source =
#include <training/opencl/kernels/activation.cl>
            ;
        manager.registerProgram(programName, source, options);
    }

    return manager.getKernel(programName, "activation_backward", caller);
}

cl::Kernel getAddBiasKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "addBias";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/fixed_bias.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getSplitterForwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "splitterForward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/splitter.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getSplitterBackwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "splitterBackward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/splitter.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getSoftMaxForwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "softmaxForward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/softmax.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getSoftMaxBackwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "softmaxBackward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/softmax.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getDropoutForwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "dropoutForward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/dropout.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getDropoutBackwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "dropoutBackward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/dropout.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getHSigmoidBackwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "hsigmoidBackward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/hsigmoid_backward.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

cl::Kernel getHSwishBackwardKernel(OpenCLKernelManager& manager, const Name& caller)
{
    const Name name = "hswishBackward";
    if (!manager.hasKernel(name))
    {
        const std::string source =
#include <training/opencl/kernels/hswish_backward.cl>
            ;
        manager.registerProgram(name, source);
    }

    return manager.getKernel(name, caller);
}

}

namespace raul
{
namespace gpu
{

void ReLU(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getReLUKernel(manager, caller);

    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void ReLU6(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getReLU6Kernel(manager, caller);

    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void ReLUBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& delta, cl::Buffer& prevDelta)
{
    auto kernel = getReLUBackwardKernel(manager, caller);

    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       out,
                       delta,
                       prevDelta);
}

void ReLU6Backward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& delta, cl::Buffer& prevDelta)
{
    auto kernel = getReLU6BackwardKernel(manager, caller);

    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       out,
                       delta,
                       prevDelta);
}

void clampForward(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t batch,
                  size_t depth,
                  size_t height,
                  size_t width,
                  const dtype min,
                  const dtype max,
                  const cl::Buffer& in,
                  const cl::Buffer& out)
{
    auto kernel = getClampKernel(manager, "Forward", caller);

    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, depth * batch },
                       caller,
                       (cl_int)((width + 3) / 4),
                       (cl_int)height,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       min,
                       max,
                       in,
                       out);
}

void clampBackward(OpenCLKernelManager& manager,
                   const Name& caller,
                   size_t batch,
                   size_t depth,
                   size_t height,
                   size_t width,
                   const dtype min,
                   const dtype max,
                   const cl::Buffer& input,
                   const cl::Buffer& deltas,
                   const cl::Buffer& prevLayerDelta)
{
    auto kernel = getClampKernel(manager, "Backward", caller);

    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, depth * batch },
                       caller,
                       (cl_int)((width + 3) / 4),
                       (cl_int)height,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       min,
                       max,
                       input,
                       deltas,
                       prevLayerDelta);
}

// ijk -> kji
void swapIndices3d(OpenCLKernelManager& manager, const Name& caller, size_t dim1, size_t dim2, size_t dim3, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getSwap_ijk2kjiKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ dim1, dim2, 1 }, caller, (cl_int)dim1, (cl_int)dim2, (cl_int)dim3, in, out);
}

void eltwiseSumOp(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, size_t inputNum, const std::array<cl::Buffer, 4>& in, cl::Buffer& out)
{
    auto kernel = getEltwiseKernel(manager, inputNum, "SUM", "-DUSE_NCHW -DUSE_SUM -DN=" + std::to_string(inputNum), caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, batch, depth, height, width, in, out);
}

void eltwiseSumOp(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t oBatch,
                  size_t oDepth,
                  size_t oHeight,
                  size_t oWidth,
                  size_t inputNum,
                  const std::array<size_t, 4> iBatches,
                  const std::array<size_t, 4> iDepths,
                  const std::array<size_t, 4> iHeights,
                  const std::array<size_t, 4> iWidths,
                  const std::array<cl::Buffer, 4>& in,
                  cl::Buffer& out)
{
    auto kernel = getBrEltWiseKernel(manager, inputNum, "SUM_BR", "-DUSE_NCHW -DUSE_SUM -DN=" + std::to_string(inputNum), caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, oBatch, oDepth, oHeight, oWidth, iBatches, iDepths, iHeights, iWidths, in, out);
}

void eltwiseSubOp(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in0, const cl::Buffer& in1, cl::Buffer& out)
{
    const size_t inputNum = 2;
    auto kernel = getEltwiseKernel(manager, inputNum, "SUB", "-DUSE_NCHW -DUSE_SUB -DN=2", caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, batch, depth, height, width, { in0, in1 }, out);
}

void eltwiseSubOp(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t oBatch,
                  size_t oDepth,
                  size_t oHeight,
                  size_t oWidth,
                  size_t iBatch0,
                  size_t iDepth0,
                  size_t iHeight0,
                  size_t iWidth0,
                  const cl::Buffer& in0,
                  size_t iBatch1,
                  size_t iDepth1,
                  size_t iHeight1,
                  size_t iWidth1,
                  const cl::Buffer& in1,
                  cl::Buffer& out)
{
    const size_t inputNum = 2;
    auto kernel = getBrEltWiseKernel(manager, inputNum, "SUB_BR", "-DUSE_NCHW -DUSE_SUB -DN=2", caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, oBatch, oDepth, oHeight, oWidth, { iBatch0, iBatch1 }, { iDepth0, iDepth1 }, { iHeight0, iHeight1 }, { iWidth0, iWidth1 }, { in0, in1 }, out);
}

void eltwiseMulOp(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, size_t inputNum, const std::array<cl::Buffer, 4>& in, cl::Buffer& out)
{
    auto kernel = getEltwiseKernel(manager, inputNum, "MUL", "-DUSE_NCHW -DUSE_PROD -DN=" + std::to_string(inputNum), caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, batch, depth, height, width, in, out);
}

void eltwiseMulOp(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t oBatch,
                  size_t oDepth,
                  size_t oHeight,
                  size_t oWidth,
                  size_t inputNum,
                  const std::array<size_t, 4> iBatches,
                  const std::array<size_t, 4> iDepths,
                  const std::array<size_t, 4> iHeights,
                  const std::array<size_t, 4> iWidths,
                  const std::array<cl::Buffer, 4>& in,
                  cl::Buffer& out)
{
    auto kernel = getBrEltWiseKernel(manager, inputNum, "MUL_BR", "-DUSE_NCHW -DUSE_PROD -DN=" + std::to_string(inputNum), caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, oBatch, oDepth, oHeight, oWidth, iBatches, iDepths, iHeights, iWidths, in, out);
}

void eltwiseDivOp(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in0, const cl::Buffer& in1, cl::Buffer& out)
{
    const size_t inputNum = 2;
    auto kernel = getEltwiseKernel(manager, inputNum, "DIV", "-DUSE_NCHW -DUSE_DIV -DN=2", caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, batch, depth, height, width, { in0, in1 }, out);
}

void eltwiseDivOp(OpenCLKernelManager& manager,
                  const Name& caller,
                  size_t oBatch,
                  size_t oDepth,
                  size_t oHeight,
                  size_t oWidth,
                  size_t iBatch0,
                  size_t iDepth0,
                  size_t iHeight0,
                  size_t iWidth0,
                  const cl::Buffer& in0,
                  size_t iBatch1,
                  size_t iDepth1,
                  size_t iHeight1,
                  size_t iWidth1,
                  const cl::Buffer& in1,
                  cl::Buffer& out)
{
    const size_t inputNum = 2;
    auto kernel = getBrEltWiseKernel(manager, inputNum, "DIV_BR", "-DUSE_NCHW -DUSE_DIV -DN=2", caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, oBatch, oDepth, oHeight, oWidth, { iBatch0, iBatch1 }, { iDepth0, iDepth1 }, { iHeight0, iHeight1 }, { iWidth0, iWidth1 }, { in0, in1 }, out);
}

void slice(OpenCLKernelManager& manager,
           const Name& caller,
           size_t axisNum,
           size_t iDepth,
           size_t iHeight,
           size_t iWidth,
           size_t axisMax,
           size_t numOfSlices,
           size_t inSize,
           size_t batchOff,
           size_t totalSize,
           const std::array<size_t, 4>& oWidth,
           const std::array<size_t, 4>& oHeight,
           const std::array<size_t, 4>& oOffset,
           const std::array<size_t, 4>& axisLen,
           const cl::Buffer& in,
           std::array<cl::Buffer, 4>& out)
{
    auto kernel = getSliceKernel(manager, axisNum, numOfSlices, caller);

    size_t gs[3]{ (iWidth + 3) / 4, iHeight, iDepth };
    gs[axisNum] = totalSize;

    switch (numOfSlices)
    {
        case 1:
            manager.callKernel(kernel,
                               cl::NDRange{ gs[0], gs[1], gs[2] },
                               caller,
                               (cl_int)iWidth,
                               (cl_int)iHeight,
                               0,
                               0,
                               (cl_int)axisMax,
                               (cl_int)numOfSlices - 1,
                               (cl_int)inSize,
                               (cl_int)batchOff,
                               (cl_int)gs[0],
                               (cl_int)gs[1],
                               in,
                               (cl_int)oWidth[0],
                               (cl_int)oHeight[0],
                               (cl_int)oOffset[0],
                               0,
                               0,
                               (cl_int)oWidth[0],
                               out[0]);
            break;
        case 2:
            manager.callKernel(kernel,
                               cl::NDRange{ gs[0], gs[1], gs[2] },
                               caller,
                               (cl_int)iWidth,
                               (cl_int)iHeight,
                               0,
                               0,
                               (cl_int)axisMax,
                               (cl_int)numOfSlices - 1,
                               (cl_int)inSize,
                               (cl_int)batchOff,
                               (cl_int)gs[0],
                               (cl_int)gs[1],
                               in,
                               (cl_int)oWidth[0],
                               (cl_int)oHeight[0],
                               (cl_int)oOffset[0],
                               0,
                               0,
                               (cl_int)oWidth[0],
                               out[0],
                               (cl_int)oWidth[1],
                               (cl_int)oHeight[1],
                               (cl_int)oOffset[1],
                               0,
                               0,
                               (cl_int)oWidth[1],
                               (cl_int)axisLen[0],
                               out[1]);
            break;
        case 3:
            manager.callKernel(kernel,
                               cl::NDRange{ gs[0], gs[1], gs[2] },
                               caller,
                               (cl_int)iWidth,
                               (cl_int)iHeight,
                               0,
                               0,
                               (cl_int)axisMax,
                               (cl_int)numOfSlices - 1,
                               (cl_int)inSize,
                               (cl_int)batchOff,
                               (cl_int)gs[0],
                               (cl_int)gs[1],
                               in,
                               (cl_int)oWidth[0],
                               (cl_int)oHeight[0],
                               (cl_int)oOffset[0],
                               0,
                               0,
                               (cl_int)oWidth[0],
                               out[0],
                               (cl_int)oWidth[1],
                               (cl_int)oHeight[1],
                               (cl_int)oOffset[1],
                               0,
                               0,
                               (cl_int)oWidth[1],
                               (cl_int)axisLen[0],
                               out[1],
                               (cl_int)oWidth[2],
                               (cl_int)oHeight[2],
                               (cl_int)oOffset[2],
                               0,
                               0,
                               (cl_int)oWidth[2],
                               (cl_int)axisLen[1],
                               out[2]);
            break;
        case 4:
            manager.callKernel(kernel,
                               cl::NDRange{ gs[0], gs[1], gs[2] },
                               caller,
                               (cl_int)iWidth,
                               (cl_int)iHeight,
                               0,
                               0,
                               (cl_int)axisMax,
                               (cl_int)numOfSlices - 1,
                               (cl_int)inSize,
                               (cl_int)batchOff,
                               (cl_int)gs[0],
                               (cl_int)gs[1],
                               in,
                               (cl_int)oWidth[0],
                               (cl_int)oHeight[0],
                               (cl_int)oOffset[0],
                               0,
                               0,
                               (cl_int)oWidth[0],
                               out[0],
                               (cl_int)oWidth[1],
                               (cl_int)oHeight[1],
                               (cl_int)oOffset[1],
                               0,
                               0,
                               (cl_int)oWidth[1],
                               (cl_int)axisLen[0],
                               out[1],
                               (cl_int)oWidth[2],
                               (cl_int)oHeight[2],
                               (cl_int)oOffset[2],
                               0,
                               0,
                               (cl_int)oWidth[2],
                               (cl_int)axisLen[1],
                               out[2],
                               (cl_int)oWidth[3],
                               (cl_int)oHeight[3],
                               (cl_int)oOffset[3],
                               0,
                               0,
                               (cl_int)oWidth[3],
                               (cl_int)axisLen[2],
                               out[3]);
            break;
        default:
            THROW_NONAME("GPU", "Unsupported format");
    }
}

void concat(OpenCLKernelManager& manager,
            const Name& caller,
            size_t axisNum,
            size_t oDepth,
            size_t oHeight,
            size_t oWidth,
            size_t axisMax,
            size_t numOfSlices,
            size_t outSize,
            size_t batchOff,
            const std::array<size_t, 4>& iWidth,
            const std::array<size_t, 4>& iHeight,
            const std::array<size_t, 4>& iOffset,
            const std::array<size_t, 4>& axisLen,
            const std::array<cl::Buffer, 4>& in,
            cl::Buffer& out)
{
    auto kernel = getConcatKernel(manager, axisNum, numOfSlices, caller);

    size_t gs[3]{ (oWidth + 3) / 4, oHeight, oDepth };
    gs[axisNum] = axisMax + axisLen[numOfSlices - 1];
    switch (numOfSlices)
    {
        case 1:
            manager.callKernel(kernel,
                               cl::NDRange{ gs[0], gs[1], gs[2] },
                               caller,
                               (cl_int)oWidth,
                               (cl_int)(oWidth * oHeight),
                               0,
                               0,
                               (cl_int)axisMax,
                               (cl_int)numOfSlices - 1,
                               (cl_int)outSize,
                               (cl_int)batchOff,
                               (cl_int)gs[0],
                               (cl_int)gs[1],
                               (cl_int)iHeight[0],
                               (cl_int)iWidth[0],
                               (cl_int)iOffset[0],
                               0,
                               0,
                               (cl_int)iWidth[0],
                               in[0],
                               out);
            break;
        case 2:
            manager.callKernel(kernel,
                               cl::NDRange{ gs[0], gs[1], gs[2] },
                               caller,
                               (cl_int)oWidth,
                               (cl_int)(oWidth * oHeight),
                               0,
                               0,
                               (cl_int)axisMax,
                               (cl_int)numOfSlices - 1,
                               (cl_int)outSize,
                               (cl_int)batchOff,
                               (cl_int)gs[0],
                               (cl_int)gs[1],
                               (cl_int)iHeight[0],
                               (cl_int)iWidth[0],
                               (cl_int)iOffset[0],
                               0,
                               0,
                               (cl_int)iWidth[0],
                               in[0],
                               (cl_int)iHeight[1],
                               (cl_int)iWidth[1],
                               (cl_int)iOffset[1],
                               0,
                               0,
                               (cl_int)iWidth[1],
                               (cl_int)axisLen[0],
                               in[1],
                               out);
            break;
        case 3:
            manager.callKernel(kernel,
                               cl::NDRange{ gs[0], gs[1], gs[2] },
                               caller,
                               (cl_int)oWidth,
                               (cl_int)(oWidth * oHeight),
                               0,
                               0,
                               (cl_int)axisMax,
                               (cl_int)numOfSlices - 1,
                               (cl_int)outSize,
                               (cl_int)batchOff,
                               (cl_int)gs[0],
                               (cl_int)gs[1],
                               (cl_int)iHeight[0],
                               (cl_int)iWidth[0],
                               (cl_int)iOffset[0],
                               0,
                               0,
                               (cl_int)iWidth[0],
                               in[0],
                               (cl_int)iHeight[1],
                               (cl_int)iWidth[1],
                               (cl_int)iOffset[1],
                               0,
                               0,
                               (cl_int)iWidth[1],
                               (cl_int)axisLen[0],
                               in[1],
                               (cl_int)iHeight[2],
                               (cl_int)iWidth[2],
                               (cl_int)iOffset[2],
                               0,
                               0,
                               (cl_int)iWidth[2],
                               (cl_int)axisLen[1],
                               in[2],
                               out);
            break;
        case 4:
            manager.callKernel(kernel,
                               cl::NDRange{ gs[0], gs[1], gs[2] },
                               caller,
                               (cl_int)oWidth,
                               (cl_int)(oWidth * oHeight),
                               0,
                               0,
                               (cl_int)axisMax,
                               (cl_int)numOfSlices - 1,
                               (cl_int)outSize,
                               (cl_int)batchOff,
                               (cl_int)gs[0],
                               (cl_int)gs[1],
                               (cl_int)iHeight[0],
                               (cl_int)iWidth[0],
                               (cl_int)iOffset[0],
                               0,
                               0,
                               (cl_int)iWidth[0],
                               in[0],
                               (cl_int)iHeight[1],
                               (cl_int)iWidth[1],
                               (cl_int)iOffset[1],
                               0,
                               0,
                               (cl_int)iWidth[1],
                               (cl_int)axisLen[0],
                               in[1],
                               (cl_int)iHeight[2],
                               (cl_int)iWidth[2],
                               (cl_int)iOffset[2],
                               0,
                               0,
                               (cl_int)iWidth[2],
                               (cl_int)axisLen[1],
                               in[2],
                               (cl_int)iHeight[3],
                               (cl_int)iWidth[3],
                               (cl_int)iOffset[3],
                               0,
                               0,
                               (cl_int)iWidth[3],
                               (cl_int)axisLen[2],
                               in[3],
                               out);
            break;
        default:
            THROW_NONAME("GPU", "Unsupported format");
    }
}

void iota(OpenCLKernelManager& manager, const Name& caller, dtype startPoint, size_t size, cl::Buffer& out)
{
    auto kernel = getIotaKernel(manager, caller);

    manager.callKernel(kernel, cl::NDRange{ (size + 3) / 4, 1, 1 }, caller, startPoint, (cl_int)size, out);
}

void selectForward(OpenCLKernelManager& manager,
                   const Name& caller,
                   size_t batch,
                   size_t depth,
                   size_t height,
                   size_t width,
                   const cl::Buffer& cond,
                   const cl::Buffer& in0,
                   const cl::Buffer& in1,
                   cl::Buffer& out)
{
    auto kernel = getSelectKernel(manager, "SELECT::FORWARD", "selectForward", caller);

    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, depth * batch },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)(depth * batch),
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)((width + 3) / 4),
                       (cl_int)height,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       cond,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       in0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       in1,
                       out);
}

void selectBackward(OpenCLKernelManager& manager,
                    const Name& caller,
                    size_t index,
                    size_t batch,
                    size_t depth,
                    size_t height,
                    size_t width,
                    const cl::Buffer& cond,
                    const cl::Buffer& deltas,
                    cl::Buffer& prevDelta)
{
    auto kernel = getSelectKernel(manager, "SELECT::BACKWARD", "selectBackward", caller);

    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, depth * batch },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)(depth * batch),
                       (cl_int)index,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)((width + 3) / 4),
                       (cl_int)height,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       cond,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       deltas,
                       prevDelta);
}

void transpose(OpenCLKernelManager& manager,
               const Name& caller,
               size_t batch,
               size_t iDepth,
               size_t iHeight,
               size_t iWidth,
               size_t oDepth,
               size_t oHeight,
               size_t oWidth,
               size_t dim0,
               size_t dim1,
               size_t dim2,
               size_t dim3,
               const cl::Buffer& in,
               cl::Buffer& out)
{
    auto kernel = getTransposeKernel(manager, caller);

    manager.callKernel(kernel,
                       cl::NDRange{ (iWidth + 3) / 4, iHeight, iDepth * batch },
                       caller,
                       (cl_int)iWidth,
                       (cl_int)iHeight,
                       0,
                       0,
                       (cl_int)oWidth,
                       (cl_int)oHeight,
                       0,
                       0,
                       (cl_int)dim0,
                       (cl_int)dim1,
                       (cl_int)dim2,
                       (cl_int)dim3,
                       (cl_int)iWidth,
                       (cl_int)iDepth,
                       (cl_int)oDepth,
                       (cl_int)((iWidth + 3) / 4),
                       (cl_int)iHeight,
                       in,
                       out);
}

void round(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getRoundKernel(manager, caller);

    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void batchnorm_backward(OpenCLKernelManager& manager,
                        const Name& caller,
                        raul::Dimension dim,
                        size_t batchSize,
                        size_t depth,
                        size_t height,
                        size_t width,
                        const cl::Buffer& deltas,
                        const cl::Buffer& xHat,
                        const cl::Buffer& varSqrt,
                        const cl::Buffer& gamma,
                        cl::Buffer& prevDeltas)
{
    auto kernel = getBackwardBatchNormKernel(manager, dim, true);
    manager.callKernel(kernel,
                       getBatchNormWorkSize(dim, depth, height, width),
                       caller / "batchnorm_backward",
                       (cl_int)batchSize,
                       (cl_int)depth,
                       (cl_int)height,
                       (cl_int)width,
                       deltas,
                       xHat,
                       varSqrt,
                       gamma,
                       prevDeltas);
}

void batchnorm_backward(OpenCLKernelManager& manager,
                        const Name& caller,
                        raul::Dimension dim,
                        size_t batchSize,
                        size_t depth,
                        size_t height,
                        size_t width,
                        const cl::Buffer& deltas,
                        const cl::Buffer& xHat,
                        const cl::Buffer& varSqrt,
                        const cl::Buffer& gamma,
                        cl::Buffer& prevDeltas,
                        cl::Buffer& nablaBeta,
                        cl::Buffer& nablaGamma)
{
    auto kernel = getBackwardBatchNormKernel(manager, dim, false);
    manager.callKernel(kernel,
                       getBatchNormWorkSize(dim, depth, height, width),
                       caller / "batchnorm_backward",
                       (cl_int)batchSize,
                       (cl_int)depth,
                       (cl_int)height,
                       (cl_int)width,
                       deltas,
                       xHat,
                       varSqrt,
                       gamma,
                       prevDeltas,
                       nablaBeta,
                       nablaGamma);
}

void batchnorm_forward(OpenCLKernelManager& manager,
                       const Name& caller,
                       raul::Dimension dim,
                       size_t batchSize,
                       size_t depth,
                       size_t height,
                       size_t width,
                       dtype momentum,
                       dtype eps,
                       bool useMomentum,
                       const cl::Buffer& input,
                       const cl::Buffer& beta,
                       const cl::Buffer& gamma,
                       cl::Buffer& mean,
                       cl::Buffer& var,
                       cl::Buffer& xHat,
                       cl::Buffer& varSqrt,
                       cl::Buffer& output,
                       cl::Buffer& meanEval,
                       cl::Buffer& varEval)
{
    auto kernel = getForwardBatchNormKernel(manager, dim);
    auto workSize = getBatchNormWorkSize(dim, depth, height, width);
    dtype reciprocalN = 1.0_dt / static_cast<dtype>(batchSize * height * width * depth / workSize[0]);
    manager.callKernel(kernel,
                       workSize,
                       caller / "batchnorm_forward",
                       (cl_int)batchSize,
                       (cl_int)depth,
                       (cl_int)height,
                       (cl_int)width,
                       reciprocalN,
                       momentum,
                       eps,
                       useMomentum ? (cl_int)1 : (cl_int)0,
                       input,
                       beta,
                       gamma,
                       mean,
                       var,
                       xHat,
                       varSqrt,
                       output,
                       meanEval,
                       varEval);
}

void batchnorm_test(OpenCLKernelManager& manager,
                    const Name& caller,
                    raul::Dimension dim,
                    size_t batchSize,
                    size_t depth,
                    size_t height,
                    size_t width,
                    dtype eps,
                    const cl::Buffer& input,
                    const cl::Buffer& beta,
                    const cl::Buffer& gamma,
                    const cl::Buffer& meanEval,
                    const cl::Buffer& varEval,
                    cl::Buffer& output)
{

    auto kernel = getTestBatchNormKernel(manager, dim);
    manager.callKernel(kernel,
                       getBatchNormWorkSize(dim, depth, height, width),
                       caller / "batchnorm_test",
                       (cl_int)batchSize,
                       (cl_int)depth,
                       (cl_int)height,
                       (cl_int)width,
                       eps,
                       input,
                       beta,
                       gamma,
                       meanEval,
                       varEval,
                       output);
}

void cumsum(OpenCLKernelManager& manager,
            const Name& caller,
            size_t x,
            size_t y,
            size_t z,
            size_t dimension,
            size_t size,
            const shape& outputStrides,
            const cl::Buffer& in,
            cl::Buffer& out,
            bool isForward)
{
    auto kernel = getCumsumKernel(manager, caller, isForward);
    manager.callKernel(kernel,
                       cl::NDRange{ size, 1, 1 },
                       caller,
                       (cl_int)x,
                       (cl_int)y,
                       (cl_int)z,
                       (cl_int)dimension,
                       (cl_int)size,
                       (cl_int)outputStrides[0],
                       (cl_int)outputStrides[1],
                       (cl_int)outputStrides[2],
                       (cl_int)outputStrides[3],
                       in,
                       out);
}

void eltwiseMaxForwardOp(OpenCLKernelManager& manager,
                         const Name& caller,
                         size_t batch,
                         size_t depth,
                         size_t height,
                         size_t width,
                         size_t index,
                         const cl::Buffer& in,
                         cl::Buffer& indexes,
                         cl::Buffer& out)
{
    auto kernel = getEltwiseExtremumKernel(manager, "MAX_FORWARD", "-DUSE_MAX -DUSE_FORWARD", caller);
    callEltwiseExtremumKernel(manager, kernel, caller, batch, depth, height, width, index, in, indexes, out);
}

void eltwiseMinForwardOp(OpenCLKernelManager& manager,
                         const Name& caller,
                         size_t batch,
                         size_t depth,
                         size_t height,
                         size_t width,
                         size_t index,
                         const cl::Buffer& in,
                         cl::Buffer& indexes,
                         cl::Buffer& out)
{
    auto kernel = getEltwiseExtremumKernel(manager, "MIN_FORWARD", "-DUSE_MIN -DUSE_FORWARD", caller);
    callEltwiseExtremumKernel(manager, kernel, caller, batch, depth, height, width, index, in, indexes, out);
}

void eltwiseMaxBackwardOp(OpenCLKernelManager& manager,
                          const Name& caller,
                          size_t batch,
                          size_t depth,
                          size_t height,
                          size_t width,
                          size_t index,
                          const cl::Buffer& in,
                          cl::Buffer& indexes,
                          cl::Buffer& out)
{
    auto kernel = getEltwiseExtremumKernel(manager, "MAX_BACKWARD", "-DUSE_MAX", caller);
    callEltwiseExtremumKernel(manager, kernel, caller, batch, depth, height, width, index, in, indexes, out);
}

void eltwiseMinBackwardOp(OpenCLKernelManager& manager,
                          const Name& caller,
                          size_t batch,
                          size_t depth,
                          size_t height,
                          size_t width,
                          size_t index,
                          const cl::Buffer& in,
                          cl::Buffer& indexes,
                          cl::Buffer& out)
{
    auto kernel = getEltwiseExtremumKernel(manager, "MIN_BACKWARD", "-DUSE_MIN", caller);
    callEltwiseExtremumKernel(manager, kernel, caller, batch, depth, height, width, index, in, indexes, out);
}

void zeroOutput(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& length, cl::Buffer& out)
{
    auto kernel = getZeroOutputKernel(manager, caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height * depth, batch }, caller, (cl_int)width, (cl_int)(height * depth), (cl_int)((width + 3) / 4), (cl_int)(height * depth), in, length, out);
}

void nonZeroMask(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    const size_t inputNum = 1;
    auto kernel = getEltwiseKernel(manager, inputNum, "NON_ZERO_MASK", "-DUSE_NCHW -DUSE_NON_ZERO_MASK -DN=1", caller);
    callEltwiseKernel(manager, kernel, caller, inputNum, batch, depth, height, width, { in }, out);
}

void sequenceMask(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& lengths, cl::Buffer& mask)
{
    auto kernel = getSeqMaskKernel(manager, caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height * depth, batch }, caller, (cl_int)width, (cl_int)(height * depth), (cl_int)((width + 3) / 4), (cl_int)(height * depth), lengths, mask);
}

void gaussianUpsamplingDistributionForward(OpenCLKernelManager& manager,
                                           const Name& caller,
                                           size_t batch,
                                           size_t depth,
                                           size_t height,
                                           size_t width,
                                           const cl::Buffer& values,
                                           const cl::Buffer& loc,
                                           const cl::Buffer& scale,
                                           cl::Buffer& out)
{
    auto kernel = getGaussianUpsamplingDistributionForwardKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ (width + 3) / 4, height * depth, batch }, caller, (cl_int)batch, (cl_int)height, (cl_int)width, 1, (cl_int)height, (cl_int)width, values, loc, scale, out);
}

void gaussianUpsamplingDistributionBackward(OpenCLKernelManager& manager,
                                            const Name& caller,
                                            size_t batch,
                                            size_t depth,
                                            size_t height,
                                            size_t width,
                                            bool backwardForLoc,
                                            const cl::Buffer& values,
                                            const cl::Buffer& loc,
                                            const cl::Buffer& scale,
                                            const cl::Buffer& deltas,
                                            cl::Buffer& prevLayerDelta)
{

    auto kernel = getGaussianUpsamplingDistributionBackwardKernel(manager, backwardForLoc, caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, batch, 1 }, caller, (cl_int)((width + 3) / 4), (cl_int)batch, (cl_int)width, (cl_int)(height * depth), values, loc, scale, deltas, prevLayerDelta);
}

void tile(OpenCLKernelManager& manager,
          const Name& caller,
          size_t iDepth,
          size_t iHeight,
          size_t iWidth,
          size_t oDepth,
          size_t oHeight,
          size_t oWidth,
          size_t inOff,
          size_t outOff,
          bool isForward,
          const cl::Buffer& in,
          cl::Buffer& out)
{
    const Name programName = isForward ? "tileForward" : "tileBackward";
    const Name options = isForward ? "" : "-DBACKWARD";
    auto kernel = getTileKernel(manager, programName, options, caller);
    manager.callKernel(kernel,
                       cl::NDRange{ ((isForward ? iWidth : oWidth) + 3) / 4, oHeight, oDepth },
                       caller,
                       (cl_int)(((isForward ? oWidth : iWidth) + 3) / 4),
                       (cl_int)(isForward ? oHeight : iHeight),
                       (cl_int)(isForward ? iWidth : oWidth),
                       (cl_int)iDepth,
                       (cl_int)iHeight,
                       (cl_int)iWidth,
                       (cl_int)oDepth,
                       (cl_int)oHeight,
                       (cl_int)oWidth,
                       (cl_int)outOff,
                       (cl_int)inOff,
                       in,
                       out);
}

void reduceTargets(OpenCLKernelManager& manager,
                   const Name& caller,
                   size_t batch,
                   size_t idepth,
                   size_t odepth,
                   size_t iheight,
                   size_t oheight,
                   size_t width,
                   size_t reductionFactor,
                   const cl::Buffer& in,
                   cl::Buffer& out)
{
    auto kernel = getTargetsReductionKernel(manager, caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, oheight * odepth, batch },
                       caller,
                       (cl_int)width,
                       (cl_int)(iheight * idepth),
                       (cl_int)(oheight * odepth),
                       (cl_int)((width + 3) / 4),
                       (cl_int)(oheight * odepth),
                       (cl_int)reductionFactor,
                       in,
                       out);
}

void reverse(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getReverseKernel(manager, "-DUSE_REVERSE_ONLY", caller);
    manager.callKernel(kernel, cl::NDRange{ (width + 3) / 4, height * depth, batch }, caller, (cl_int)width, (cl_int)(height * depth), (cl_int)((width + 3) / 4), (cl_int)(height * depth), in, out);
}

void reverse(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& length, cl::Buffer& out)
{
    auto kernel = getReverseKernel(manager, "", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height * depth, batch }, caller, (cl_int)width, (cl_int)(height * depth), (cl_int)((width + 3) / 4), (cl_int)(height * depth), in, length, out);
}

void initAlignment(OpenCLKernelManager& manager, const Name& caller, dtype val, size_t batch, size_t height, cl::Buffer& out)
{
    auto kernel = getInitAlignmentKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ (height + 3) / 4, batch, 1 }, caller, val, (cl_int)height, (cl_int)batch, out);
}

void copy(OpenCLKernelManager& manager,
          const Name& caller,
          size_t sourceLen,
          size_t destinationLen,
          size_t sourceOffset,
          size_t destinationOffset,
          bool sumWithOldValues,
          const cl::Buffer& in,
          cl::Buffer& out)
{
    const Name options = sumWithOldValues ? "-DUSE_OLD_OUTPUT" : "";
    auto kernel = getCopyKernel(manager, options, caller);
    manager.callKernel(
        kernel, cl::NDRange{ (sourceLen + 3) / 4, 1, 1 }, caller, (cl_int)sourceLen, (cl_int)destinationLen, (cl_int)sourceOffset, (cl_int)destinationOffset, (cl_int)(sourceLen + 3) / 4, in, out);
}

void globalL2SquareNorm(OpenCLKernelManager& manager, const Name& caller, size_t inputSize, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getGlobalL2SquareNormKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ 1, 1, 1 }, caller, (cl_int)inputSize, (cl_int)((inputSize + 7) / 8), in, out);
}

void clipGradients(OpenCLKernelManager& manager, const Name& caller, size_t inputSize, dtype clipNorm, const cl::Buffer& currGlobalNorm, cl::Buffer& grad)
{
    auto kernel = getGradientClippingKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ (inputSize + 3) / 4, 1, 1 }, caller, (cl_int)inputSize, (cl_int)((inputSize + 3) / 4), clipNorm, currGlobalNorm, grad);
}

void adam(OpenCLKernelManager& manager,
          const Name& caller,
          size_t size,
          dtype alpha,
          dtype beta1,
          dtype beta2,
          dtype epsilon,
          size_t useSimpleEpsilon,
          const cl::Buffer& grad,
          const cl::Buffer& betaT1,
          const cl::Buffer& betaT2,
          cl::Buffer& m,
          cl::Buffer& v,
          cl::Buffer& param)
{
    auto kernel = getAdamKernel(manager, caller);
    manager.callKernel(
        kernel, cl::NDRange{ (size + 3) / 4, 1, 1 }, caller, (cl_int)((size + 3) / 4), (cl_int)size, alpha, beta1, beta2, epsilon, (cl_int)useSimpleEpsilon, grad, betaT1, betaT2, m, v, param);
}

void dynamicDepthwiseConv2DForward(OpenCLKernelManager& manager,
                                   const Name& caller,
                                   size_t batchSize,
                                   size_t inputC,
                                   size_t outputH,
                                   size_t outputW,
                                   size_t channelMultiplier,
                                   size_t filterH,
                                   size_t filterW,
                                   const cl::Buffer& in0,
                                   const cl::Buffer& in1,
                                   cl::Buffer& out)
{
    auto kernel = getDynamicDepthwiseConv2DKernel(manager, "-DUSE_FORWARD", "Forward", "", caller);
    manager.callKernel(
        kernel, cl::NDRange{ 1, 1, 1 }, caller, (cl_int)batchSize, (cl_int)inputC, (cl_int)outputH, (cl_int)outputW, (cl_int)channelMultiplier, (cl_int)filterH, (cl_int)filterW, in0, in1, out);
}

void dynamicDepthwiseConv2DBackward(OpenCLKernelManager& manager,
                                    const Name& caller,
                                    size_t batchSize,
                                    size_t inputC,
                                    size_t outputH,
                                    size_t outputW,
                                    size_t channelMultiplier,
                                    size_t filterH,
                                    size_t filterW,
                                    bool isForInput,
                                    const cl::Buffer& in0,
                                    const cl::Buffer& in1,
                                    cl::Buffer& out)
{
    auto kernel = getDynamicDepthwiseConv2DKernel(manager, (isForInput ? "-DUSE_BACKWARD -DFOR_INPUT" : "-DUSE_BACKWARD -DFOR_FILTERS"), "Backward", (isForInput ? "Input" : "Filters"), caller);
    manager.callKernel(
        kernel, cl::NDRange{ 1, 1, 1 }, caller, (cl_int)batchSize, (cl_int)inputC, (cl_int)outputH, (cl_int)outputW, (cl_int)channelMultiplier, (cl_int)filterH, (cl_int)filterW, in0, in1, out);
}

void reduceDimForward(OpenCLKernelManager& manager,
                      const Name& caller,
                      size_t batchSize,
                      size_t iDepth,
                      size_t iHeight,
                      size_t iWidth,
                      size_t oDepth,
                      size_t oHeight,
                      size_t oWidth,
                      size_t dimension,
                      const std::string& reductionType,
                      const cl::Buffer& in,
                      cl::Buffer& out)
{
    static std::unordered_map<std::string, Name> options{ { "mean", "-DUSE_MEAN" }, { "sum", "-DUSE_SUM" }, { "count_non_zero_elems", "-DUSE_COUNT_NON_ZERO_ELEMS" } };

    auto kernel = getReduceForwardKernel(manager, dimension, options[reductionType] + " -DAXIS=" + std::to_string(dimension), caller);

    size_t gs[3]{ (iWidth + 3) >> 2, iHeight, iDepth };
    gs[dimension] = 1;
    for (size_t i = 0; i < batchSize; ++i)
    {
        const auto inputOffset = i * iDepth * iHeight * iWidth;
        const auto outputOffset = i * oDepth * oHeight * oWidth;
        manager.callKernel(kernel,
                           cl::NDRange{ gs[0], gs[1], gs[2] },
                           caller,
                           (cl_int)iHeight,
                           (cl_int)iWidth,
                           0,
                           0,
                           (cl_int)oHeight,
                           (cl_int)oWidth,
                           0,
                           0,
                           (cl_int)iHeight,
                           (cl_int)iWidth,
                           (cl_int)iDepth,
                           (cl_int)oWidth,
                           (cl_int)oHeight,
                           (cl_int)inputOffset,
                           (cl_int)outputOffset,
                           1,
                           4,
                           (cl_int)gs[0],
                           (cl_int)gs[1],
                           (cl_int)batchSize,
                           in,
                           out);
    }
}

void reduceDimBackward(OpenCLKernelManager& manager,
                       const Name& caller,
                       size_t batchSize,
                       size_t iDepth,
                       size_t iHeight,
                       size_t iWidth,
                       size_t oDepth,
                       size_t oHeight,
                       size_t oWidth,
                       size_t dimension,
                       dtype divisor,
                       const cl::Buffer& deltas,
                       cl::Buffer& prevDelta)
{
    auto kernel = getReduceBackwardKernel(manager, dimension, "-DAXIS=" + std::to_string(dimension), caller);

    for (size_t i = 0; i < batchSize; ++i)
    {
        const auto inputOffset = i * iDepth * iHeight * iWidth;
        const auto outputOffset = i * oDepth * oHeight * oWidth;
        manager.callKernel(kernel,
                           cl::NDRange{ (oWidth + 3) >> 2, oHeight, oDepth },
                           caller,
                           (cl_int)((oWidth + 3) >> 2),
                           (cl_int)(oHeight),
                           (cl_int)iHeight,
                           (cl_int)iWidth,
                           (cl_int)oHeight,
                           (cl_int)oWidth,
                           (cl_int)inputOffset,
                           (cl_int)outputOffset,
                           divisor,
                           deltas,
                           prevDelta);
    }
}

void reduceDefaultForward(OpenCLKernelManager& manager, const Name& caller, size_t inputSize, dtype divisor, size_t countNonZeroElems, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getReduceDefaultForwardKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ 1, 1, 1 }, caller, (cl_int)inputSize, (cl_int)((inputSize + 7) / 8), divisor, (cl_int)countNonZeroElems, in, out);
}

void reduceDefaultBackward(OpenCLKernelManager& manager,
                           const Name& caller,
                           size_t batchSize,
                           size_t oDepth,
                           size_t oHeight,
                           size_t oWidth,
                           dtype divisor,
                           const cl::Buffer& deltas,
                           cl::Buffer& prevDelta)
{
    auto kernel = getReduceDefaultBackwardKernel(manager, caller);
    size_t batchOffset = 0;
    for (size_t b = 0; b < batchSize; ++b)
    {
        manager.callKernel(kernel,
                           cl::NDRange{ (oWidth + 3) >> 2, oHeight, oDepth },
                           caller,
                           (cl_int)((oWidth + 3) >> 2),
                           (cl_int)oHeight,
                           (cl_int)oDepth,
                           (cl_int)oHeight,
                           (cl_int)oWidth,
                           (cl_int)batchOffset,
                           divisor,
                           deltas,
                           prevDelta);
        batchOffset += oDepth * oHeight * oWidth;
    }
}

void expForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "expForward", "-DUSE_EXP", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void expBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "expBackward", "-DUSE_EXP", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       out,
                       deltas,
                       prevDelta);
}

void sqrtForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "sqrtForward", "-DUSE_SQRT", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void sqrtBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "sqrtBackward", "-DUSE_SQRT", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       out,
                       deltas,
                       prevDelta);
}

void rsqrtForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "rsqrtForward", "-DUSE_RSQRT", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void rsqrtBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "rsqrtBackward", "-DUSE_RSQRT", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       out,
                       deltas,
                       prevDelta);
}

void squareForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "squareForward", "-DUSE_SQUARE", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void squareBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "squareBackward", "-DUSE_SQUARE", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       in,
                       deltas,
                       prevDelta);
}

void logForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "logForward", "-DUSE_LOG", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void logBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "logBackward", "-DUSE_LOG", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       in,
                       deltas,
                       prevDelta);
}

void addBias(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, dtype bias, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getAddBiasKernel(manager, caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, bias, in, out);
}

void sigmoidForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "sigmoidForward", "-DUSE_SIGMOID", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void sigmoidBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "sigmoidBackward", "-DUSE_SIGMOID", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       out,
                       deltas,
                       prevDelta);
}

void softplusForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, dtype beta, dtype threshold, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "softplusForward", "-DUSE_SOFTPLUS", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       beta,
                       threshold,
                       in,
                       out);
}

void softplusBackward(OpenCLKernelManager& manager,
                      const Name& caller,
                      size_t batch,
                      size_t depth,
                      size_t height,
                      size_t width,
                      dtype beta,
                      dtype threshold,
                      const cl::Buffer& out,
                      const cl::Buffer& deltas,
                      cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "softplusBackward", "-DUSE_SOFTPLUS", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       beta,
                       threshold,
                       out,
                       deltas,
                       prevDelta);
}

void tanhForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "tanhForward", "-DUSE_TANH", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void tanhBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "tanhBackward", "-DUSE_TANH", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       out,
                       deltas,
                       prevDelta);
}

void swishForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "swishForward", "-DUSE_SWISH", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void swishBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "swishBackward", "-DUSE_SWISH", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       in,
                       deltas,
                       prevDelta);
}

void splitterForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getSplitterForwardKernel(manager, caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void splitterBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getSplitterBackwardKernel(manager, caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       deltas,
                       prevDelta);
}

void softmaxForward(OpenCLKernelManager& manager, const Name& caller, size_t externalDimSize, size_t internalDimSize, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getSoftMaxForwardKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ 1, 1, 1 }, caller, (cl_int)externalDimSize, (cl_int)internalDimSize, in, out);
}

void softmaxBackward(OpenCLKernelManager& manager, const Name& caller, size_t externalDimSize, size_t internalDimSize, const cl::Buffer& out, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getSoftMaxBackwardKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ 1, 1, 1 }, caller, (cl_int)externalDimSize, (cl_int)internalDimSize, out, deltas, prevDelta);
}

void geluErfForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "geluErfForward", "-DUSE_GELUERF", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void geluErfBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "geluErfBackward", "-DUSE_GELUERF", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       in,
                       deltas,
                       prevDelta);
}

void geluTanhForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "geluTanhForward", "-DUSE_GELU", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void geluTanhBackward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getActivationBackwardKernel(manager, "geluTanhBackward", "-DUSE_GELU", caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)height,
                       (cl_int)width,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       in,
                       deltas,
                       prevDelta);
}

void dropoutForward(OpenCLKernelManager& manager,
                    const Name& caller,
                    size_t batch,
                    size_t depth,
                    size_t height,
                    size_t width,
                    dtype scale,
                    const cl::Buffer& in,
                    const cl::Buffer& proba,
                    cl::Buffer& out)
{
    auto kernel = getDropoutForwardKernel(manager, caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)((width + 3) / 4),
                       (cl_int)height,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       scale,
                       in,
                       proba,
                       out);
}

void dropoutBackward(OpenCLKernelManager& manager,
                     const Name& caller,
                     size_t batch,
                     size_t depth,
                     size_t height,
                     size_t width,
                     dtype scale,
                     const cl::Buffer& deltas,
                     const cl::Buffer& proba,
                     cl::Buffer& prevDelta)
{
    auto kernel = getDropoutBackwardKernel(manager, caller);
    manager.callKernel(kernel,
                       cl::NDRange{ (width + 3) / 4, height, batch * depth },
                       caller,
                       (cl_int)((width + 3) / 4),
                       (cl_int)height,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       (cl_int)height,
                       (cl_int)width,
                       0,
                       0,
                       scale,
                       deltas,
                       proba,
                       prevDelta);
}

void hsigmoidForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "sigmoidForward", "-DUSE_HSIGMOID", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void hsigmoidBackward(OpenCLKernelManager& manager, const Name& caller, size_t size, dtype leftDivisor, dtype rightDivisor, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getHSigmoidBackwardKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ 1, 1, 1 }, caller, (cl_int)size, leftDivisor, rightDivisor, in, deltas, prevDelta);
}

void hswishForward(OpenCLKernelManager& manager, const Name& caller, size_t batch, size_t depth, size_t height, size_t width, const cl::Buffer& in, cl::Buffer& out)
{
    auto kernel = getActivationForwardKernel(manager, "swishForward", "-DUSE_HSWISH", caller);
    manager.callKernel(
        kernel, cl::NDRange{ (width + 3) / 4, height, batch * depth }, caller, (cl_int)height, (cl_int)width, (cl_int)height, (cl_int)width, 0, 0, (cl_int)height, (cl_int)width, 0, 0, in, out);
}

void hswishBackward(OpenCLKernelManager& manager, const Name& caller, size_t size, dtype a, dtype b, dtype c, const cl::Buffer& in, const cl::Buffer& deltas, cl::Buffer& prevDelta)
{
    auto kernel = getHSwishBackwardKernel(manager, caller);
    manager.callKernel(kernel, cl::NDRange{ 1, 1, 1 }, caller, (cl_int)size, a, b, c, in, deltas, prevDelta);
}

} // namespace gpu
} // namespace raul
