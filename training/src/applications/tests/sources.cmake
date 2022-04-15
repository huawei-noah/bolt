
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.10...3.20 FATAL_ERROR)

if (RAUL_TESTS_BUILD_ACTIVATIONS)
    target_sources(RaulTests PRIVATE
            tests/activations/Test_ActivationFunc_GeLU.cpp
            tests/activations/Test_ActivationFunc_Hsigmoid.cpp
            tests/activations/Test_ActivationFunc_Hswish.cpp
            tests/activations/Test_ActivationFunc_LeakyReLU.cpp
            tests/activations/Test_ActivationFunc_ReLU.cpp
            tests/activations/Test_ActivationFunc_Sigmoid.cpp
            tests/activations/Test_ActivationFunc_SoftPlus.cpp
            tests/activations/Test_ActivationFunc_Swish.cpp
            tests/activations/Test_ActivationFunc_Tanh.cpp
            tests/activations/Test_LogSoftMax.cpp
            tests/activations/Test_SoftMax.cpp
            )
endif ()

if (RAUL_TESTS_BUILD_INITIALIZERS)
    target_sources(RaulTests PRIVATE
            tests/initializers/Test_Initializer_ConstantInitializer.cpp
            tests/initializers/Test_Initializer_FuncCompare.cpp
            tests/initializers/Test_Initializer_RandomNormInitializer.cpp
            tests/initializers/Test_Initializer_RandomUniformInitializer.cpp
            tests/initializers/Test_Initializer_XavierInitializer.cpp
            )
endif ()

if (RAUL_TESTS_BUILD_LAYERS)
    target_sources(RaulTests PRIVATE
            tests/layers/Test_CNNAveragePool.cpp
            tests/layers/Test_CNNBatchNorm.cpp
            tests/layers/Test_CNNGlobalAverage.cpp
            tests/layers/Test_CNNLayer1D.cpp
            tests/layers/Test_CNNLayer2D.cpp
            tests/layers/Test_CNNLayerDepthwise.cpp
            tests/layers/Test_CNNMaxPool.cpp
            tests/layers/Test_CNNPaddingLayer.cpp
            tests/layers/Test_DataLayer.cpp
            tests/layers/Test_LayerNorm.cpp
            tests/layers/Test_Layer_AdditiveAttention.cpp
            tests/layers/Test_Layer_ArgMax.cpp
            tests/layers/Test_Layer_ArgMin.cpp
            tests/layers/Test_Layer_BahdanauMonotonicAttention.cpp
            tests/layers/Test_Layer_BatchExpander.cpp
            tests/layers/Test_Layer_BatchNorm.cpp
            tests/layers/Test_Layer_BiLSTM.cpp
            tests/layers/Test_Layer_Clamp.cpp
            tests/layers/Test_Layer_CumSum.cpp
            tests/layers/Test_Layer_Dropout.cpp
            tests/layers/Test_Layer_DynamicConvolutionAttention.cpp
            tests/layers/Test_Layer_DynamicDepthwiseConvolution2D.cpp
            tests/layers/Test_Layer_ElementWiseCompare.cpp
            tests/layers/Test_Layer_ElementWiseDiv.cpp
            tests/layers/Test_Layer_ElementWiseMax.cpp
            tests/layers/Test_Layer_ElementWiseMin.cpp
            tests/layers/Test_Layer_ElementWiseMul.cpp
            tests/layers/Test_Layer_ElementWiseSub.cpp
            tests/layers/Test_Layer_ElementWiseSum.cpp
            tests/layers/Test_Layer_Exp.cpp
            tests/layers/Test_Layer_FakeQuant.cpp
            tests/layers/Test_Layer_FixedBias.cpp
            tests/layers/Test_Layer_GaussianUpsampling.cpp
            tests/layers/Test_Layer_GRU.cpp
            tests/layers/Test_Layer_GRUCell.cpp
            tests/layers/Test_Layer_IndexFill.cpp
            tests/layers/Test_Layer_L2Norm.cpp
            tests/layers/Test_Layer_L2SquaredNorm.cpp
            tests/layers/Test_Layer_LSTM.cpp
            tests/layers/Test_Layer_LSTMCell.cpp
            tests/layers/Test_Layer_LocationSensitiveAttention.cpp
            tests/layers/Test_Layer_Log.cpp
            tests/layers/Test_Layer_NonZeroMask.cpp
            tests/layers/Test_Layer_RSqrt.cpp
            tests/layers/Test_Layer_RandomChoice.cpp
            tests/layers/Test_Layer_RandomSelect.cpp
            tests/layers/Test_Layer_RandomTensor.cpp
            tests/layers/Test_Layer_ReduceMax.cpp
            tests/layers/Test_Layer_ReduceMean.cpp
            tests/layers/Test_Layer_ReduceMin.cpp
            tests/layers/Test_Layer_ReduceNonZero.cpp
            tests/layers/Test_Layer_ReduceStd.cpp
            tests/layers/Test_Layer_ReduceSum.cpp
            tests/layers/Test_Layer_RepeatInterleave.cpp
            tests/layers/Test_Layer_Reverse.cpp
            tests/layers/Test_Layer_Roll.cpp
            tests/layers/Test_Layer_Round.cpp
            tests/layers/Test_Layer_Scale.cpp
            tests/layers/Test_Layer_Select.cpp
            tests/layers/Test_Layer_Splitter.cpp
            tests/layers/Test_Layer_Sqrt.cpp
            tests/layers/Test_Layer_Square.cpp
            tests/layers/Test_Layer_TacotronDataInitialization.cpp
            tests/layers/Test_Layer_TargetsReduction.cpp
            tests/layers/Test_Layer_Tile.cpp
            tests/layers/Test_Layer_ZeroOutput.cpp
            tests/layers/Test_Layer_Zoneout.cpp
            tests/layers/Test_Linear.cpp
            tests/layers/Test_MaskedFillLayer.cpp
            tests/layers/Test_MatMul.cpp
            tests/layers/Test_NetworkParams.cpp
            tests/layers/Test_ReshapeLayer.cpp
            tests/layers/Test_Slicing.cpp
            tests/layers/Test_Transpose.cpp
            tests/layers/Test_TransposedCNNLayer1D.cpp
            tests/layers/Test_TransposedCNNLayer2D.cpp
            )
endif ()

if (RAUL_TESTS_BUILD_CORE)
    target_sources(RaulTests PRIVATE
            tests/lib/Test_CIFAR_Dataset.cpp
            tests/lib/Test_Common.cpp
            tests/lib/Test_DataLoader.cpp
            tests/lib/Test_DataTransformations.cpp
            tests/lib/Test_Dataset.cpp
            tests/lib/Test_ElementSequence.cpp
            tests/lib/Test_ImageRescale.cpp
            tests/lib/Test_LoadData.cpp
            tests/lib/Test_LossScale.cpp
            tests/lib/Test_MNIST_Dataset.cpp
            tests/lib/Test_MemoryManager.cpp
            tests/lib/Test_MemoryManagerGPU.cpp
            tests/lib/Test_NNTest.cpp
            tests/lib/Test_Name.cpp
            tests/lib/Test_NameGenerator.cpp
            tests/lib/Test_Quantization.cpp
            tests/lib/Test_Random.cpp
            tests/lib/Test_Tensor.cpp
            tests/lib/Test_TensorGPU.cpp
            tests/lib/Test_Train.cpp
            tests/lib/Test_Yato.cpp
            tests/lib/gpu/Test_GpuAxpy.cpp
            tests/lib/gpu/Test_GpuGemm.cpp
            tests/lib/gpu/Test_GpuIm2Col.cpp
            tests/lib/gpu/Test_GpuIota.cpp
            tests/lib/gpu/Test_OpenCL.cpp
            tests/lib/workflow/Test_IntervalTree.cpp
            tests/lib/workflow/Test_Workflow.cpp
            tests/lib/workflow/Test_WorkflowAllocation.cpp
            tests/lib/workflow/Test_WorkflowBenchmarks.cpp
            tests/lib/workflow/Test_WorkflowCheckpointing.cpp
            tests/lib/workflow/Test_WorkflowCompression.cpp
            tests/lib/workflow/Test_WorkflowInterference.cpp
            tests/lib/workflow/Test_WorkflowPool.cpp
            tests/lib/workflow/Test_WorkflowOverrideLayerExecutionTarget.cpp
            tests/lib/workflow/Test_WorkflowTools.h
            )
endif ()

if (RAUL_TESTS_CONFIG_ENABLE_LIBJPG)
    target_sources(RaulTests PRIVATE
            tests/tools/LoadJpeg.h
            tests/lib/Test_CNNJpegOpen.cpp
            )
endif ()

if (RAUL_TESTS_BUILD_LOSS)
    target_sources(RaulTests PRIVATE
            tests/losses/Test_BinaryCrossEntropyLoss.cpp
            tests/losses/Test_CrossEntropyLoss.cpp
            tests/losses/Test_KLDivLoss.cpp
            tests/losses/Test_L1Loss.cpp
            tests/losses/Test_MSELoss.cpp
            tests/losses/Test_NLLLoss.cpp
            tests/losses/Test_SigmoidCrossEntropyLoss.cpp
            tests/losses/Test_SoftmaxCrossEntropyLoss.cpp
            tests/losses/Test_WeightedLoss.cpp
            )
endif ()

if (RAUL_TESTS_BUILD_META)
    target_sources(RaulTests PRIVATE
            tests/meta/Test_Meta_Layer_Recurrent.cpp
            tests/meta/Test_Meta_Layer_Sequential.cpp
            tests/meta/Test_Meta_Layer_Stacked.cpp
            )
endif ()

if (RAUL_TESTS_BUILD_OPTIMIZERS)
    target_sources(RaulTests PRIVATE
            tests/optimizers/Test_Optimizer.cpp
            tests/optimizers/Test_Optimizer_ASGD.cpp
            tests/optimizers/Test_Optimizer_Adadelta.cpp
            tests/optimizers/Test_Optimizer_Adagrad.cpp
            tests/optimizers/Test_Optimizer_Adam.cpp
            tests/optimizers/Test_Optimizer_AdamW.cpp
            tests/optimizers/Test_Optimizer_Adamax.cpp
            tests/optimizers/Test_Optimizer_LAMB.cpp
            tests/optimizers/Test_Optimizer_Momentum.cpp
            tests/optimizers/Test_Optimizer_Nesterov.cpp
            tests/optimizers/Test_Optimizer_RMSprop.cpp
            tests/optimizers/Test_Optimizer_Ranger.cpp
            tests/optimizers/Test_Optimizer_Rprop.cpp
            tests/optimizers/Test_Optimizer_SGD.cpp
            tests/optimizers/regularizers/Test_Regularizer_L1.cpp
            tests/optimizers/regularizers/Test_Regularizer_L2.cpp
            tests/optimizers/schedulers/Test_Scheduler.cpp
            tests/optimizers/schedulers/Test_Scheduler_CosineAnnealing.cpp
            tests/optimizers/schedulers/Test_Scheduler_Exponential.cpp
            tests/optimizers/schedulers/Test_Scheduler_Lambda.cpp
            tests/optimizers/schedulers/Test_Scheduler_WarmUp.cpp
            )
endif ()

if (RAUL_TESTS_BUILD_TOPOLOGIES)
    target_sources(RaulTests PRIVATE
            tests/topologies/TacotronTestTools.cpp
            tests/topologies/TacotronTrainingParams.cpp
            tests/topologies/Test_BERT.cpp
            tests/topologies/Test_MobilenetV2.cpp
            tests/topologies/Test_MobilenetV3.cpp
            tests/topologies/Test_NINCifar.cpp
            tests/topologies/Test_ResNet.cpp
            tests/topologies/Test_SSDWakeUp.cpp
            tests/topologies/Test_Tacotron.cpp
            tests/topologies/Test_TacotronMsk.cpp
            tests/topologies/Test_TacotronSpB.cpp
            tests/topologies/Test_Transformer.cpp
            tests/topologies/TacotronTestTools.h
            tests/topologies/TacotronTrainingParams.h
            )
endif ()

if (RAUL_TESTS_BUILD_POSTPROCESSING)
    target_sources(RaulTests PRIVATE
            tests/postprocessing/Test_PostProcessing_GradientClipping.cpp
            )
endif ()

target_sources(RaulTests PRIVATE
        tests/tools/TestTools.cpp
        tests/tools/TestTools.h
        tests/tools/callbacks/LayerTypeStatistics.h
        tests/tools/callbacks/MultiCallback.h
        tests/tools/callbacks/TensorChecker.h
        tests/tools/callbacks/TensorTracer.h
        )

target_sources(RaulTests PRIVATE
        tests/other/Test_CNNCIFAR.cpp
        tests/other/Test_CNNMnist.cpp
        tests/other/Test_CNNMnistDepthwise.cpp
        tests/other/Test_CNNResidualDownsampleMnist.cpp
        tests/other/Test_CNNResidualMnist.cpp
        tests/other/Test_CNNSkipResidualMnist.cpp
        tests/other/Test_MLPMnist.cpp
        )
if (RAUL_BUILD_BINDINGS)
    target_sources(RaulTests PRIVATE
            other/Test_Bindings.cpp)
endif ()

if (RAUL_BUILD_CONVERTER)
    target_sources(RaulTests PRIVATE
            tests/other/Test_LoadBoltModel.cpp
            tests/other/Test_ModelConverter.cpp
            )
endif ()

if (RAUL_BUILD_C_API)
    target_sources(RaulTests PRIVATE
            tests/lib/Test_API.cpp
            )

endif ()
