// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import com.huawei.noah.*;

public final class TaskFlowTest {
    public static void GSR_TaskFlow(
        String rootPath, String outputPrefix, AffinityType affinity, int numThreads)
    {
        String modelPath = rootPath + "/model_zoo/caffe_models/gsr/gsr_flow.prototxt";
        int inputNum = 4;
        int outputNum = 1;
        String[] inputName = {"words", "positions", "token_type", "mask"};
        String[] outputName = {"normalized_token_embeddings_name"};
        int[] inputN = {1, 1, 1, 1};
        int[] inputC = {128, 128, 128, 128};
        int[] inputH = {1, 1, 1, 1};
        int[] inputW = {1, 1, 1, 1};
        DataType[] inputDataType = {
            DataType.UINT32, DataType.UINT32, DataType.UINT32, DataType.FP32};
        DataFormat[] inputDataFormat = {
            DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        int[] outputN = {1};
        int[] outputC = {312};
        int[] outputH = {1};
        int[] outputW = {1};
        DataType[] outputDataType = {DataType.FP32};
        DataFormat[] outputDataFormat = {DataFormat.NORMAL};
        TaskFlow taskFlow = new TaskFlow(modelPath, DataType.FP32, affinity, numThreads, false);
        String wordsPath = rootPath + "/testing_data/nlp/gsr/64.seq";
        float[] words = TestUtils.readSequenceDataFromFile(wordsPath, 0);
        float[] positions = new float[words.length];
        float[] token_type = new float[words.length];
        float[] mask = new float[words.length];
        for (int j = 0; j < words.length; j++) {
            positions[j] = (float)j;
            token_type[j] = (float)0;
            mask[j] = (float)1;
        }
        for (int j = 0; j < inputC.length; j++) {
            inputC[j] = words.length;
        }
        float[][] inputData = {words, positions, token_type, mask};
        int tasksSize = 4096;
        double startTime = TestUtils.getMillisTime();
        for (int i = 0; i < tasksSize; i++) {
            long task = taskFlow.taskBuild(modelPath, inputNum, inputN, inputC, inputH, inputW,
                inputName, inputDataType, inputDataFormat, inputData, outputNum, outputN, outputC,
                outputH, outputW, outputName, outputDataType, outputDataFormat);
            taskFlow.enqueue(task);
        }
        long[] tasks_addr = taskFlow.dequeue(true);
        double endTime = TestUtils.getMillisTime();
        System.out.println(outputPrefix + affinity + ", " + numThreads + " threads, GSR " +
            String.format("%.3f", (endTime - startTime) / tasksSize) + " ms/sequence, model " +
            modelPath);
        String resultPath = rootPath + "/testing_data/nlp/gsr/64_out.seq";
        float[][] resultData = {TestUtils.readSequenceDataFromFile(resultPath, 0)};
        for (int i = 0; i < tasksSize; i++) {
            BoltResult boltResult = taskFlow.getOutput(tasks_addr[i], outputNum, outputName);
            float[][] result = boltResult.getResultData();
            int[][] dimension = boltResult.getResultDimension();
            int length = TestUtils.verify(resultData, result, dimension, 0.1f);
            if (length == 0) {
                System.err.println("[ERROR] verify null data in GSR, in Java API test");
                System.exit(1);
            }
        }
        taskFlow.destructor();
    }

    public static void main(String[] args)
    {
        String outputPrefix = "[INFO] ";
        if (args.length > 0) {
            outputPrefix += args[0] + ", ";
        }
        String rootPath = args[1];
        GSR_TaskFlow(rootPath, outputPrefix, AffinityType.CPU_HIGH_PERFORMANCE, 16);
    }
}
