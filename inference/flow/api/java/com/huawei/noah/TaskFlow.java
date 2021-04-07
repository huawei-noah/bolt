/**
 * @file
 * @brief Java TaskFlow Class Document
 *
 * @copyright
 * @code
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * @endcode
 */
package com.huawei.noah;

import java.util.ArrayList;

public final class TaskFlow {
    static
    {
        loadLibrary("c++_shared", true);
        loadLibrary("OpenCL", true);
        loadLibrary("kernelsource", true);
        loadLibrary("BoltModel", false);
        loadLibrary("protobuf", false);
        loadLibrary("flow", false);
    }

    public TaskFlow()
    {
        this.flowAddr = 0;
    }

    public TaskFlow(String graphPath,
        DataType precision,
        AffinityType affinityPolicy,
        int cpuThreads,
        boolean useGPU)
    {
        this.flowAddr = createFlow(
            graphPath, precision.toString(), affinityPolicy.toString(), cpuThreads, useGPU);
    }

    public native int taskFlowRegisterFunction(String functionName, long function);

    public long taskBuild(String graphPath,
        int inputNumber,
        int[] inputN,
        int[] inputC,
        int[] inputH,
        int[] inputW,
        String[] inputNames,
        DataType[] inputDataType,
        DataFormat[] inputDataFormat,
        float[][] inputData,
        int outputNumber,
        int[] outputN,
        int[] outputC,
        int[] outputH,
        int[] outputW,
        String[] outputNames,
        DataType[] outputDataType,
        DataFormat[] outputDataFormat)
    {
        String[] inputDataTypeString = new String[inputNumber];
        String[] inputDataFormatString = new String[inputNumber];
        for (int i = 0; i < inputNumber; i++) {
            inputDataTypeString[i] = inputDataType[i].toString();
            inputDataFormatString[i] = inputDataFormat[i].toString();
        }
        String[] outputDataTypeString = new String[outputNumber];
        String[] outputDataFormatString = new String[outputNumber];
        for (int i = 0; i < outputNumber; i++) {
            outputDataTypeString[i] = outputDataType[i].toString();
            outputDataFormatString[i] = outputDataFormat[i].toString();
        }

        long task_addr = createTask(graphPath, inputNumber, inputN, inputC, inputH, inputW,
            inputNames, inputDataTypeString, inputDataFormatString, inputData, outputNumber, outputN,
            outputC, outputH, outputW, outputNames, outputDataTypeString, outputDataFormatString);
        return task_addr;
    }

    public void enqueue(long task_addr)
    {
        taskEnqueue(this.flowAddr, task_addr);
        this.tasksAddr.add(new Long(task_addr));
    }

    public long[] dequeue(boolean block)
    {
        long[] finished_tasks_addr = tasksDequeue(this.flowAddr, block);
        for (int i = 0; i < finished_tasks_addr.length; i++) {
            int index = this.tasksAddr.indexOf((new Long(finished_tasks_addr[i])));
            if (index != -1) {
                this.tasksAddr.remove(index);
            }
        }
        return finished_tasks_addr;
    }

    public BoltResult getOutput(long task_addr, int outputNumber, String[] outputNames)
    {
        return getTaskResult(
            task_addr, outputNumber, outputNames, BoltResult.class.getName().replace(".", "/"));
    }

    public void destructor()
    {
        if (this.flowAddr != 0) {
            destroyFlow(this.flowAddr);
            this.flowAddr = 0;
        }
        for (int i = 0; i < this.tasksAddr.size(); i++) {
            destroyTask(this.tasksAddr.get(i).longValue());
        }
        this.tasksAddr.clear();
    }

    private long flowAddr;

    private ArrayList<Long> tasksAddr = new ArrayList<Long>();

    private native long createFlow(
        String graphPath, String precision, String affinityPolicy, int cpuThreads, boolean useGPU);

    private native long createTask(String graphPath,
        int inputNumber,
        int[] inputN,
        int[] inputC,
        int[] inputH,
        int[] inputW,
        String[] inputNames,
        String[] inputDataType,
        String[] inputDataFormat,
        float[][] inputData,
        int outputNumber,
        int[] outputN,
        int[] outputC,
        int[] outputH,
        int[] outputW,
        String[] outputNames,
        String[] outputDataType,
        String[] outputDataFormat);

    private native void taskEnqueue(long flow_addr, long task_addr);

    private native long[] tasksDequeue(long flow_addr, boolean block);

    private native BoltResult getTaskResult(
        long task_addr, int outputNumber, String[] outputNames, String boltResultClassPath);

    private native void destroyFlow(long flow_Addr);

    private native void destroyTask(long task_addr);

    private static void loadLibrary(String libraryName, boolean optional)
    {
        try {
            System.loadLibrary(libraryName);
        } catch (UnsatisfiedLinkError e) {
            if (!optional) {
                e.printStackTrace();
            }
        }
    }
}
