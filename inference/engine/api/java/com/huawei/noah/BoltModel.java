/**
 * @file
 * @brief Java BoltModel Class Document
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

import java.io.FileNotFoundException;

public final class BoltModel implements Cloneable {
    static
    {
        loadLibrary("c++_shared", true);
        loadLibrary("OpenCL", true);
        loadLibrary("kernelsource", true);
        loadLibrary("BoltModel", false);
    }

    public BoltModel()
    {
        this.modelAddr = 0;
        this.IResult = -1;
    }

    /**
     * @brief initial model and alloc memory
     * @param  modelPath       model file path of String type
     * @param  affinity        CPU affinity setting of AffinityType(enum) type
     * @param  inputNum        the number of input data of int type
     * @param  inputName       the array of all input data's name of string type
     * @param  inputN          the array of all input data's n dimension of int type
     * @param  inputC          the array of all input data's c dimension of int type
     * @param  inputH          the array of all input data's h dimension of int type
     * @param  inputW          the array of all input data's w dimension of int type
     * @param  inputDataType   the array of all input data's data type of DataType(enum) type
     * @param  inputDataFormat the array of all input data's data format of DataFormat(enum) type
     *
     * @return
     *
     * @note destroy model when pipeline end
     * @code
     *     BoltModel example = BoltModel(...);
     *     ...
     *     example.estructor();
     * @endcode
     */
    public BoltModel(String modelPath,
        AffinityType affinity,
        int inputNum,
        String[] inputName,
        int[] inputN,
        int[] inputC,
        int[] inputH,
        int[] inputW,
        DataType[] inputDataType,
        DataFormat[] inputDataFormat)
    {
        String[] inputDataTypeString = new String[inputNum];
        String[] inputDataFormatString = new String[inputNum];
        for (int i = 0; i < inputNum; i++) {
            inputDataTypeString[i] = inputDataType[i].toString();
            inputDataFormatString[i] = inputDataFormat[i].toString();
        }

        this.modelAddr = createModel(modelPath, affinity.toString());
        if (0 != this.modelAddr) {
            prepareModel(this.modelAddr, inputNum, inputName, inputN, inputC, inputH, inputW,
                inputDataTypeString, inputDataFormatString);
            this.IResult = allocAllResultHandle(this.modelAddr);
        } else {
            this.IResult = -1;
            System.err.println("[ERROR] model cannot be created in " + this.getClass().getName());
        }
    }

    /**
     * @brief initial model and alloc memory, and the output is decided by user
     * @param  modelPath       model file path of String type
     * @param  affinity        CPU affinity setting of AffinityType(enum) type
     * @param  inputNum        the number of input data of int type
     * @param  inputName       the array of all input data's name of string type
     * @param  inputN          the array of all input data's n dimension of int type
     * @param  inputC          the array of all input data's c dimension of int type
     * @param  inputH          the array of all input data's h dimension of int type
     * @param  inputW          the array of all input data's w dimension of int type
     * @param  inputDataType   the array of all input data's data type of DataType(enum) type
     * @param  inputDataFormat the array of all input data's data format of DataFormat(enum) type
     * @param  outputNum       the number of output data of int type
     * @param  outputName      the array of all output data's name of string type
     *
     * @return
     *
     * @note destroy model when pipeline end
     * @code
     *     BoltModel example = BoltModel(...);
     *     ...
     *     example.estructor();
     * @endcode
     */
    public BoltModel(String modelPath,
        AffinityType affinity,
        int inputNum,
        String[] inputName,
        int[] inputN,
        int[] inputC,
        int[] inputH,
        int[] inputW,
        DataType[] inputDataType,
        DataFormat[] inputDataFormat,
        int outputNum,
        String[] outputName)
    {
        String[] inputDataTypeString = new String[inputNum];
        String[] inputDataFormatString = new String[inputNum];
        for (int i = 0; i < inputNum; i++) {
            inputDataTypeString[i] = inputDataType[i].toString();
            inputDataFormatString[i] = inputDataFormat[i].toString();
        }

        this.modelAddr = createModel(modelPath, affinity.toString());
        if (0 != this.modelAddr) {
            prepareModel(this.modelAddr, inputNum, inputName, inputN, inputC, inputH, inputW,
                inputDataTypeString, inputDataFormatString);
            this.IResult = allocSpecificResultHandle(this.modelAddr, outputNum, outputName);
        } else {
            this.IResult = -1;
            System.err.println("[ERROR] model cannot be created in " + this.getClass().getName());
        }
    }

    /**
     * @brief clone BoltModel
     *
     * @return cloneModel: shared weight with original model but has different tensor space
     */
    protected Object clone()
    {
        BoltModel cloneModel = new BoltModel();
        if (0 != this.modelAddr) {
            cloneModel.modelAddr = cloneModel(this.modelAddr);
        } else {
            cloneModel.modelAddr = 0;
        }
        if (-1 != this.IResult) {
            cloneModel.IResult = cloneResultHandle(this.IResult);
        } else {
            cloneModel.IResult = -1;
        }
        return cloneModel;
    }

    /**
     * @brief set process to run on specified CPU core
     * @param  cpuId         cpu core id(0, 1, 2...)
     * @param  device        cpu core architecture(ARM_A76)
     *
     * @return
     */
    public void setRuntimeDevice(int cpuId, DeviceType device) throws FileNotFoundException
    {
        if (0 == this.modelAddr) {
            throw new FileNotFoundException();
        }
        setRuntimeDeviceJNI(this.modelAddr, cpuId, device.toString());
    }

    /**
     * @brief set process cpu affinity according cpu average occupy
     *
     * @return
     */
    public void setRuntimeDeviceDynamic() throws FileNotFoundException
    {
        if (0 == this.modelAddr) {
            throw new FileNotFoundException();
        }
        setRuntimeDeviceDynamicJNI(this.modelAddr);
    }

    /**
     * @brief inference result from input
     * @param  inputNum     the number of input data of int type
     * @param  inputName    the array of all input data's name of string type
     * @param  inputData    the 2D array of all input data of float type
     *
     * @return BoltResult : the result class of bolt model after inference
     */
    public BoltResult run(int inputNum, String[] inputName, float[][] inputData)
    {
        if (0 == this.modelAddr) {
            return null;
        }
        runModel(this.modelAddr, this.IResult, inputNum, inputName, inputData);
        BoltResult boltResult =
            getOutput(this.IResult, BoltResult.class.getName().replace(".", "/"));
        return boltResult;
    }

    /**
     * @brief inference result from resized input
     * @param  inputNum        the number of input data of int type
     * @param  inputName       the array of all input data's name of String type
     * @param  inputN          the array of all input data's n dimension of int type
     * @param  inputC          the array of all input data's c dimension of int type
     * @param  inputH          the array of all input data's h dimension of int type
     * @param  inputW          the array of all input data's w dimension of int type
     * @param  inputDataType   the array of all input data's data type of DataType(enum) type
     * @param  inputDataFormat the array of all input data's data format of DataFormat(enum) type
     * @param  inputData       the 2D array of all input data of float type
     *
     * @return BoltResult : the result class of bolt model after inference
     */
    public BoltResult run(int inputNum,
        String[] inputName,
        int[] inputN,
        int[] inputC,
        int[] inputH,
        int[] inputW,
        DataType[] inputDataType,
        DataFormat[] inputDataFormat,
        float[][] inputData)
    {
        if (0 == this.modelAddr) {
            return null;
        }
        String[] inputDataTypeString = new String[inputNum];
        String[] inputDataFormatString = new String[inputNum];
        for (int i = 0; i < inputNum; i++) {
            inputDataTypeString[i] = inputDataType[i].toString();
            inputDataFormatString[i] = inputDataFormat[i].toString();
        }

        resizeModelInput(this.modelAddr, inputNum, inputName, inputN, inputC, inputH, inputW,
            inputDataTypeString, inputDataFormatString);
        runModel(this.modelAddr, this.IResult, inputNum, inputName, inputData);
        BoltResult boltResult =
            getOutput(this.IResult, BoltResult.class.getName().replace(".", "/"));
        return boltResult;
    }

    /**
     * @brief recycle memory and destroy model
     *
     * @return
     */
    public void destructor()
    {
        if (-1 != this.IResult) {
            freeResultHandle(this.IResult);
            this.IResult = -1;
        }
        if (0 != this.modelAddr) {
            destroyModel(this.modelAddr);
            this.modelAddr = 0;
        }
    }

    private long modelAddr;

    private long IResult;

    private native long createModel(String modelPath, String affinity);

    private native void prepareModel(long modelAddr,
        int inputNum,
        String[] inputName,
        int[] inputN,
        int[] inputC,
        int[] inputH,
        int[] inputW,
        String[] inputDataType,
        String[] inputDataFormat);

    private native long cloneModel(long modelAddr);

    private native long cloneResultHandle(long IResult);

    private native void resizeModelInput(long modelAddr,
        int inputNum,
        String[] inputName,
        int[] inputN,
        int[] inputC,
        int[] inputH,
        int[] inputW,
        String[] inputDataType,
        String[] inputDataFormat);

    private native long allocAllResultHandle(long modelAddr);

    private native long allocSpecificResultHandle(
        long modelAddr, int outputNum, String[] outputName);

    private native void setRuntimeDeviceJNI(long modelAddr, int cpuId, String device);

    private native void setRuntimeDeviceDynamicJNI(long modelAddr);

    private native void runModel(
        long modelAddr, long IResult, int inputNum, String[] inputName, float[][] inputData);

    private native BoltResult getOutput(long IResult, String boltResultPath);

    private native void freeResultHandle(long IResult);

    private native void destroyModel(long modelAddr);

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
