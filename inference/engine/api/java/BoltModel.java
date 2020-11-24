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

import java.io.File;
import java.io.FileNotFoundException;

/** affinity policy */
enum AffinityType {
    CPU_HIGH_PERFORMANCE,  ///< performance is high priority(use CPU big core)
    CPU_LOW_POWER,         ///< power is high priority(use CPU small core)
    GPU                    ///< use ARM MALI GPU
}

/** heterogeneous device type */
enum DeviceType {
    CPU_ARM_V7,   ///< ARMv7 CPU
    CPU_ARM_V8,   ///< ARMv8 CPU
    CPU_ARM_A55,  ///< ARM A55 CPU
    CPU_ARM_A76,  ///< ARM A76 CPU
    GPU_MALI,     ///< ARM MALI GPU
    CPU_X86_AVX2, ///< X86_64 AVX2 CPU
    CPU_SERIAL    ///< CPU serial
}

/** data precision */
enum DataType {
    FP32,   ///< 32 bit float
    FP16,   ///< 16 bit float
    INT32,  ///< 32 bit integer
    UINT32  ///< 32 bit unsigned char
}

/** multi-dimensions data format */
enum DataFormat {
    NCHW,   ///< batch->channel->high->width data order
    NHWC,   ///< batch->high->width->channel data order
    MTK,    ///< batch->time->unit data order
    NORMAL  ///< vectorize input of row major
}

public final class BoltModel implements Cloneable {
    private static void loadLibrary(String libraryAbsolutePath, boolean optional)
    {
        File file = new File(libraryAbsolutePath);
        if (file.exists()) {
            System.load(libraryAbsolutePath);
        } else {
            if (!optional) {
                System.err.println("[ERROR] unable to load " + libraryAbsolutePath);
            }
        }
    }

    static
    {
        String dir = System.getProperty("user.dir");
        loadLibrary(dir + "/libc++_shared.so", true);
        loadLibrary("/system/lib64/libOpenCL.so", true);
        loadLibrary(dir + "/libkernelsource.so", true);
        loadLibrary(dir + "/libBoltModel.so", false);
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

    private native long cloneResult(long IResult);

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

    private native long allocSpecificResultHandle(long modelAddr, int outputNum, String[] outputName);

    private native void setRuntimeDeviceJNI(int cpuId, String device);

    private native void setRuntimeDeviceDynamicJNI();

    private native void runModel(
        long modelAddr, long IResult, int inputNum, String[] inputName, float[][] inputData);

    private native BoltResult getOutput(long IResult);

    private native void freeResultHandle(long IResult);

    private native void destroyModel(long modelAddr);

    public String affinityMapping(AffinityType affinity)
    {
        String ret = "CPU_AFFINITY_HIGH_PERFORMANCE";
        if (affinity == AffinityType.CPU_HIGH_PERFORMANCE) {
            ret = "CPU_AFFINITY_HIGH_PERFORMANCE";
        } else if (affinity == AffinityType.CPU_LOW_POWER) {
            ret = "CPU_AFFINITY_LOW_POWER";
        } else if (affinity == AffinityType.GPU) {
            ret = "GPU";
        } else {
            System.err.println("[ERROR] unsupported CPU affinity in " + this.getClass().getName());
        }
        return ret;
    }

    public String deviceMapping(DeviceType device)
    {
        String ret = "CPU_ARM_V8";
        if (device == DeviceType.CPU_ARM_V7) {
            ret = "CPU_ARM_V7";
        } else if (device == DeviceType.CPU_ARM_V8) {
            ret = "CPU_ARM_V8";
        } else if (device == DeviceType.CPU_ARM_A55) {
            ret = "CPU_ARM_A55";
        } else if (device == DeviceType.CPU_ARM_A76) {
            ret = "CPU_ARM_A76";
        } else if (device == DeviceType.GPU_MALI) {
            ret = "GPU_MALI";
        } else if (device == DeviceType.CPU_X86_AVX2) {
            ret = "CPU_X86_AVX2";
        } else if (device == DeviceType.CPU_SERIAL) {
            ret = "CPU_SERIAL";
        } else {
            System.err.println("[ERROR] unsupported device in " + this.getClass().getName());
        }
        return ret;
    }

    public String dataTypeMapping(DataType data_type)
    {
        String ret = "FP32";
        if (data_type == DataType.FP32) {
            ret = "FP32";
        } else if (data_type == DataType.FP16) {
            ret = "FP16";
        } else if (data_type == DataType.INT32) {
            ret = "INT32";
        } else if (data_type == DataType.UINT32) {
            ret = "UINT32";
        } else {
            System.err.println("[ERROR] unsupported data type in " + this.getClass().getName());
        }
        return ret;
    }

    private String dataFormatMapping(DataFormat data_format)
    {
        String ret = "NCHW";
        if (data_format == DataFormat.NCHW) {
            ret = "NCHW";
        } else if (data_format == DataFormat.NHWC) {
            ret = "NHWC";
        } else if (data_format == DataFormat.MTK) {
            ret = "MTK";
        } else if (data_format == DataFormat.NORMAL) {
            ret = "NORMAL";
        } else {
            System.err.println("[ERROR] unsupported data format in " + this.getClass().getName());
        }
        return ret;
    }

    BoltModel()
    {
        this.modelAddr = 0;
        this.IResult = -1;
    }

    /**
     * @brief initial model and alloc memory
     * @param  modelPath       model file path of String type
     * @param  affinity        CPU affinity setting of AffinityType(enum) type
     * @param  device          heterogeneous device setting of DeviceType(enum) type
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
    BoltModel(String modelPath,
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
        String affinityString = affinityMapping(affinity);
        String[] inputDataTypeString = new String[inputNum];
        String[] inputDataFormatString = new String[inputNum];
        for (int i = 0; i < inputNum; i++) {
            inputDataTypeString[i] = dataTypeMapping(inputDataType[i]);
            inputDataFormatString[i] = dataFormatMapping(inputDataFormat[i]);
        }

        this.modelAddr = createModel(modelPath, affinityString);
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
     * @param  device          heterogeneous device setting of DeviceType(enum) type
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
    BoltModel(String modelPath,
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
        String affinityString = affinityMapping(affinity);
        String[] inputDataTypeString = new String[inputNum];
        String[] inputDataFormatString = new String[inputNum];
        for (int i = 0; i < inputNum; i++) {
            inputDataTypeString[i] = dataTypeMapping(inputDataType[i]);
            inputDataFormatString[i] = dataFormatMapping(inputDataFormat[i]);
        }

        this.modelAddr = createModel(modelPath, affinityString);
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
    protected Object clone() {
        BoltModel cloneModel = new BoltModel();
        if (0 != this.modelAddr) {
            cloneModel.modelAddr = cloneModel(this.modelAddr);
        } else {
            cloneModel.modelAddr = 0;
        }
        if (-1 != this.IResult) {
            cloneModel.IResult = cloneResult(this.IResult);
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
        String deviceString = deviceMapping(device);
        setRuntimeDeviceJNI(cpuId, deviceString);
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
        setRuntimeDeviceDynamicJNI();
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
        BoltResult boltResult = getOutput(this.IResult);
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
            inputDataTypeString[i] = dataTypeMapping(inputDataType[i]);
            inputDataFormatString[i] = dataFormatMapping(inputDataFormat[i]);
        }

        resizeModelInput(this.modelAddr, inputNum, inputName, inputN, inputC, inputH, inputW,
            inputDataTypeString, inputDataFormatString);
        runModel(this.modelAddr, this.IResult, inputNum, inputName, inputData);
        BoltResult boltResult = getOutput(this.IResult);
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
}
