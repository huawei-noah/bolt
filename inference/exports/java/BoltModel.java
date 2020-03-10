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

/** CPU affinity policy */
enum AffinityType {
    HIGH_PERFORMANCE,    ///< performance is high priority(use big core)
    LOW_POWER            ///< power is high priority(use small core)
}

/** heterogeneous device type */
enum DeviceType {
    CPU,    ///< CPU
    GPU     ///< GPU
}

/** data precision */
enum DataType {
    FP32,    ///< 32 bit float
    FP16,    ///< 16 bit float
    INT32,   ///< 32 bit integer
    UINT32   ///< 32 bit unsigned char
}

/** multi-dimensions data format */
enum DataFormat {
    NCHW,    ///< batch->channel->high->width data order
    NHWC,    ///< batch->high->width->channel data order
    NORMAL   ///< vectorize input of row major
}

public final class BoltModel {
    private static void loadLibrary(String libraryAbsolutePath, boolean optional) {
        File file = new File(libraryAbsolutePath);
        if (file.exists()) {
	    System.load(libraryAbsolutePath);
        }
        else {
            if (!optional) {
                System.err.println("[ERROR] unable to load " + libraryAbsolutePath);
                System.exit(1);
            }
        }
    }

    static {
	loadLibrary("/data/local/tmp/CI/java/libOpenCL.so", true);
	loadLibrary("/data/local/tmp/CI/java/libkernelbin.so", true);
	loadLibrary("/data/local/tmp/CI/java/libBoltModel.so", false);
    }

    private long modelAddr;

    private long IResult;

    private native long model_create(String modelPath, String affinity, String device);

    private native void model_ready(long modelAddr, int num_input,
        String[] input_names, int[] n, int[] c, int[] h, int[] w, String[] dt_input, String[] df_input);

    private native void model_resize_input(long modelAddr, int num_input,
        String[] input_names, int[] n, int[] c, int[] h, int[] w, String[] dt_input, String[] df_input);

    private native long IResult_malloc_all(long modelAddr);

    private native long IResult_malloc_part(long modelAddr, int num_outputs, String[] outputNames);

    private native void model_run(long modelAddr, long IResult,
        int num_input, String[] input_names, float[][] inputData);

    private native BoltResult getOutput(long IResult);

    private native void IResult_free(long IResult);

    private native void destroyModel(long modelAddr);
    
    public String AffinityMapping(AffinityType affinity) {
        String ret = "HIGH_PERFORMANCE";
        if (affinity == AffinityType.HIGH_PERFORMANCE) {
	    ret = "HIGH_PERFORMANCE";
        } else if (affinity == AffinityType.LOW_POWER) {
	    ret = "LOW_POWER";
        } else {
            System.err.println("[ERROR] unsupported CPU affinity in " + this.getClass().getName());
            System.exit(1);
        }
        return ret;
    }
    
    public String DeviceMapping(DeviceType device) {
        String ret = "CPU";
        if (device == DeviceType.CPU) {
	    ret = "CPU";
	} else if (device == DeviceType.GPU) {
	    ret = "GPU";
        } else {
            System.err.println("[ERROR] unsupported device in " + this.getClass().getName());
            System.exit(1);
	} 
        return ret;
    }

    public String DataTypeMapping(DataType data_type) {
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
            System.exit(1);
	}
        return ret;
    }

    private String DataFormatMapping(DataFormat data_format) {
        String ret  = "NCHW";
	if (data_format == DataFormat.NCHW) {
	    ret = "NCHW";
	} else if (data_format == DataFormat.NHWC) {
	    ret = "NHWC";
        } else if (data_format == DataFormat.NORMAL) {
	    ret = "NORMAL";
	} else {
            System.err.println("[ERROR] unsupported data format in " + this.getClass().getName());
            System.exit(1);
	}
        return ret;
    }

    /**
    * @brief initial model and alloc memory
    * @param  modelPath     model file path of String type
    * @param  affinity      CPU affinity setting of AffinityType(enum) type
    * @param  device        heterogeneous device setting of DeviceType(enum) type
    * @param  num_input     the number of input data of int type
    * @param  input_names   the array of all input data's name of string type
    * @param  n		    the array of all input data's n dimension of int type
    * @param  c             the array of all input data's c dimension of int type
    * @param  h             the array of all input data's h dimension of int type
    * @param  w             the array of all input data's w dimension of int type    
    * @param  dts           the array of all input data's data type of DataType(enum) type 
    * @param  dfs           the array of all input data's data format of DataFormat(enum) type
    *
    * @return
    *
    * @note destroy model when pipeline end
    * @code
    *     BoltModel(...);
    *     ...
    *     Destructor();
    * @endcode
    */    
    BoltModel(String modelPath, AffinityType affinity, DeviceType device,
        int num_input, String[] input_names, int[] n, int[] c, int[] h, int[] w,
        DataType[] dts, DataFormat[] dfs)
    {
        String input_affinity = AffinityMapping(affinity);
	String input_device = DeviceMapping(device);
	String[] dts_str = new String[num_input];
	String[] dfs_str = new String[num_input];
	for (int i = 0; i < num_input; i++) {
	    dts_str[i] = DataTypeMapping(dts[i]);
	    dfs_str[i] = DataFormatMapping(dfs[i]);
	}

	this.modelAddr = model_create(modelPath, input_affinity, input_device);

	model_ready(this.modelAddr, num_input, input_names, n, c, h, w, dts_str, dfs_str);

	this.IResult = IResult_malloc_all(this.modelAddr);
    }

    /**
    * @brief initial model and alloc memory, and the output is decided by user
    * @param  modelPath     model file path of String type
    * @param  affinity      CPU affinity setting of AffinityType(enum) type
    * @param  device        heterogeneous device setting of DeviceType(enum) type
    * @param  num_input     the number of input data of int type
    * @param  input_names   the array of all input data's name of string type
    * @param  n             the array of all input data's n dimension of int type
    * @param  c             the array of all input data's c dimension of int type
    * @param  h             the array of all input data's h dimension of int type
    * @param  w             the array of all input data's w dimension of int type
    * @param  dts           the array of all input data's data type of DataType(enum) type
    * @param  dfs           the array of all input data's data format of DataFormat(enum) type
    * @param  num_output    the number of output data of int type
    * @param  output_names  the array of all output data's name of string type
    *
    * @return
    *
    * @note destroy model when pipeline end
    * @code
    *     BoltModel(...);
    *     ...
    *     Destructor();
    * @endcode
    */
    BoltModel(String modelPath, AffinityType affinity, DeviceType device,
        int num_input, String[] input_names,
        int[] n, int[] c, int[] h, int[] w,
        DataType[] dts, DataFormat[] dfs,
        int num_output, String[] output_names)
    {
        String input_affinity = AffinityMapping(affinity);
        String input_device = DeviceMapping(device);
        String[] dts_str = new String[num_input];
        String[] dfs_str = new String[num_input];
        for (int i = 0; i < num_input; i++) {
            dts_str[i] = DataTypeMapping(dts[i]);
            dfs_str[i] = DataFormatMapping(dfs[i]);
        }

        this.modelAddr = model_create(modelPath, input_affinity, input_device);

        model_ready(this.modelAddr, num_input, input_names, n, c, h, w, dts_str, dfs_str);

        this.IResult = IResult_malloc_part(this.modelAddr, num_output, output_names);
    }

    /**
    * @brief inference result from input
    * @param  num_input     the number of input data of int type
    * @param  input_names   the array of all input data's name of string type
    * @param  inputData     the 2D array of all input data of float type
    *
    * @return BoltResult : the result class of bolt model after inference
    */
    public BoltResult Run(int num_input, String[] input_names, float[][] inputData) {
        model_run(this.modelAddr, this.IResult, num_input, input_names, inputData);
        BoltResult bolt_result = getOutput(this.IResult);
        return bolt_result;
    }
 
    /**
    * @brief inference result from resized input
    * @param  num_input     the number of input data of int type
    * @param  input_names   the array of all input data's name of String type
    * @param  n             the array of all input data's n dimension of int type
    * @param  c             the array of all input data's c dimension of int type
    * @param  h             the array of all input data's h dimension of int type
    * @param  w             the array of all input data's w dimension of int type
    * @param  dts           the array of all input data's data type of DataType(enum) type
    * @param  dfs           the array of all input data's data format of DataFormat(enum) type
    * @param  inputData     the 2D array of all input data of float type
    *
    * @return BoltResult : the result class of bolt model after inference 
    */
    public BoltResult Run(
        int num_input, String[] input_names, int[] n, int[] c, int[] h, int[] w,
        DataType[] dts, DataFormat[] dfs,
        float[][] inputData) {
        String[] dts_str = new String[num_input];
        String[] dfs_str = new String[num_input];
        for (int i = 0; i < num_input; i++) {
            dts_str[i] = DataTypeMapping(dts[i]);
            dfs_str[i] = DataFormatMapping(dfs[i]);
        }
        model_resize_input(this.modelAddr, num_input, input_names, n, c, h, w, dts_str, dfs_str);
	model_run(this.modelAddr, this.IResult, num_input, input_names, inputData);
	BoltResult bolt_result = getOutput(this.IResult);
        return bolt_result;
    }

    /**
    * @brief recycle memory and destroy model
    *
    * @return
    */
    public void Destructor() {
	IResult_free(this.IResult);
	destroyModel(this.modelAddr);
    }
}
