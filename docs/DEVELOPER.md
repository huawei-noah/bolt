# Customize Models

- ### model-tools customization

  ​    In model-tools, you can define any operator for model conversion.

  1. Switch to code of the specific framework (caffe/onnx/tflite) you are working on;
  2.  Judge the op whether it is a weight-op or non-weight-op;
  3. Define the Operator parameter format;
  4. Extract the meta information of the op;
  5. Extract the weight data if the op is a weight-op, otherwise skip this step. 

  

  

  - [ ] caffe converter

    - add `pooling` in caffe converter

    (1) Switch to bolt/model-tools/src/caffe, which is the caffe converter for bolt;

    (2) Pooling is non-weight-op.

    (3) Define `pooling` parameter format.

    Add `pooling` definition of bolt in <u>bolt/model-tools/include/model_tools.h</u>

    ```
    // Addition begin
    typedef struct {
    	U32 kernel_size_h;
    	U32 kernel_size_w;
    	U32 stride_h;
    	U32 stride_w;
    	U32 padding_top;
    	U32 padding_bottom;
    	U32 padding_left;
    	U32 padding_right;
    	RoundMode rm;
    	PoolingMode mode;
    } PoolingParamSpec;
    // Addition end
    ```

    Modified "OperatorType" data structure  in <u>bolt/uni/include/op_type.h</u>

    ```
    typedef enum {
    ...
        OT_Pooling,    //  Addition 
    ...
    } OperatorType 
    ```

    Modified "inline const char* const* OperatorTypeName()" function in <u>bolt/uni/include/op_type.h</u> 

    ```
    inline const char* const* OperatorTypeName() {
        static const char* const names[] = {
            ...
            "OT_Pooling",    // Addition, please corresponds to the OperatorType
            ...
        }
    }
    ```

    (4) Extract the meta information of `pooling` operator in caffe.

    Modified the function named "OperatorType convert_caffe_type(std::string inputType)" in <u>bolt/model-tools/caffe/caffe_adaptee.h</u> .

    Add the caffe type mapping code as following:

    ```
    OperatorType convert_caffe_type(std::string inputType) {
        // Addition begin
        if (inputType == "Pooling") {    
            return OT_Pooling;    
        }    // Addition end
        else if (inputType ==  "Convolution") {
           ...
        } 
    }
    ```

    Extract the meta information of pooling operator from caffe model, add the function "ParameterSpec adapt_Pooling()  override" in <u>bolt/model-tools/caffe/caffe_adaptee.h</u>

    ```
    // Addition begin
    ParameterSpec adapt_Pooling() override {
        ParameterSpec curPs;
        PoolingParamSpec pps;
        if (layer.pooling_param().has_kernel_w() && layer.pooling_param().has_kernel_h()) {
            pps.kernel_size_w = layer.pooling_param().kernel_w();
            pps.kernel_size_h = layer.pooling_param().kernel_h();
        } else {
            pps.kernel_size_h = layer.pooling_param().kernel_size();
            pps.kernel_size_w = pps.kernel_size_h;
        }
        if (layer.pooling_param().has_stride_w() && layer.pooling_param().has_stride_h()) {
            pps.stride_w = layer.pooling_param().stride_w();
            pps.stride_h = layer.pooling_param().stride_h();
        } else {
            pps.stride_h = layer.pooling_param().stride();
            pps.stride_w = pps.stride_h;
        }
        bool global_pooling = layer.pooling_param().global_pooling();
        if (global_pooling) {
            pps.kernel_size_h = 0;
            pps.kernel_size_w = 0;
            pps.stride_h = 1;
            pps.stride_w = 1;
        }else {
            CHECK_REQUIREMENT(pps.kernel_size_h > 0);
        }
        if (layer.pooling_param().has_pad_w() && layer.pooling_param().has_pad_h()) {
            pps.padding_left = layer.pooling_param().pad_w();
            pps.padding_right = pps.padding_left;
            pps.padding_top = layer.pooling_param().pad_h();
            pps.padding_bottom = pps.padding_top;
        } else {
            pps.padding_top = layer.pooling_param().has_pad() ? layer.pooling_param().pad() : 0;
            pps.padding_bottom = pps.padding_top;
            pps.padding_left = pps.padding_top;
            pps.padding_right = pps.padding_top;
        }
            
        if (layer.pooling_param().has_round_mode() && layer.pooling_param().round_mode() == 1) {
            pps.rm = FLOOR;
        }else{
            pps.rm = CEIL;
        }
        switch (layer.pooling_param().pool()) {
            case caffe::PoolingParameter_PoolMethod_MAX: {
                pps.mode = POOLING_MAX;
                break;
            }
            case caffe::PoolingParameter_PoolMethod_AVE: {
            	pps.mode = POOLING_MEAN;
            	break;
            }
            default: {
            	std::cerr << "[ERROR] encounter unsupported Pooling method " << layer.pooling_param().pool() << std::endl;
            	break;
            }
        }
        curPs.pooling_spec = pps;
        return curPs;
    }
    // Addition end
    ```

     (5) Pooling is non-weight op, skip this step.

    

  - [ ] onnx converter

    - add `pooling` in onnx converter

    (1) Switch to bolt/model-tools/src/onnx, which is the onnx converter for bolt;

    (2) Pooling is non-weight-op;

    (3) Define `pooling` parameter format.

    Note: Definition actions same with add pooling in caffe converter step(3) . Please refer the former content. 

    (4) Extract the meta information of `pooling` operator in onnx.

    Modified the function named "OperatorType convert_onnx_type(std::string inputType)" in <u>bolt/model-tools/onnx/onnx_adaptee.h</u> .

    Add the onnx type mapping code as following:

    ```
    OperatorType convert_onnx_type(std::string inputType) {
        // Addition begin
        if (inputType == "AveragePool" || inputType == "MaxPool") {
            return OT_Pooling;
        } // Addition end
        else if (inputType == "Conv") {
            ...
        }
    }
    ```

    Extract the meta information of pooling operator from onnx model, add the function "ParameterSpec adapt_Pooling()  override" in <u>bolt/model-tools/onnx/onnx_adaptee.h</u>

    ```
    // Addition begin
    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec curPs;
        PoolingParamSpec pps;   
        std::string autoPad = get_node_str_attribute_by_name(node, "auto_pad"); 
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");
    
        if (op == "AveragePool" || op == "ReduceMean") {
            pps.mode = POOLING_MEAN;
        } else {
            pps.mode = POOLING_MAX;
        }
    
    	if (autoPad == "SAME_UPPER") {
    		pps.rm = CEIL;
    	} else {
    		pps.rm = FLOOR;
    	}
    
    	if (kernelShape.size() == 2) {
    		pps.kernel_size_h = kernelShape[0];
    		pps.kernel_size_w = kernelShape[1];
    	} else {
    		pps.kernel_size_h = 0;
    		pps.kernel_size_w = 0;
    		std::cerr << "[Info] pooling: kernel_size unknown. This could be global pooling." << std::endl;
    	}
    
    	if (strides.size() == 2) {
    		pps.stride_h = strides[0];
    		pps.stride_w = strides[1];
    	} else {
    		pps.stride_h = 0;
    		pps.stride_w = 0;
    		std::cerr << "[Info] pooling: stride unknown. This could be global pooling." << std::endl;
    	}
    
    	if (pads.size() == 4) {
    		pps.padding_top = pads[0];
    		pps.padding_bottom = pads[2];
    		pps.padding_left = pads[1];
    		pps.padding_right = pads[3];
    	} else {
    		pps.padding_top = 0;
    		pps.padding_bottom = 0;
    		pps.padding_left = 0;
    		pps.padding_right = 0;
    	}  
        curPs.pooling_spec = pps;
        return curPs;  
    }
    // Addition end
    ```

    (5) Pooling is non-weight op, skip this step.

    

  - [ ] tflite converter

    - add `pooling` in tflite converter

    (1) Switch to bolt/model-tools/src/onnx, which is the onnx converter for bolt;

    (2) Pooling is non-weight-op;

    (3) Define `pooling` parameter format.

    Note: Definition actions same with add pooling in caffe converter step(3) . Please refer the former content.

    (4) Extract the meta information of `pooling` operator in tflite.

    Modified the function named "OperatorType convert_tflite_type(std::string inputType)" in <u>bolt/model-tools/tflite/tflite_adaptee.h</u> .

    Add the tflite type mapping code as following:

    ```
    OperatorType convert_tflite_type(tflite::BuiltinOperator tfliteType) {
        // Addition begin
        if (tfliteType == tflite::BuiltinOperator_MAX_POOL_2D) {
            return OT_Pooling;
        } // Addition end
        else if (tfliteType == tflite::BuiltinOperator_CONCATENATION) {
            ...
        }
    }
    ```

    Extract the meta information of pooling operator from tflite model, add the function "ParameterSpec adapt_Pooling()  override" in <u>bolt/model-tools/tflite/tflite_adaptee.h</u>

    ```
    // Addition begin
    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec curPs;
        const auto& tflitePoolOption = ops[curIndex]->builtin_options.AsPool2DOptions();
        PoolingParamSpec poolingPs;
        poolingPs.kernel_size_h = tflitePoolOption->filter_height;
        poolingPs.kernel_size_w = tflitePoolOption->filter_width;
        poolingPs.stride_h = tflitePoolOption->stride_h;
        poolingPs.stride_w = tflitePoolOption->stride_w;
        poolingPs.padding_top = 0;
        poolingPs.padding_bottom = 0;
        poolingPs.padding_left = 0;
        poolingPs.padding_right = 0;
        poolingPs.rm = CEIL;
        if (opCode == tflite::BuiltinOperator_MAX_POOL_2D) {
        poolingPs.mode = POOLING_MAX;
        } else if (opCode == tflite::BuiltinOperator_AVERAGE_POOL_2D) {
        poolingPs.mode = POOLING_MEAN;
        }        
        curPs.pooling_spec = poolingPs;
        return curPs;
    }
    // Addition end
    ```

    (5) Pooling is non-weight op, skip this step.

     

- ### tensor_computing customization

  In tensor_computing, you can define any operator for operator computing process.

- ### inference customization

  In inference, you can define any operator for the inference of your model.
  
  1. Add the definition of the specific operator in <u>bolt/inference/include</u>;
  2. If the specific operator implement in CPU is different from its implement in GPU, implement should be  divided into CPU and GPU version. If the specific operator implement in CPU  is same with its implement in GPU, skip this step.
  
  
  
  - [ ] Example: add `pooling` operator in <u>bolt/inference</u>
  
    1.  Create `pooling.hpp` in <u>bolt/inference/include</u> , add the definition of `pooling` operator;
  
    ```
    // Addition begin
    #ifndef _POOLING_H
    #define _POOLING_H
    #include "operator.hpp"
    #include "tensor_computing.h"
    
    class Pooling: public Operator {
    public:
        Pooling(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW,
                U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm)
        {
            this->mode = mode;
            this->kernelSizeH = ksH;
            this->kernelSizeW = ksW;
            this->strideH = strideH;
            this->strideW = strideW;
            this->paddingT = paddingT;
            this->paddingB = paddingB;
            this->paddingL = paddingL;
            this->paddingR = paddingR;
            this->rm = rm;
        }
    
        PoolingDesc create_PoolingDesc(PoolingMode pm, U32 ksH, U32 ksW, U32 strideH, U32 strideW,
                                    U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm)
        {
            PoolingDesc poolingDesc;
            poolingDesc.pm = pm;
            poolingDesc.kernelSize_h = ksH;
            poolingDesc.kernelSize_w = ksW;
            poolingDesc.stride_h = strideH;
            poolingDesc.stride_w = strideW;
            poolingDesc.padding_top = paddingT;
            poolingDesc.padding_bottom = paddingB;
            poolingDesc.padding_left = paddingL;
            poolingDesc.padding_right = paddingR;
            poolingDesc.rm = rm;
            return poolingDesc;
        }
    
        void set_kernelSize(U32 globalKernelSizeH, U32 globalKernelSizeW) {
            this->kernelSizeH = globalKernelSizeH;
            this->kernelSizeW = globalKernelSizeW;
        }
    
        void set_stride(U32 globalStrideH, U32 globalStrideW) {
            this->strideH = globalStrideH;
            this->strideW = globalStrideW;
        }
    
        virtual void run() = 0;
        virtual EE infer_output_tensors_size(Vec<TensorDesc>, Vec<TensorDesc>*) = 0;
    #ifdef _USE_MALI
        virtual EE infer_output_tensors_size(Vec<TensorDesc>, Vec<TensorDesc>*, Vec<GCLMemDesc>*,  Vec<GCLMemDesc>*){return NOT_SUPPORTED;}
    #endif
    
    protected:
        PoolingMode mode;
        RoundMode rm;
        U32 kernelSizeH;
        U32 kernelSizeW;
        U32 strideH;
        U32 strideW;
        U32 paddingT;
        U32 paddingB;
        U32 paddingL;
        U32 paddingR;
    };
    
    #endif //_POOLING_H
    // Addition end 
    ```
  
    2. `pooling` operator implement in CPU is different from its implement in GPU. So `pooling` implement should be two version: CPU and GPU
  
    Create `pooling_cpu.hpp` and add `pooling` CPU implement in <u>bolt/inference/include/cpu</u> :
  
    ```
    #ifndef _POOLING_CPU_H
    #define _POOLING_CPU_H
    #include <math.h>
    #include "operator.hpp"
    #include "tensor_computing.h"
    #include "tensor_desc.h"
    #include "model_tools.h"
    #include "pooling.hpp"
    
    class PoolingCPU: public Pooling {
    public:
    
    /**
     * @param mode
     * @param ks
     * @param stride
     * @param padding
     * @param name
     */
        PoolingCPU(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW, U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm): 
            Pooling(mode, ksH, ksW, strideH, strideW, paddingT, paddingB, paddingL, paddingR, rm){}
    
        virtual void run() override
        {
            UTIL_TIME_TIC(__CLASS_FUNCTION__)
            Tensor inputTensor = this->inputTensors[0];
            TensorDesc inputDesc = inputTensor.get_desc();
            PoolingDesc poolingDesc = Pooling::create_PoolingDesc(this->mode, this->kernelSizeH, this->kernelSizeW, this->strideH, this->strideW,
                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->rm);
            Tensor outputTensor = this->outputTensors[0];
            TensorDesc outputDesc = outputTensor.get_desc();
            F16 scales[2];
            if (DT_I8 == inputDesc.dt) {
                scales[0] = inputTensor.get_scale();
            }
            CHECK_STATUS(pooling(inputDesc, inputTensor.get_val(),
                                 poolingDesc, scales,
                                 outputDesc, outputTensor.get_val(),
                                 this->schedule));
            if (DT_I8 == inputDesc.dt) {
                outputTensor.set_scale(scales[1]);
            }
    
            UTIL_TIME_TOC(__CLASS_FUNCTION__)
        }
    
    
        virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
        {
            auto inDim = inDims[0];
            DataType dt;
            DataFormat df;
            U32 width ;
            U32 height;
            U32 numChannels;
            U32 num;
            CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &num, &numChannels, &height, &width));
    
            TensorDesc inputDesc = tensor4df(dt, df, num, numChannels, height, width);
            if (this->kernelSizeH == 0 && this->kernelSizeW == 0) {
                Pooling::set_kernelSize(height, width);
                Pooling::set_stride(1, 1);
            }
            PoolingDesc poolingDesc = Pooling::create_PoolingDesc(this->mode, this->kernelSizeH, this->kernelSizeW, this->strideH, this->strideW,
                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->rm);
            CHECK_STATUS(pooling_infer_output_size(inputDesc, poolingDesc, &((*outDims)[0]), this->schedule));
            return SUCCESS;
        }
    
    };
    
    #endif //_POOLINGCPU_H
    ```
  
    Create `pooling_ocl.hpp` and add `pooling` GPU implement in bolt/inference/include/ocl :
  
    ```
    #ifndef _POOLING_OCL_H
    #define _POOLING_OCL_H
    #include <math.h>
    #include "operator.hpp"
    #include "tensor_computing.h"
    #include "tensor_desc.h"
    #include "model_tools.h"
    #include "pooling.hpp"
    
    class PoolingOCL: public Pooling {
    public:
    
    /**
     * @param mode
     * @param ks
     * @param stride
     * @param padding
     * @param name
     */
        PoolingOCL(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW, U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm):
             Pooling(mode, ksH, ksW, strideH, strideW, paddingT, paddingB, paddingL, paddingR, rm){}
    
        virtual void run() override
        {
            UTIL_TIME_TIC(__CLASS_FUNCTION__)
            Tensor inputTensor = this->inputTensors[0];
            TensorDesc inputDesc = inputTensor.get_desc();
            PoolingDesc poolingDesc = Pooling::create_PoolingDesc(this->mode, this->kernelSizeH, this->kernelSizeW, this->strideH, this->strideW,
                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->rm);
            Tensor outputTensor = this->outputTensors[0];
            TensorDesc outputDesc = outputTensor.get_desc();
            F16 scales[2];
            if (DT_I8 == inputDesc.dt) {
                scales[0] = inputTensor.get_scale();
            }
            CHECK_STATUS(pooling(inputDesc, inputTensor.get_val(),
                                 poolingDesc, scales,
                                 outputDesc, outputTensor.get_val(),
                                 this->schedule, &this->oclExtInfo));
            if (DT_I8 == inputDesc.dt) {
                outputTensor.set_scale(scales[1]);
            }
    
            UTIL_TIME_TOC(__CLASS_FUNCTION__)
        }
    
        virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
        {
            auto inDim = inDims[0];
            DataType dt;
            DataFormat df;
            U32 width ;
            U32 height;
            U32 numChannels;
            U32 num;
            CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &num, &numChannels, &height, &width));
    
            TensorDesc inputDesc = tensor4df(dt, df, num, numChannels, height, width);
            if (this->kernelSizeH == 0 && this->kernelSizeW == 0) {
                Pooling::set_kernelSize(height, width);
                Pooling::set_stride(1, 1);
            }
            PoolingDesc poolingDesc = Pooling::create_PoolingDesc(this->mode, this->kernelSizeH, this->kernelSizeW, this->strideH, this->strideW,
                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->rm);
            CHECK_STATUS(pooling_infer_output_size(inputDesc, poolingDesc, &((*outDims)[0]), this->schedule));
            return SUCCESS;
        }
    
        virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims, Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
        {
            auto inDim = inDims[0];
            DataType dt;
            DataFormat df;
            U32 width ;
            U32 height;
            U32 numChannels;
            U32 num;
            CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &num, &numChannels, &height, &width));
    
            this->oclExtInfo.maliInfo.gclmemInputDesc = &((*gclmemInputDesc)[0]);
            this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
            TensorDesc inputDesc = tensor4df(dt, df, num, numChannels, height, width);
            if (this->kernelSizeH == 0 && this->kernelSizeW == 0) {
                Pooling::set_kernelSize(height, width);
                Pooling::set_stride(1, 1);
            }
            
            PoolingDesc poolingDesc = Pooling::create_PoolingDesc(this->mode, this->kernelSizeH, this->kernelSizeW, this->strideH, this->strideW,
                this->paddingT, this->paddingB, this->paddingL, this->paddingR, this->rm);
            CHECK_STATUS(pooling_infer_output_size(inputDesc, poolingDesc, &((*outDims)[0]), this->schedule, &this->oclExtInfo));
            return SUCCESS;
        }
    
    
    private:
    };
    
    #endif //_POOLING_OCL_H
    ```

# How to contribute

- ### submit issue

  - [ ] question

    Submit any question you have encountered when you use Bolt. You can give feedback to us through committing issues. Refer to  https://github.com/huawei-noah/bolt/issues, create your new issue and submit it. The issue can be a bug in Bolt, a suggestion for Bolt, or anything you don't understand about Bolt.

    

  - [ ] feature request

    Submit any feature that you want but it has not been implemented in Bolt. We have created a [special issue](https://github.com/huawei-noah/bolt/issues/5) and you can leave a commit under this issue . We will seriously consider the needs of all users and continue to enrich the functions of Bolt.

- ### pull request

  - [ ] add a license
  
    Add the license at the head of your source codes indicating your codes will be open to all.
  
  - [ ] provide an executable unit test
  
    Fork [Bolt](https://github.com/huawei-noah/bolt) on your github account. Modify your code and make sure your code pass all testing cases. Commit the code and initiate a pull request on github.
  
     
  
    
