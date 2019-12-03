## How to customize Bolt for your model
Your own model may have some operators that have not been implemented in Bolt yet. Don't worry, because you can self-define the missing operators.

Here is an example of defining the "pooling" operator: 



#### 1 What you need to implement in the <u>**model-tools**</u> project

In `model-tools/include/model-tools.h`, you should add "pooling" to the list of operators

   ```c++
   typedef enum{
       ...,
       OT_Pooling    // add pooling data type
   }
   ```

Then, to define the attributes that "pooling" needs:

   ```c++
   typedef struct{
       U32 kernel_size;
       U32 stride;
       U32 padding;
       RoundMode rm;
       PoolingMode mode;
   } PoolingParamSpec;
   
   typedef enum{
       Max,
       Mean,
   } PoolingMode;    // defined in uni/include/type.h 
   
   typedef enum{
       CEIL,
       FLOOR,
   } RoundMode;    // defined in uni/include/type.h
   ```

Moreover, please check the converter tutorial of caffe or onnx or tflite, and add the converting part of "pooling" operator.

Take Caffe as an example, you need to add the following codes in `model-tools/src/caffe/converter_caffe.cpp`

   ```c++
   else if(layer.type == "Pooling"){
       PoolingParamSpec pps;
       pps.kernel_size = layer.pooling_param().kernel_size();
       pps.stride = layer.pooling_param().stride();
       bool global_pooling = layer.pooling_param().global_pooling();
       if(global_pooling){
           pps.kernel_size = 0;
           pps.stride = 1;
       }else{
           return NOT_MATCH;
       }
       pps.padding = layer.pooling_param().has_pad()?layer.pooling_param().pad():0;
       if(layer.pooling_param().has_round_mode() && layer.pooling_param().round_mode() == 1){
           pps.rm = FLOOR;
       }else{
           pps.rm = CEIL;
       }
       switch(layer.pooling_param().pool()){
           case caffe::PoolingParameter_PoolMethod_MAX: {
               pps.mode = Max;
               break;
           }
           case caffe::PoolingParameter_PoolMethod_AVE: {
           	pps.mode = Mean;
           	break;
           }
           default:
           	return NOT_MATCH;
       }
       ops_ptr[i].ps.pooling_param_spec = pps;
   }
   ```
You have completed the support of "pooling" within the model-tools project.



#### 2 What you need to implement in the **<u>tensor_computing</u>** project

In the tensor_computing project, you need to implement the actual "pooling" operations. 

- Firstly, define the interfaces of "pooling" in `tensor_computing/include/tensor_computing.h`. Each operator must have at least two functions: one to infer the output tensor size and one for the actual computation. The main function should take the system architecture as a parameter.


   ```c++
    EE pooling_infer_output_size(TensorDesc inputDesc, PoolingDesc poolingDesc, TensorDesc *outputDesc);

    EE pooling(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, const void* scale, TensorDesc outputDesc, void* output, Arch arch);
   ```

- Secondly, define the interface functions in `tensor_computing/src/pooling.cpp`. For the main function, please hide the actual computation in this level. Just call the functions from the lower level depending on the system architecture.


   ```c++
    #include <cmath>
    #include "sys.h"
    #include "type.h"
    #include "tensor_desc.h"
    #include "error.h"
    #include "tensor_computing.h"
    #include "cpu/general/tensor_computing_general.h"
    #include "cpu/arm/tensor_computing_arm.h"

    EE pooling_infer_output_size(TensorDesc inputDesc, PoolingDesc poolingDesc, TensorDesc* outputDesc)
    {
        if (nullptr == outputDesc) {
            CHECK_STATUS_WITH_RETURN(NULL_POINTER);
        }
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        U32 stride = poolingDesc.stride;
        U32 padding = poolingDesc.padding;
        U32 kernelSize = poolingDesc.kernelSize;
        RoundMode rm = poolingDesc.rm;
        U32 oh = 0, ow = 0;
        switch (rm) {
            case CEIL: {
                oh = (U32)(ceil((double(ih + 2.0 * padding - kernelSize) / stride))) + 1;
                ow = (U32)(ceil((double(iw + 2.0 * padding - kernelSize) / stride))) + 1;
                break;
            }
            case FLOOR: {
                oh = (U32)(floor((double(ih + 2.0 * padding - kernelSize) / stride))) + 1;
                ow = (U32)(floor((double(iw + 2.0 * padding - kernelSize) / stride))) + 1;
                break;
            }
            default: {
                CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
            }
        }
        *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);
        return SUCCESS;
    }

    EE pooling(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, const void* scale, TensorDesc outputDesc, void* output, Arch arch)
    {
        EE ret = SUCCESS;
        switch (arch) {
            case CPU_GENERAL:
                ret = pooling_general(inputDesc, input,
                                      poolingDesc,
                                      outputDesc, output);
                break;
            case ARM_A55:
                ret = pooling_arm(inputDesc, input,
                                  poolingDesc, scale,
                                  outputDesc, output);
                break;
            case ARM_A76:
                ret = pooling_arm(inputDesc, input,
                                  poolingDesc, scale,
                                  outputDesc, output);
                break;
            default:
                ret = NOT_SUPPORTED;
        }
        return ret;
    }
   ```

- Thirdly, we encourage you to first implement a naive version and make sure your logical understanding is correct. You need to add a function declaration in `src/cpu/general/tensor_computing_general.h`:

   ```c++
   EE pooling_general(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, TensorDesc outputDesc, void* output);
   ```

   Then add `src/cpu/general/pooling.cpp` with your function definition.

   ```c++
   #include "type.h"
   #include "tensor_desc.h"
   #include "error.h"
   #include "tensor_computing_type.h"
   #include "cpu/general/tensor_computing_general.h"
   #include "cpu/general/common_general.h"
   
   EE pooling(F16 *input, F16* output,
                U32 in, U32 ic, U32 ih, U32 iw,
                U32 stride, U32 padding, U32 kernel_H, U32 kernel_W,
                PoolingMode pm, RoundMode rm,
                U32 alignSize)
   {
       F16 F16_MIN = -65504;
       U32 oh = 0, ow = 0;
       if (rm == CEIL) {
           oh = (U32)(ceil((double(ih + 2.0 * padding - kernel_H) / stride))) + 1;
           ow = (U32)(ceil((double(iw + 2.0 * padding - kernel_W) / stride))) + 1;
       }
       if (rm == FLOOR) {
           oh = (U32)(floor((double(ih + 2.0 * padding - kernel_H) / stride))) + 1;
           ow = (U32)(floor((double(iw + 2.0 * padding - kernel_W) / stride))) + 1;
       }
   
       assert(ic % alignSize == 0);
       ic = ic / alignSize;
   
       for (U32 n=0; n<in; n++){
           for (U32 c=0; c<ic; c++){
               for (U32 j=0; j<alignSize; j++){
                   for (I32 h=0; h<(I32)oh; h++){
                       for (I32 w=0; w<(I32)ow; w++){
                           int hstart = int(h * stride - padding);
                           int wstart = int(w * stride - padding);
                           int hend = hstart + kernel_H;
                           int wend = wstart + kernel_W;
                           hstart = (hstart < 0) ? 0 : hstart;
                           wstart = (wstart < 0) ? 0 : wstart;
                           hend = (hend > (int)ih) ? ih : hend;
                           wend = (wend > (int)iw) ? iw : wend;
                           float poolSize = (hend - hstart)*(wend - wstart);
   
                           F16 value;
                           switch(pm){
                               case Max:
                                   value = F16_MIN;
                                   break;
                               case Mean:
                                   value = 0;
                                   break;
                               default:
                                   return NOT_SUPPORTED;
                           }
                           for (int x = hstart; x < hend; x++) {
                               for (int y = wstart; y < wend; y++) {
                                   U32 in_off = ((((n*ic + c)*ih) + x)*iw + y)*alignSize + j;
                                   switch(pm){
                                       case Max:
                                           value = (value > input[in_off]) ? value : input[in_off];
                                           break;
                                       case Mean:
                                           value += input[in_off];
                                           break;
                                       default:
                                           return NOT_SUPPORTED;
                                   }
                               }
                           }
                           switch(pm){
                               case Max:
                                   value = value;
                                   break;
                               case Mean:
                                   value = value / poolSize;
                                   break;
                               default:
                                   return NOT_SUPPORTED;
                           }
   
                           U32 out_off = ((((n*ic + c)*oh) + h)*ow + w)*alignSize + j;
                           output[out_off] = value;
                       }
                   }
               }
           }
       }
       return SUCCESS;
   }
   
   EE pooling_general(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, TensorDesc outputDesc, void* output)
   {
       if (nullptr == input || nullptr == output) {
           CHECK_STATUS_WITH_RETURN(NULL_POINTER);
       }
       DataType idt, odt;
       DataFormat idf, odf;
       U32 in = 0, ic = 0, ih = 0, iw = 0,
           on = 0, oc = 0, oh = 0, ow = 0;
       CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
       CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
   
       if (in != on || ic != oc) {
           CHECK_STATUS_WITH_RETURN(NOT_MATCH);
       }
       if (idf != DF_NCHWC8 || odf != idf) {
           CHECK_STATUS_WITH_RETURN(NOT_MATCH);
       }
   
       U32 stride = poolingDesc.stride;
       U32 padding = poolingDesc.padding;
       U32 kernelSize = poolingDesc.kernelSize;
   
       EE ret = SUCCESS;
       switch (idt) {
           case DT_F16:
               pooling((F16*)input, (F16*)output,
                                in, ic, ih, iw,
                                stride, padding, kernelSize, kernelSize,
                                poolingDesc.pm, poolingDesc.rm,
                                8);
               break;
           default:
               return NOT_SUPPORTED;
       }
       return ret;
   }
   ```

   After this, you can already verify the operator by running the model (after finishing the update of engine). For better performance, you are encouraged to implement optimized versions for different architectures (e.g. ARM CPU) and different inference precisions (fp16, int8, binary). Please refer to `src/cpu/arm/` for more details.

   You are also encouraged to create a unit test to compare the optimized versions with the naive version. Please refer to `tests/test_pooling.cpp` for more details.



#### 3 What you need to implement in <u>**engine**</u> project

- In the engine project , you need to add the "pooling" operator as a class in `include/pooling.hpp`.

```cpp
#ifndef _POOLING_H
#define _POOLING_H
#include <math.h>
#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "model_tools.h"

template <Arch A>
class Pooling: public Operator<A> {
public:

/**
 * @param mode
 * @param ks
 * @param stride
 * @param padding
 * @param name
 */
    Pooling(PoolingMode mode, U32 ks, U32 stride, U32 padding, RoundMode rm)
    {
        this->mode = mode;
        this->kernelSize = ks;
        this->stride = stride;
        this->padding = padding;
        this->rm = rm;
        this->set_op_type(OT_Pooling);
    }

    PoolingDesc create_PoolingDesc(PoolingMode pm, U32 stride, U32 padding, U32 kernelSize, RoundMode rm)
    {
        PoolingDesc poolingDesc;
        poolingDesc.pm = pm;
        poolingDesc.stride = stride;
        poolingDesc.padding = padding;
        poolingDesc.kernelSize = kernelSize;
        poolingDesc.rm = rm;
        return poolingDesc;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        PoolingDesc poolingDesc = create_PoolingDesc(this->mode, this->stride, this->padding, this->kernelSize, this->rm);
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        Arch a = ARM_A55;

        F16 scales[2];
        if (DT_I8 == inputDesc.dt) {
            scales[0] = inputTensor.get_scale();
        }
        pooling(inputDesc, inputTensor.get_val().get(),
                poolingDesc, scales,
                outputDesc, outputTensor.get_val().get(),
                a);
        if (DT_I8 == inputDesc.dt) {
            outputTensor.set_scale(scales[1]);
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    void set_kernelSize(U32 globalKernelSize) {    
        this->kernelSize = globalKernelSize;
    }

    void set_stride(U32 globalStride) {
        this->stride = globalStride;
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        auto inDim = inDims[0];
        DataType dt;
        DataFormat df;
        U32 width ;
        U32 height;
        U32 numChannels;
        U32 num;
        EE ret = tensor4dGet(inDim, &dt, &df, &num, &numChannels, &height, &width);

        TensorDesc inputDesc = tensor4df(dt, df, num, numChannels, height, width);
        if (this->kernelSize == 0) {
            set_kernelSize(height);
            set_stride(1);
        }
        PoolingDesc poolingDesc = create_PoolingDesc(this->mode, this->stride, this->padding, this->kernelSize, this->rm);
        ret = pooling_infer_output_size(inputDesc, poolingDesc, &((*outDims)[0]));
        if (ret != SUCCESS) {
            std::cerr << "[ERROR] pooling_infer_output_size() failed " << ret << std::endl;
        }
        return ret;
    }


private:
    PoolingMode mode;
    RoundMode rm;
    U32 kernelSize;
    U32 stride;
    U32 padding;
};

#endif //_POOLING_H

```

 

Following the aboves steps, you can add support for any operator in Bolt. We welcome you as a contributor and hope you enjoy it.

If you would like to implement a new category of models, you can refer to `engine/src/cnn.cpp` and `engine/src/sequential.cpp` . It may be a challenge but it will also be worthwhile to conquer.

