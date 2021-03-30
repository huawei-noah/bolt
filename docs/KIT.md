# Contents
&nbsp;&nbsp;&nbsp;&nbsp;[Overview](#overview)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[iOS Overview](#ios-overview)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Android Overview](#android-overview)  
&nbsp;&nbsp;&nbsp;&nbsp;[Examples](#examples)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Image Classification](#image-classification)  

# Overview

Kit is an experimental feature based on [Flow](DEVELOPER.md), which aims to simplify the integration of bolt into applications.
At this stage we are still rapidly exploring different designs. In the long run we want to provide symmetrical APIs for different platforms including iOS, Android, etc.
In the [kit](../kit) directory, you can find the available demo project. In order to use the demo, bolt should be compiled first and some headers and libraries need to be installed into the project, which is also taken care of in [install.sh](../install.sh).

- ## iOS Overview

  iOS demo is using the Objective-C Language and the C++ API of Flow. Mainbody of the codes is in [ViewController.mm](../kit/iOS/ImageClassification/ImageClassification/ViewController.mm). There are some notes regarding iOS kits:

  - Compilation flags. The C++ API of Flow requires quite a few headers, and some compilation flags need to be set. For convenience, you can include [kit_flags.h](../kit/iOS/image_classification/ImageClassificationDemo/libbolt/headers/kit_flags.h) before including flow.h.
  - Model path in flow prototxt. Flow reads the model paths in prototxt in order to locate the models. On iOS, however, the exact storage path for model files is dynamically determined. ViewController.mm demonstrates how to update prototxt with the new model path.

- ## Android Overview

  Android demo is using the C++ API of Flow via simple JNI. Mainbody of the codes is in [native-lib.cpp](../kit/Android/ImageClassification/app/src/main/cpp/native-lib.cpp).
  
  - Compilation flags. Similar to iOS, some compilation flags are also set in [kit_flags.h](../kit/Android/ImageClassification/app/src/main/cpp/libbolt/headers/kit_flags.h).
  - GPU usage. The current project demonstrates CPU inference. We are still in the middle of refactoring the memory API, and when it completes the GPU usage will be symmetrical to CPU. To prevent careless mistakes, the project will only be set up when GPU compilation is off.

# Examples

- ## Image Classification

  <div align=center><img src="images/ImageClassification.PNG" width = 30% height = 30% /></div>

  The demo takes video input from camera, and uses [GhostNet](https://github.com/huawei-noah/ghostnet) model trained on ImageNet. Given the same FLOPs, GhostNet shows a clear advantage over other lightweight CNNs. The models that we provide are trained with width as 1.0 on TensorFlow, which reaches a TOP1 accuracy of 74%.

  You can easily switch to other models trained on other datasets, following the steps below. As a tutorial, we will show how to change the model to the FP16 GhostNet that is also included in the project (kit/models). Tested with single thread on our iPhone SE, switching to FP16 GhostNet allows the processing of each 224x224 image frame in under 9 ms as shown in the figure above. You can try other models if your device is older than iPhone X and thus not in ARMv8.2 architecture.

  0. In [image_classification.prototxt](../kit/iOS/ImageClassification/ImageClassification/libbolt/image_classification.prototxt), you can see that the Inference node includes a path to ghostnet_f32.bolt. Actually, it is not necessary to change this path to ghostnet_f16.bolt, because this path will be dynamically overwritten as explained above. We will show how to switch to FP16 in Step 1.

        **In the following steps, if the file name is not specified, please check ViewController.mm.**

  1. Switch to FP16 model. Change code to:

      ```
      NSString *boltPath=[[NSBundle mainBundle]pathForResource:@"ghostnet_f16" ofType:@"bolt"];
      ```
    
      Please also change the variable inferencePrecision to DT_F16.

  2. Adjust the pixelProcess function, which is registered as the preprocessing function for the Inference node. For FP16 inference, actual input to the model should be in FP16:

      ```
      F16 *oneArr = (F16 *)((CpuMemory *)outputs["input:0"]->get_memory())->get_ptr();
      ```

      If you are using your own model, change "input:0" to the name of your model input tensor.

      The provided Ghostnet requires input pixels organized as BGRBGRBGR... Adjust accordingly if your other model is trained with different preprocessing (i.e. normalizing each channel).

  3. Adjust the postProcess function, which is registered as the postprocessing function for the Inference node. For FP16 inference, the output score is also in FP16:

      ```
      F16 *score1000 =(F16 *)((CpuMemory *)inputs[boltModelOutputName]->get_memory())->get_ptr();
      ```

      If necessary, change boltModelOutputName to the name of your model output tensor. If your model is not trained on ImageNet, there may not be 1000 scores. You may also change the topK variable.

  4. If necessary, replace imagenet_classes.txt. Add codes to handle the class index numbers that Flow outputs.

        *NOTE: Android can also follow the above steps and make similar modifications.*

- ## Camera Enlarge

  <div align=center><img src="images/CameraEnlarge.PNG" width = 30% height = 30% /></div>

  The demo takes video input from camera, 32 pixels x 32 pixels, and uses [ESR_EA](https://github.com/huawei-noah/vega/blob/master/docs/en/algorithms/esr_ea.md) model to enlarge input image to 64 pixels x 64 pixels.

  You can easily switch to other models trained on other datasets, following the steps below. As a tutorial, we will show how to change the model to the FP16 ESR_EA that is also included in the project (kit/models).

  0. Similar with Image Classification

  1. Similar with Image Classification
  2. Adjust the pixelProcess function, which is registered as the preprocessing function for the Inference node. For FP16 inference, actual input to the model should be in FP16:

     ```
     F16 *oneArr = (F16 *)((CpuMemory *)outputs["input.1"]->get_memory())->get_ptr();
     ```

     If you are using your own model, change "input.1" to the name of your model input tensor.
     The provided Ghostnet requires input pixels organized as RGBRGBRGB... Adjust accordingly if your other model is trained with different preprocessing (i.e. normalizing each channel).
     
  3. Adjust the postProcess function, which is registered as the postprocessing function for the Inference node. For FP16 inference, the output pixel data is also in FP16,Process the data, assign values ​​less than 1 to 0, assign values ​​greater than 255 to 255, and then split and reorganize the data:

     ```
     F16 *rgbData =(F16 *)((CpuMemory *)inputs[boltModelOutputName]->get_memory())->get_ptr();
     F16 *rArr=(F16*)malloc(sizeof(F32*)*imgHeight*2*imgWidth*2);
     F16 *gArr=(F16*)malloc(sizeof(F32*)*imgHeight*2*imgWidth*2);
     F16 *bArr=(F16*)malloc(sizeof(F32*)*imgHeight*2*imgWidth*2);
     for (int i = 0; i <(imgHeight*2)*(imgWidth*2)*3; i++) {
         if(rgbData[i]<=1) {
             int a=0;
             rgbData[i]=a;
         }else if (rgbData[i]>255) {
             int b=255;
             rgbData[i]=b;
         }
    
         if (i<(imgHeight*2)*(imgWidth*2)) {
             gArr[i]=rgbData[i];
         } else if(i<(imgHeight*2)*(imgWidth*2)*2) {
             bArr[i-(imgHeight*2)*(imgWidth*2)]=rgbData[i];
         } else {
             rArr[i-2*(imgHeight*2)*(imgWidth*2)]=rgbData[i];
         }
     }
     ``` 
