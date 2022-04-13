# Contents
---
&nbsp;&nbsp;&nbsp;&nbsp;[Overview](#overview)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[iOS Overview](#ios-overview)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Android Overview](#android-overview)  
&nbsp;&nbsp;&nbsp;&nbsp;[Examples](#examples)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Image Classification](#image-classification)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Camera Enlarge](#camera-enlarge)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Semantics Analysis](#semantics-analysis)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Chinese Speech Recognition](#chinese-speech-recognition)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Face Detection](#face-detection)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Reading Comprehension](#reading-comprehension)  

# Overview
---
Kit is an experimental feature based on [Flow](DEVELOPER.md), which aims to simplify the integration of bolt into applications.
At this stage we are still rapidly exploring different designs. In the long run we want to provide symmetrical APIs for different platforms including iOS, Android, etc.
In the [kit](../kit) directory, you can find the available demo project. In order to use the demo, bolt should be compiled first and some headers and libraries need to be installed into the project, which is also taken care of in [install.sh](../install.sh).

- ### iOS Overview

  iOS demo is using the Objective-C Language and the C++ API of Flow. Mainbody of the codes is in [ViewController.mm](../kit/iOS/ImageClassification/ImageClassification/ViewController.mm). There are some notes regarding iOS kits:

  - Compilation flags. The C++ API of Flow requires quite a few headers, and some compilation flags need to be set. For convenience, you can include [kit_flags.h](../kit/iOS/image_classification/ImageClassificationDemo/libbolt/headers/kit_flags.h) before including flow.h.
  - Model path in flow prototxt. Flow reads the model paths in prototxt in order to locate the models. On iOS, however, the exact storage path for model files is dynamically determined. ViewController.mm demonstrates how to update prototxt with the new model path.

- ### Android Overview

  Android demo is using the C++ API of Flow via simple JNI. Mainbody of the codes is in [native-lib.cpp](../kit/Android/ImageClassification/app/src/main/cpp/native-lib.cpp) or [MainActivity.java](../kit/Android/ImageClassification/app/src/main/java/com/example/imageclassificationapp/MainActivity.java).
  
  - Compilation flags. Similar to iOS, some compilation flags are also set in [kit_flags.h](../kit/Android/ImageClassification/app/src/main/cpp/libbolt/headers/kit_flags.h).
  - GPU usage. The current project demonstrates CPU inference. We are still in the middle of refactoring the memory API, and when it completes the GPU usage will be symmetrical to CPU. To prevent careless mistakes, the project will only be set up when GPU compilation is off.

# Examples
---

- ### Image Classification

  <div align=center><img src="images/ImageClassification.gif" width = 20% height = 20% /></div>

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

- ### Camera Enlarge

  <div align=center><img src="images/CameraEnlarge.PNG" width = 20% height = 20% /></div>

  The demo takes video input from camera, 32 pixels x 32 pixels, and uses [ESR_EA](https://github.com/huawei-noah/vega/blob/master/docs/en/algorithms/esr_ea.md) model to enlarge input image to 64 pixels x 64 pixels.

  You can easily switch to other models trained on other datasets, following the steps below. As a tutorial, we will show how to change the model to the FP16 ESR_EA that is also included in the project (kit/models).

  0. Similar with Image Classification.

  1. Similar with Image Classification.
  
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

- ### Semantics Analysis

  <div align=center><img src="images/SemanticsAnalysis.gif" width = 20% height = 20% /></div>

  The demo tokenize input words, and use [tinybert](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) model to do senmantic analysis.
   
  You can easily switch to other models trained on other datasets, following the steps below. As a tutorial, we will show how to change the model to the FP32 Tinybert that is also included in the project.
   
  0. Copy the path of the model file to the cache path so that the Jni method in the dynamic library can be called.

     ```
     copyAssetResource2File(MODEL, modelPath);
     ```
  
  1. set the input and output names and other input parameters according to your model to initialize BoltModel.
  
     ```
     int inputNum = 3;
     int outputNum = 1;
     String[] inputName = {"input_ids","position_ids","token_type_ids"};
     String[] outputName = {"logit"};
     int[] inputN = {1,1,1};
     int[] inputCMax = {64,64,64};
     int[] inputH = {1,1,1};
     int[] inputW = {1,1,1};
     DataType[] intputDataType = {DataType.INT32,DataType.INT32,DataType.INT32};
     DataFormat[] intputDataFormat = {DataFormat.NORMAL,DataFormat.NORMAL,DataFormat.NORMAL};
     BoltModel boltModel = new BoltModel(modelPath, AffinityType.CPU_HIGH_PERFORMANCE, inputNum, inputName, inputN,
             inputCMax, inputH, inputW, intputDataType, intputDataFormat, outputNum, outputName);
     ```
  
  2. Call the run method of the BoltModel class to obtain the output result. Tokenizers are the processed input data, and inputCActual is the actual length of the input data. Call getResultData of BoltResult class to get the analysis result, get the result array, two float data.
  
     ```
     float[][] tokenizers = appTokenizer.runTokenizer(sentence);
     int[] inputCActual = {tokenizers[0].length, tokenizers[1].length, tokenizers[2].length};

     BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputCActual, inputH,
             inputW, intputDataType, intputDataFormat, tokenizers);
     float[][] result = boltResult.getResultData();
     ```
  
  3. Obtain the analysis result by comparing the size of the two probabilities in the result array.
  
     ```
     if (result[0][0]>result[0][1]) {
        tvIntent.setText("negative");
     } else {
       tvIntent.setText("positive");
     }
     ```
  
- ### Chinese Speech Recognition

  <div align=center><img src="images/ChineseSpeechRecognition.gif" width = 20% height = 20% /></div>

  The demo recognizes the input Chinese speech, and uses the [ASR](https://github.com/huawei-noah/xxx) model to convert Chinese text.
    
  You can easily switch to other models trained on other datasets, following the steps below. As a tutorial, we will show how to change the model to the FP32 ASR that is also included in the project.
    
  0. Call the copyAssetAndWrite method to copy the path, and then change the path of the bin file and bolt model in the prototxt file to the copied path.
    
  1. Import flow_asr.h in native-lib, flow_asr defines the pre- and post-processing methods and the initialization of flow and the acquisition of results,add init method and get result method in native-lib.cpp.

     ```
     extern "C"
     JNIEXPORT void JNICALL
     Java_com_huawei_noah_MainActivity_initFlow(JNIEnv *env, jobject thiz, jstring encoder_path,
                                                jstring predic_path, jstring joint_path,
                                                jstring pinyin_path,jstring label_path) {
         encoderGraphPath = env->GetStringUTFChars(encoder_path, nullptr);
         predictionGraphPath = env->GetStringUTFChars(predic_path, nullptr);
         jointGraphPath = env->GetStringUTFChars(joint_path, nullptr);
         pinyin2hanziGraphPath = env->GetStringUTFChars(pinyin_path, nullptr);
         labelFilePath = env->GetStringUTFChars(label_path, nullptr);

         initASRFlow();
     }

     extern "C"
     JNIEXPORT jstring JNICALL
     Java_com_huawei_noah_MainActivity_runFlow(JNIEnv *env, jobject thiz, jstring wav_file_path) {
         std::string wavFilePath = env->GetStringUTFChars(wav_file_path, nullptr);
         std::string hanzi = runASRFlow(wavFilePath);
         return env->NewStringUTF(hanzi.c_str());
     }
     ```
    
  2. Call Jni method  initFlow.
  
     ```
     initFlow(getCacheDir()+"/encoder_flow.prototxt",getCacheDir()+"/prediction_flow.prototxt",
                    getCacheDir()+"/joint_flow.prototxt",getCacheDir()+"/pinyin2hanzi_flow.prototxt",getCacheDir()+"/asr_labels.txt");
     
     ```
    
  3. Call Jni method  runFlow Incoming audio files in wav format get result.
  
     ```
     runFlow(wavFileName)
     ```
  
- ### Face Detection

  <div align=center><img src="images/20_bolt_face_detection.gif" width = 20% height = 20% /></div>
    
  The demo detects the input picture, and outputs A photo framed a human face.
    
  0. bolt path get Similar with Semantics.
    
  1. Call the getDetectionImgPath method Bitmap and model path to go directly to the detection result picture path.
  
     ```
     resultImgPath=boltResult.getDetectionImgPath(bitmap,boltPath);
     ```
    
  2. The parameters in the prior_boxes_generator method in the jni method initBolt are fixed input parameters of the model and cannot be changed.

     ```
     prior_boxes_generator(320,240,0.7,0.3);
     ```
     
- ### Reading Comprehension

<div align=center><img src="images/ReadingComprehension.gif" width = 20% height = 20% /></div>


The demo is to input a piece of content, and input a content-related question will output the corresponding answer

0. Call the copyAssetAndWrite method to copy the path, and the model path is used in the BoltModel class.

1. Incoming content and questions to obtain the input data required by the dynamic library.

 ```
 float[][] tokenizers = appTokenizer.runTokenizer(content.getText().toString(), question.getText().toString());
 ```
 
 2. set the input and output names and other input parameters according to your model to initialize BoltModel.
 
    ```
    BoltModel boltModel = new BoltModel(modelPath, AffinityType.CPU_HIGH_PERFORMANCE, inputNum, inputName, inputN,inputCMax, inputH, inputW, inputDatatype, inputDataFormat, outputNum, outputName);
    BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputCActual, inputH, inputW,
                                    inputDatatype, inputDataFormat, tokenizers);
                                    
    ```
    
    3. Call the run method of the BoltModel class to obtain the output result. Tokenizers are the processed input data, and inputCActual is the actual length of the input data. Call getResultData of BoltResult class to get the analysis result, get the result array, two float data.
    
       ```
       BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputCActual, inputH, inputW,
                                       inputDatatype, inputDataFormat, tokenizers);
       float[][] result = boltResult.getResultData();
       ```
       
       4. Call the getResultAnswer method to get the answer of the output result conversion
       
       ```
       String resultStr = getResultAnswer(result);
       
       ```