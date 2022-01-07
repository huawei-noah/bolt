// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#import "kit_flags.h"
#import "ultra_face.h"
#import "BoltResult.h"
#include "core.hpp"
#include "highgui.hpp"
#include "imgproc.hpp"
#include "opencv.hpp"

using namespace cv;
char *modelPath = (char *)"";
char *affinityPolicyName = (char *)"CPU_AFFINITY_HIGH_PERFORMANCE";
char *algorithmMapPath = (char *)"";

@implementation BoltResult


-(void)initBolt:(NSString *)model ResultPath:(NSString *)resultPath{
    _modelPathStr=model;
    _resultImgPath=resultPath;
    prior_boxes_generator(320, 240, 0.7, 0.3);
}

std::map<std::string, std::shared_ptr<Tensor>> get_output(
    std::shared_ptr<CNN> pipeline, std::string affinity)
{
    std::map<std::string, std::shared_ptr<Tensor>> outMap = pipeline->get_output();
    if (affinity == "GPU") {
#ifdef _USE_MALI
        for (auto iter : outMap) {
            Tensor result = *(iter.second);
            auto mem = (OclMemory *)result.get_memory();
            mem->get_mapped_ptr();
        }
#else
        UNI_WARNING_LOG("this binary not support GPU, please recompile project with GPU "
                        "compile options\n");
#endif
    }
    return outMap;
}

-(void)getResultImg:(UIImage *)image{
    modelPath=(char *)[_modelPathStr UTF8String];
    std::string output_video_path = [_resultImgPath UTF8String];

    cv::Mat img=[self cvMatFromUIImage:image];
    image_h = img.rows;  // global variable
    image_w = img.cols;  // global variable
    int img_channel = img.channels();  // local variable
    cv::Mat img_float;
    cv::Mat img_resize;
    std::vector<float> vec_original;
    std::shared_ptr<U8> input_ptr(new U8[image_h * image_w * img_channel * sizeof(float)]);
    float* vec_normalize = (float*)(input_ptr.get());
    auto pipeline = createPipeline(affinityPolicyName, modelPath, algorithmMapPath);
    std::map<std::string, TensorDesc> inputDescMap = pipeline->get_input_desc();
    auto item = inputDescMap.begin();
    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
    std::map<std::string, std::shared_ptr<Tensor>> outMap;

    while (1) {
        img.convertTo(img_float, CV_32F);
        cv::resize(img_float, img_resize, cv::Size(320, 240));    // magic number
        vec_original.assign((float*)img_resize.datastart, (float*)img_resize.dataend);
        int iter_index = 0;
        for (int i = img.channels() - 1; i >= 0; i--) {
            for (unsigned int j = 0; j < vec_original.size() / img.channels(); j++) {
                vec_normalize[iter_index] = (vec_original[j * img.channels() + i] - 127.0) / 128.0;
                iter_index++;
            }
        }
        model_tensors_input[item->first] = input_ptr;
        pipeline->set_input_by_assign(model_tensors_input);
        pipeline->run();
        outMap = get_output(pipeline, affinityPolicyName);
        std::vector<FaceInfo> bbox_collection;
        Tensor box_tensor = *(outMap["boxes"].get());
        Tensor score_tensor = *(outMap["scores"].get());
        bounding_boxes_generator(bbox_collection, box_tensor, score_tensor);
        std::vector<FaceInfo> bolt_final_result;
        nms(bbox_collection, bolt_final_result, hard_nms);
        for (unsigned int i = 0; i < bolt_final_result.size(); i++) {
            auto face = bolt_final_result[i];
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite(output_video_path, img);
        break;
    }
    std::cout << "result saved at " << output_video_path << std::endl;
}

#pragma mark-UIImage to cv::Mat
- (cv::Mat)cvMatFromUIImage:(UIImage *)image{
    BOOL hasAlpha = NO;
    CGImageRef imageRef = image.CGImage;
    CGImageAlphaInfo alphaInfo = (CGImageAlphaInfo)(CGImageGetAlphaInfo(imageRef) & kCGBitmapAlphaInfoMask);
    if (alphaInfo == kCGImageAlphaPremultipliedLast ||
        alphaInfo == kCGImageAlphaPremultipliedFirst ||
        alphaInfo == kCGImageAlphaLast ||
        alphaInfo == kCGImageAlphaFirst) {
        hasAlpha = YES;
    }
    
    cv::Mat cvMat;
    CGBitmapInfo bitmapInfo;
    CGFloat cols = CGImageGetWidth(imageRef);
    CGFloat rows = CGImageGetHeight(imageRef);
    
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(imageRef);
    size_t numberOfComponents = CGColorSpaceGetNumberOfComponents(colorSpace);
    if (numberOfComponents == 1){// check whether the UIImage is greyscale already
        cvMat = cv::Mat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
        bitmapInfo = kCGImageAlphaNone | kCGBitmapByteOrderDefault;
    }
    else {
        cvMat = cv::Mat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
        bitmapInfo = kCGBitmapByteOrder32Host;
        // kCGImageAlphaNone is not supported in CGBitmapContextCreate.
        // Since the original image here has no alpha info, use kCGImageAlphaNoneSkipLast
        // to create bitmap graphics contexts without alpha info.
        bitmapInfo |= hasAlpha ? kCGImageAlphaPremultipliedFirst : kCGImageAlphaNoneSkipFirst;
    }
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    bitmapInfo                  // Bitmap info flags
                                                    );
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), imageRef);     // decode
    CGContextRelease(contextRef);
    
    return cvMat;
}

-(void)destroy{
    destroy_priors();
}
@end
