// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#include "bolt.h"
#include <sys/time.h>
#include <unistd.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>

@property (nonatomic,assign) IBOutlet UILabel *scoreLabel;
@property (nonatomic,assign) BOOL isFirst;
@property (nonatomic,strong) AVCaptureVideoDataOutput *videoOutput;
@property (nonatomic,strong) NSMutableArray *rgbDataArr;
@property (nonatomic,strong) dispatch_queue_t queue;

@property (nonatomic,strong) NSArray *transTypeArr;

@end

#define _USE_FP32
const int topK=5;
const int width=224;
const int height=224;
AFFINITY_TYPE affinity = CPU_HIGH_PERFORMANCE;

@implementation ViewController

static double ut_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    NSString *typePath=[[NSBundle mainBundle]pathForResource:@"imagenet_classes" ofType:@"txt"];
    NSString *typeStr=[NSString stringWithContentsOfFile:typePath encoding:NSUTF8StringEncoding error:nil];
    _transTypeArr=[NSArray arrayWithArray:[typeStr componentsSeparatedByString:@"\n"] ];
    
    
    [self setupAVCapture];
    // Do any additional setup after loading the view.
}

-(void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    if (!_isFirst) {// 第一帧画面比较暗 延时获取
        __weak typeof(self)weakSelf=self;
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(NSEC_PER_SEC*0.3)), dispatch_get_main_queue(), ^{
            weakSelf.isFirst=YES;
            
        });
        return;
    }
    UIImage* image = [self imageWithImageSimple:[self imageFromSampleBuffer:sampleBuffer] scaledToSize:CGSizeMake(width, height)];//转指定大小图片

    [self.videoOutput setSampleBufferDelegate:nil queue:self.queue];//代理置空，确保图片识别成功后再获取帧数据

    __weak typeof(self)weakSelf=self;
    dispatch_async(dispatch_get_global_queue(0, 0), ^{

        //获取UIImage像素ARGB值
        CGImageAlphaInfo alphaInfo = CGImageGetAlphaInfo(image.CGImage);
        CGColorSpaceRef colorRef = CGColorSpaceCreateDeviceRGB();

        // Get source image data
        uint8_t *bufferInt = (uint8_t *) malloc(sizeof(uint8_t*)*width * height * 4);

        CGContextRef imageContext = CGBitmapContextCreate(bufferInt,
                width, height,
                8, static_cast<size_t>(width * 4),
                colorRef, alphaInfo);

        CGContextDrawImage(imageContext, CGRectMake(0, 0, width, height), image.CGImage);
        CGContextRelease(imageContext);
        CGColorSpaceRelease(colorRef);
        
        const unsigned char *buffer = (const unsigned char *)bufferInt;
        [weakSelf beginLoadData:buffer];
        
        free(bufferInt);

        weakSelf.queue = dispatch_queue_create("myQueue", NULL);
        [weakSelf.videoOutput setSampleBufferDelegate:weakSelf queue:weakSelf.queue];//数据识别成功 重新设置代理
      
    });
}

-(void)beginLoadData:(const unsigned char * )myBuffer
{
   float *oneArr = (float *) malloc(sizeof(float*)*width * height * 3);

   for (int i = 0; i < height; i++) {
       for (int y = 0; y < width; y++) {
           unsigned char r = myBuffer[i * width * 4 + y * 4 + 1];
           unsigned char g = myBuffer[i * width * 4 + y * 4 + 2];
           unsigned char b = myBuffer[i * width * 4 + y * 4 + 3];

           oneArr[i * 3 * width + y * 3] = b;
           oneArr[i * 3 * width + y * 3 + 1] = g;
           oneArr[i * 3 * width + y * 3 + 2] = r;
       }
   }
    
    DATA_TYPE precisionMode = FP_32;
    ModelHandle model_address;
    NSString *boltPath=[[NSBundle mainBundle]pathForResource:@"ghostnet_f32" ofType:@"bolt"];
    char* modelPath =(char *)[boltPath UTF8String];
    model_address = CreateModel(modelPath, affinity, NULL);
    
    int num_input = GetNumInputsFromModel(model_address);
    int *n = (int *)malloc(sizeof(int) * num_input);
    int *c = (int *)malloc(sizeof(int) * num_input);
    int *h = (int *)malloc(sizeof(int) * num_input);
    int *w = (int *)malloc(sizeof(int) * num_input);
    char **name = (char **)malloc(sizeof(char *) * num_input);
    for (int i = 0; i < num_input; i++) {
        name[i] = (char *)malloc(sizeof(char) * 1024);
    }
    DATA_TYPE *dt_input = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * num_input);
    DATA_FORMAT *df_input = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * num_input);

    GetInputDataInfoFromModel(model_address, num_input, name, n, c, h, w, dt_input, df_input);

    unsigned char **input_ptr = (unsigned char **)malloc(sizeof(unsigned char *) * num_input);
    for (int i = 0; i < num_input; i++) {
        printf("input name = %s in = %d ic = %d ih = %d iw = %d\n", name[i], n[i], c[i], h[i], w[i]);
        int length = n[i] * c[i] * h[i] * w[i];
        switch (precisionMode) {
 #ifdef _USE_FP32
            case FP_32: {
                float *ptr = (float *)malloc(sizeof(float) * length);
                for (int i = 0; i < length; i++) {
                ptr[i] = oneArr[i];
                     
                }
                input_ptr[i] = (unsigned char *)ptr;;
                break;
            }
 #endif
 #ifdef _USE_FP16
            case FP_16: {

                F16 *ptr = (F16 *)malloc(sizeof(F16) * length);
                for (int i = 0; i < length; i++) {
                    ptr[i] = oneArr[i];;
                }
                input_ptr[i] = (unsigned char *)ptr;
                break;
            }
 #endif
            default:
                printf("[ERROR] unsupported data precision in C API test\n");
                exit(1);
        }
    }
    
    PrepareModel(model_address, num_input, (const char **)name, n, c, h, w, dt_input, df_input);

    ResultHandle model_result = AllocAllResultHandle(model_address);
    int model_result_num = GetNumOutputsFromResultHandle(model_result);
    int *output_n = (int *)malloc(sizeof(int) * model_result_num);
    int *output_c = (int *)malloc(sizeof(int) * model_result_num);
    int *output_h = (int *)malloc(sizeof(int) * model_result_num);
    int *output_w = (int *)malloc(sizeof(int) * model_result_num);
    char **outputNames = (char **)malloc(sizeof(char *) * model_result_num);
    for (int i = 0; i < model_result_num; i++) {
        outputNames[i] = (char *)malloc(sizeof(char) * 1024);
    }
    DATA_TYPE *dt_output = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * model_result_num);
    DATA_FORMAT *df_output = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * model_result_num);

    GetOutputDataInfoFromResultHandle(model_result, model_result_num, outputNames, output_n,
        output_c, output_h, output_w, dt_output, df_output);
    
     unsigned char **user_out_ptr =
            (unsigned char **)malloc(sizeof(unsigned char *) * model_result_num);
        for (int i = 0; i < model_result_num; i++) {
            printf("output name = %s on = %d oc = %d oh = %d ow = %d\n", outputNames[i], output_n[i],
                output_c[i], output_h[i], output_w[i]);
            int length = output_n[i] * output_c[i] * output_h[i] * output_w[i];
            switch (precisionMode) {
    #ifdef _USE_FP32
                case FP_32: {
                    float *ptr = (float *)malloc(sizeof(float) * length);
                    user_out_ptr[i] = (unsigned char *)ptr;
                    break;
                }
    #endif
    #ifdef _USE_FP16
                case FP_16: {
                    F16 *ptr = (F16 *)malloc(sizeof(F16) * length);
                    user_out_ptr[i] = (unsigned char *)ptr;
                    break;
                }
    #endif
                default:
                    printf("[ERROR] unsupported data precision in C API test\n");
                    exit(1);
            }
        }


        double timeBegin = ut_time_ms();
        RunModel(model_address, model_result, num_input, (const char **)name, (void **)input_ptr);
        double timeEnd = ut_time_ms();
        double useTime = timeEnd - timeBegin;//处理所需时间
        printf("useTime = %f", useTime);
        
        unsigned char **bolt_out_ptr =
            (unsigned char **)malloc(sizeof(unsigned char *) * model_result_num);
        GetOutputDataFromResultHandle(model_result, model_result_num, (void **)bolt_out_ptr);
    
        for (int i = 0; i < model_result_num; i++) {
            int length = output_n[i] * output_c[i] * output_h[i] * output_w[i];
            switch (precisionMode) {
    #ifdef _USE_FP32
                case FP_32: {
                    memcpy(user_out_ptr[i], bolt_out_ptr[i], sizeof(float) * length);
                    break;
                }
    #endif
    #ifdef _USE_FP16
                case FP_16: {
                    memcpy(user_out_ptr[i], bolt_out_ptr[i], sizeof(F16) * length);
                    break;
                }
    #endif
                default:
                    printf("[ERROR] unsupported data precision in C API test\n");
                    exit(1);
            }
        }
    
      float *val=(float *)(*(user_out_ptr));
      int *topIndexArr=(int*)malloc(sizeof(int*)* 5);;
    
      for (int i = 0; i < topK; i++) {
        int max_index = 0;
        for (int j = 1; j < 1000; j ++) {
            if (val[j] > val[max_index]) {
                max_index = j;
            }
        }
        topIndexArr[i] = max_index;
        val[max_index] = -100;
      }

        __weak typeof(self)weakSelf = self;
        dispatch_async(dispatch_get_main_queue(), ^{
            for (int i = 0; i < topK; i++) {
                if (i == 0) {
                   weakSelf.scoreLabel.text=[NSString stringWithFormat:@"%d:%@",i+1,weakSelf.transTypeArr[topIndexArr[i]]];
                } else {
                   weakSelf.scoreLabel.text=[NSString stringWithFormat:@"%@\n%d,%@",weakSelf.scoreLabel.text,i+1,weakSelf.transTypeArr[topIndexArr[i]]];
                }
            }
            free(topIndexArr);
        });
        
    
        FreeResultHandle(model_result);
        DestroyModel(model_address);
        free(n);
        free(c);
        free(h);
        free(w);
        free(dt_input);
        free(df_input);
        for (int i = 0; i < num_input; i++) {
            free(name[i]);
            free(input_ptr[i]);
        }
        free(name);
        free(input_ptr);
        free(output_n);
        free(output_c);
        free(output_h);
        free(output_w);
        free(dt_output);
        free(df_output);
        for (int i = 0; i < model_result_num; i++) {
            free(outputNames[i]);
            free(user_out_ptr[i]);
        }
        free(outputNames);
        free(user_out_ptr);
        free(bolt_out_ptr);
    
        free(oneArr);
        fflush(stdout);
}

-(void)setupAVCapture
{
    NSError *error=nil;
    
    AVCaptureSession *session=[[AVCaptureSession alloc] init];
    session.sessionPreset=AVCaptureSessionPreset1280x720;
    [session beginConfiguration];
    
    AVCaptureDevice *device=[AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceInput *deviceInput=[AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if ([session canAddInput:deviceInput]) {
        [session addInput:deviceInput];
    }
    
    _videoOutput=[[AVCaptureVideoDataOutput alloc]init];
    _videoOutput.alwaysDiscardsLateVideoFrames=YES;
    _videoOutput.videoSettings=[NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA] forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    if ([session canAddOutput:_videoOutput]) {
        [session addOutput:_videoOutput];
    }
    
    self.queue=dispatch_queue_create("myQueue", NULL);
    [_videoOutput setSampleBufferDelegate:self queue:self.queue];
    AVCaptureVideoPreviewLayer *preLayer=[AVCaptureVideoPreviewLayer layerWithSession:session];
    preLayer.frame = CGRectMake((self.view.frame.size.width - width) / 2, 60, width, height);
    preLayer.videoGravity=AVLayerVideoGravityResizeAspectFill;
    [self.view.layer addSublayer:preLayer];

    [session commitConfiguration];
    [session startRunning];
}

-(UIImage *)imageFromSampleBuffer:(CMSampleBufferRef)sampleBuffer
{
    CVImageBufferRef imageBuffer=CMSampleBufferGetImageBuffer(sampleBuffer);

    CVPixelBufferLockBaseAddress(imageBuffer, 0);

    size_t bytesPerRow=CVPixelBufferGetBytesPerRow(imageBuffer);

    size_t width=CVPixelBufferGetWidth(imageBuffer);
    size_t height=CVPixelBufferGetHeight(imageBuffer);

    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
    
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);

    CGColorSpaceRef colorSpace=CGColorSpaceCreateDeviceRGB();

    CGContextRef context=CGBitmapContextCreate(baseAddress, width, height, 8, bytesPerRow, colorSpace, kCGBitmapByteOrder32Little|kCGImageAlphaPremultipliedFirst);
    
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
   
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    UIImage *image = [UIImage imageWithCGImage:quartzImage scale:1.0 orientation:UIImageOrientationRight];
    
    CGImageRelease(quartzImage);
    return (image);
}

-(UIImage *)imageWithImageSimple:(UIImage*)image scaledToSize:(CGSize)newSize
{
    UIGraphicsBeginImageContext(newSize);
    [image drawInRect:CGRectMake(0, 0, newSize.width, newSize.height)];
    UIImage *newImage=UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}



@end
