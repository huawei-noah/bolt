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
#include "kit_flags.h"
#include "flow.h"
@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>

@property (nonatomic,strong) UIImageView *imgView;
@property (nonatomic,assign) IBOutlet UILabel *scoreLabel;
@property (nonatomic,strong) AVCaptureVideoDataOutput *videoOutput;
@property (nonatomic,strong) NSMutableArray *rgbDataArr;
@property (nonatomic,strong) dispatch_queue_t queue;

@property (nonatomic,strong) NSArray *transTypeArr;
@property (nonatomic,assign) BOOL isFirst;

@property (nonatomic,strong) NSString *dstPath;

@end

DataType inferencePrecision = DT_F32;
const int topK=5;
const int width=224;
const int height=224;
Flow flowExample;

using namespace std;

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
    self.view.backgroundColor=[UIColor whiteColor];
    
    flowRegisterFunction("pixelProcess", pixelProcess);
    flowRegisterFunction("postProcess", postProcess);
   
    NSString *typePath=[[NSBundle mainBundle]pathForResource:@"imagenet_classes" ofType:@"txt"];
    NSString *typeStr=[NSString stringWithContentsOfFile:typePath encoding:NSUTF8StringEncoding error:nil];
    _transTypeArr=[NSArray arrayWithArray:[typeStr componentsSeparatedByString:@"\n"] ];

    [self setupAVCapture];
    
    NSString *graphPathStr=[[NSBundle mainBundle]pathForResource:@"image_classification" ofType:@"prototxt"];
         
    NSArray *path = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *docDirectory = [path objectAtIndex:0];

    _dstPath = [docDirectory stringByAppendingPathComponent:@"image_classification.prototxt"];
    [[NSFileManager defaultManager] copyItemAtPath:graphPathStr toPath:_dstPath error:nil];
         
    NSString *myStr=[[NSString alloc]initWithContentsOfFile:_dstPath encoding:NSUTF8StringEncoding error:nil];
    NSMutableArray *arr=[NSMutableArray arrayWithArray:[myStr componentsSeparatedByString:@"inference_parameter:"] ];
         
    NSString *boltPath=[[NSBundle mainBundle]pathForResource:@"ghostnet_f32" ofType:@"bolt"];
     
    NSString *changeStr=[NSString stringWithFormat:@"%@inference_parameter:\"%@\"\ninference_parameter:\"\"\n}", arr[0], boltPath];
      
    NSError *error=nil;
    [changeStr writeToFile:_dstPath atomically:YES encoding:NSUTF8StringEncoding error:&error];
    if (error) {
        NSLog(@"%@",error);
    }
     
    char* gPath =(char *)[_dstPath UTF8String];
    std::string imageClassificationGraphPath = gPath;
    std::vector<std::string> graphPath = {imageClassificationGraphPath};
    int threads = 1;

    flowExample.init(graphPath, inferencePrecision, AFFINITY_CPU_HIGH_PERFORMANCE, threads, false);
}

EE pixelProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    // RGBA
    unsigned char *myBuffer = (unsigned char *)((CpuMemory*)inputs["input:1"]->get_memory())->get_ptr();

    
   
    F32 *oneArr = (F32 *)((CpuMemory *)outputs["input:0"]->get_memory())->get_ptr();

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
    return SUCCESS;
}

EE postProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    std::string flowInferenceNodeOutputName = "output";
    std::string boltModelOutputName = "MobileNetV2/Predictions/Softmax:0";
    
    int *flowInferenceNodeOutput = (int *)((CpuMemory *)outputs[flowInferenceNodeOutputName]->get_memory())->get_ptr();

    F32 *score1000 = (F32 *)((CpuMemory *)inputs[boltModelOutputName]->get_memory())->get_ptr();

    for (int i = 0; i < topK; i++) {
        int max_index = 0;
        for (int j = 1; j < 1000; j ++) {
            if (score1000[j] > score1000[max_index]) {
              max_index = j;
            }
        }
        flowInferenceNodeOutput[i] = max_index;
        score1000[max_index] = -65504;
    }
    return SUCCESS;
}


std::map<std::string, std::shared_ptr<Tensor>> inputOutput(const unsigned char * myBuffer)
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    TensorDesc inputDesc = tensor4df(DT_U8, DF_NCHW, 1, width, height, 4);
   
    tensors["input:1"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["input:1"]->resize(inputDesc);
    tensors["input:1"]->alloc();
    void *ptr = (void *)((CpuMemory *)tensors["input:1"]->get_memory())->get_ptr();
    memcpy(ptr, myBuffer, tensorNumBytes(inputDesc));
    
    tensors["output"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["output"]->resize(
        tensor2df(DT_I32, DF_NCHW, 1, topK));
    tensors["output"]->alloc();
    
    return tensors;
}

-(void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    if (!_isFirst) {
        // The first frame is dark
        __weak typeof(self)weakSelf=self;
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(NSEC_PER_SEC*0.3)), dispatch_get_main_queue(), ^{
            weakSelf.isFirst=YES;
            
        });
        return;
    }
    UIImage* image = [self imageWithImageSimple:[self imageFromSampleBuffer:sampleBuffer] scaledToSize:CGSizeMake(width, height)];

    [self.videoOutput setSampleBufferDelegate:nil queue:self.queue];

    __weak typeof(self)weakSelf=self;
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
//        CGImageRef img = [image CGImage];
//        CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(img));
//        const unsigned char *buffer = CFDataGetBytePtr(data);
//        CFRelease(data);
        
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
        [weakSelf.videoOutput setSampleBufferDelegate:weakSelf queue:weakSelf.queue];
      
    });
}

-(void)beginLoadData:(const unsigned char * )myBuffer
{
   
    char* gPath =(char *)[_dstPath UTF8String];
    
    int num = 1;
    std::string imageClassificationGraphPath = gPath;

    for (int i = 0; i < num; i++) {
        std::map<std::string, std::shared_ptr<Tensor>> data = inputOutput(myBuffer);
        Task task(imageClassificationGraphPath, data);
        flowExample.enqueue(task);
    }

    std::vector<Task> results;
    double start = ut_time_ms();
    UNI_PROFILE(results = flowExample.dequeue(true), std::string("image_classification"),
        std::string("image_classification"));
    double end = ut_time_ms();
    
    int *top5 = (int *)((CpuMemory *)results[0].data["output"]->get_memory())->get_ptr();
    
    __weak typeof(self)weakSelf = self;
    dispatch_async(dispatch_get_main_queue(), ^{
        for (int i = 0; i < 5; i++) {
            if (i == 0) {
                weakSelf.scoreLabel.text=[NSString stringWithFormat:@"%d:%@",i+1,weakSelf.transTypeArr[top5[i]]];
            } else {
                weakSelf.scoreLabel.text=[NSString stringWithFormat:@"%@\n%d,%@",weakSelf.scoreLabel.text,i+1,weakSelf.transTypeArr[top5[i]]];
            }
        }
        weakSelf.scoreLabel.text=[NSString stringWithFormat:@"%@\ntime=%lfms",weakSelf.scoreLabel.text,(end - start) / num];
    });
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
