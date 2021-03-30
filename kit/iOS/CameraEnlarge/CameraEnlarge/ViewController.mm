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
#import <Accelerate/Accelerate.h>
#include "kit_flags.h"
#include "flow.h"

@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate,UIGestureRecognizerDelegate>
@property (nonatomic,strong) AVCaptureVideoDataOutput *videoOutput;
@property (nonatomic,strong) dispatch_queue_t queue;
@property(nonatomic,assign)IBOutlet UIImageView *photo;
@property(nonatomic,assign)CGFloat currentZoomFactor;
@property(nonatomic,assign)CGFloat maxZoomFactor;
@property(nonatomic,assign)CGFloat minZoomFactor;
@property (nonatomic,strong) AVCaptureDevice *device;
@property(nonatomic,strong)UIImage *nowImg;
@property (nonatomic,strong) NSString *dstPath;
@property (nonatomic,assign) BOOL isFirst;
@property(nonatomic,assign)IBOutlet NSLayoutConstraint *finishImgWidth;
@end

DataType inferencePrecision = DT_F32;
const int imgWidth=32;
const int imgHeight=32;
Flow flowExample;

using namespace std;

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    flowRegisterFunction("pixelProcess", pixelProcess);
    flowRegisterFunction("postProcess", postProcess);
    
    self.photo.contentMode=UIViewContentModeScaleAspectFill;
    
    self.finishImgWidth.constant=2*imgWidth;
    
    [self setupAVCapture];
    
    UIPinchGestureRecognizer *pinchGesture = [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(zoomChangePinchGestureRecognizerClick:)];
    pinchGesture.delegate = self;
    [self.view addGestureRecognizer:pinchGesture];
    self.currentZoomFactor = 1;
    self.minZoomFactor=1;
    self.maxZoomFactor=15;
    
    NSString *graphPathStr=[[NSBundle mainBundle]pathForResource:@"image_classification" ofType:@"prototxt"];
         
    NSArray *path = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *docDirectory = [path objectAtIndex:0];

    _dstPath = [docDirectory stringByAppendingPathComponent:@"image_classification.prototxt"];
    [[NSFileManager defaultManager] copyItemAtPath:graphPathStr toPath:_dstPath error:nil];
         
    NSString *myStr=[[NSString alloc]initWithContentsOfFile:_dstPath encoding:NSUTF8StringEncoding error:nil];
    NSMutableArray *arr=[NSMutableArray arrayWithArray:[myStr componentsSeparatedByString:@"inference_parameter:"] ];
         
    NSString *boltPath=[[NSBundle mainBundle]pathForResource:@"esr_1_f32" ofType:@"bolt"];
     
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

    // Do any additional setup after loading the view.
}

- (BOOL)gestureRecognizerShouldBegin:(UIGestureRecognizer *)gestureRecognizer
{
    if ([gestureRecognizer isKindOfClass:[UIPinchGestureRecognizer class]]){
        self.currentZoomFactor = self.device.videoZoomFactor;
    }
    return YES;
}

//缩放手势
- (void)zoomChangePinchGestureRecognizerClick:(UIPinchGestureRecognizer *)pinchGestureRecognizer
{
    if (pinchGestureRecognizer.state == UIGestureRecognizerStateBegan ||
        pinchGestureRecognizer.state == UIGestureRecognizerStateChanged)
    {
        CGFloat currentZoomFactor = self.currentZoomFactor * pinchGestureRecognizer.scale;
        
        NSLog(@"%.1f,%.1f",self.currentZoomFactor,pinchGestureRecognizer.scale);
        [self changeFactor:currentZoomFactor];
    }

}

-(void) changeFactor:(CGFloat)currentZoomFactor{
    if (currentZoomFactor < self.maxZoomFactor &&
        currentZoomFactor > self.minZoomFactor){
        
        NSError *error = nil;
        if ([self.device lockForConfiguration:&error] ) {
//            dispatch_async(dispatch_get_main_queue(), ^{
                [self.device rampToVideoZoomFactor:currentZoomFactor withRate:3];//rate越大，动画越慢
                [self.device unlockForConfiguration];
//            });
 
        }
        else {
            NSLog( @"Could not lock device for configuration: %@", error );
        }
    }
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

    UIImage* image =[self imageWithImageSimple:[self imageFromSampleBuffer:sampleBuffer] scaledToSize:CGSizeMake(imgWidth, imgHeight)];
   
    [self.videoOutput setSampleBufferDelegate:nil queue:self.queue];
    __weak typeof(self)weakSelf=self;
       dispatch_async(dispatch_get_global_queue(0, 0), ^{
            CGImageAlphaInfo alphaInfo = CGImageGetAlphaInfo(image.CGImage);
            CGColorSpaceRef colorRef = CGColorSpaceCreateDeviceRGB();

            // Get source image data ARGB
            uint8_t *buffer = (uint8_t *) malloc(sizeof(uint8_t*)*imgWidth * imgHeight * 4);

            CGContextRef imageContext = CGBitmapContextCreate(buffer,
                        imgWidth, imgHeight,
                        8, static_cast<size_t>(imgWidth * 4),
                        colorRef, alphaInfo);

            CGContextDrawImage(imageContext, CGRectMake(0, 0, imgWidth, imgHeight), image.CGImage);
            CGContextRelease(imageContext);
            CGColorSpaceRelease(colorRef);

            const unsigned char *myBuffer=(const unsigned char *)buffer;
          
           
            [weakSelf beginLoadData:myBuffer];
            free(buffer);

            weakSelf.queue = dispatch_queue_create("myQueue", NULL);
            [weakSelf.videoOutput setSampleBufferDelegate:weakSelf queue:weakSelf.queue];
         
       });
    
    
}


- (void)imageFromBRGABytes:(unsigned char *)imageBytes imageSize:(CGSize)imageSize {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(imageBytes,imageSize.width,imageSize.height,8,imageSize.width *4,
                                                 colorSpace,kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedFirst);
    
    CGImageRef imageRef = CGBitmapContextCreateImage(context);
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    __weak typeof(self)weakSelf = self;
    dispatch_async(dispatch_get_main_queue(), ^{
        weakSelf.photo.image =image;
    });
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


-(void)setupAVCapture
{
    NSError *error=nil;
    
    AVCaptureSession *session=[[AVCaptureSession alloc] init];
    session.sessionPreset=AVCaptureSessionPresetPhoto;
    [session beginConfiguration];
    
    _device=[AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceInput *deviceInput=[AVCaptureDeviceInput deviceInputWithDevice:_device error:&error];
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
    preLayer.frame = CGRectMake((self.view.frame.size.width - imgWidth) / 2, 60, imgWidth, imgHeight);
    preLayer.videoGravity=AVLayerVideoGravityResizeAspectFill;
    [self.view.layer addSublayer:preLayer];

    [session commitConfiguration];
    [session startRunning];
}

std::map<std::string, std::shared_ptr<Tensor>> inputOutput(const unsigned char * myBuffer)
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    TensorDesc inputDesc = tensor4df(DT_U8, DF_NCHW, 1, 4, imgWidth, imgHeight);
   
    tensors["input"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["input"]->resize(inputDesc);
    tensors["input"]->alloc();
    void *ptr = (void *)((CpuMemory *)tensors["input"]->get_memory())->get_ptr();
    memcpy(ptr, myBuffer, tensorNumBytes(inputDesc));
    
    tensors["output"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["output"]->resize(
        tensor2df(DT_I32, DF_NCHW, 1, (imgWidth*2)*(imgWidth*2)*3));
    tensors["output"]->alloc();
    
    return tensors;
}


EE pixelProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    // RGBA
    unsigned char *myBuffer = (unsigned char *)((CpuMemory*)inputs["input"]->get_memory())->get_ptr();

    F32 *oneArr = (F32 *)((CpuMemory *)outputs["input.1"]->get_memory())->get_ptr();

    F32 *rArr=(F32*)malloc(sizeof(F32*)*imgHeight*imgWidth);
    F32 *gArr=(F32*)malloc(sizeof(F32*)*imgHeight*imgWidth);
    F32 *bArr=(F32*)malloc(sizeof(F32*)*imgHeight*imgWidth);
    for (int i = 0; i < imgHeight; i++) {
        for (int y = 0; y < imgWidth; y++) {
            unsigned char r = myBuffer[i * imgWidth * 4 + y * 4 +1];
            unsigned char g = myBuffer[i * imgWidth * 4 + y * 4 + 2];
            unsigned char b = myBuffer[i * imgWidth * 4 + y * 4 + 3];

            rArr[i*imgHeight+y]=r;
            gArr[i*imgHeight+y]=g;
            bArr[i*imgHeight+y]=b;
            
        }
    }
    
    for(int i=0;i<imgWidth*imgHeight*3;i++)
    {
        if(i<imgWidth*imgHeight)
        {
            oneArr[i]=rArr[i];
            
        }else if(i<imgWidth*imgHeight*2)
        {
            oneArr[i]=gArr[i-imgWidth*imgHeight];
        }else{
            oneArr[i]=bArr[i-imgWidth*imgHeight*2];
        }
       
    }


    free(rArr);
    free(gArr);
    free(bArr);
    
    return SUCCESS;
}

EE postProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    std::string flowInferenceNodeOutputName = "output";
    std::string boltModelOutputName = "1811";

    uint8_t *flowInferenceNodeOutput = (uint8_t *)((CpuMemory *)outputs[flowInferenceNodeOutputName]->get_memory())->get_ptr();

    F32 *rgbData = (F32 *)((CpuMemory *)inputs[boltModelOutputName]->get_memory())->get_ptr();
   
    F32 *rArr=(F32*)malloc(sizeof(F32*)*imgHeight*2*imgWidth*2);
    F32 *gArr=(F32*)malloc(sizeof(F32*)*imgHeight*2*imgWidth*2);
    F32 *bArr=(F32*)malloc(sizeof(F32*)*imgHeight*2*imgWidth*2);
    for (int i = 0; i <(imgHeight*2)*(imgWidth*2)*3; i++) {

        if(rgbData[i]<=1)
        {
            int a=0;
            rgbData[i]=a;
        }else if (rgbData[i]>255)
        {
            int b=255;
            rgbData[i]=b;
        }
        
        if (i<(imgHeight*2)*(imgWidth*2)) {
            
            gArr[i]=rgbData[i];
        }else if(i<(imgHeight*2)*(imgWidth*2)*2)
        {

            bArr[i-(imgHeight*2)*(imgWidth*2)]=rgbData[i];
        }else{
  
            rArr[i-2*(imgHeight*2)*(imgWidth*2)]=rgbData[i];
        }
       
    }
    for (int i=0; i<(imgHeight*2)*(imgWidth*2); i++) {
        int r=rArr[i];
        int g=gArr[i];
        int b=bArr[i];
        
        flowInferenceNodeOutput[i*3]=(unsigned char)r;
        flowInferenceNodeOutput[i*3+1]=(unsigned char)g;
        flowInferenceNodeOutput[i*3+2]=(unsigned char)b;
    }
    free(rArr);
    free(gArr);
    free(bArr);
   
    return SUCCESS;
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
    UNI_PROFILE(results = flowExample.dequeue(true), std::string("image_classification"),
        std::string("image_classification"));

    uint8_t *resultRGBArr = (uint8_t *)((CpuMemory *)results[0].data["output"]->get_memory())->get_ptr();
    
    uint8_t *endResult = (uint8_t *) malloc(sizeof(uint8_t*)*(imgWidth*2) *(imgHeight*2) * 4);
    for (int i = 0; i <(imgHeight*2)*(imgWidth*2)*4; i++) {
        if(i%4!=0){
            endResult[i]=resultRGBArr[i-(i/4)];
        }else{
            int alpha=255;
            endResult[i]=(unsigned char)alpha;
        }
    }
    [self imageFromBRGABytes:endResult imageSize:CGSizeMake(imgWidth*2, imgHeight*2)];
    free(endResult);

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
