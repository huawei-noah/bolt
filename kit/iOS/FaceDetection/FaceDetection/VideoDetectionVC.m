// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#import "VideoDetectionVC.h"
#import <AVFoundation/AVFoundation.h>
#import "BoltResult.h"
@interface VideoDetectionVC ()<AVCaptureVideoDataOutputSampleBufferDelegate>
@property(nonatomic,assign)IBOutlet UIImageView* resultImg;
@property(nonatomic,strong)dispatch_queue_t queue;
@property (nonatomic,assign) BOOL isFirst;

@property(nonatomic,strong)AVCaptureSession *session;
@property(nonatomic,strong)AVCaptureDevice *device;
@property(nonatomic,strong)AVCaptureDeviceInput *deviceInput;
@property(nonatomic,strong)AVCaptureVideoDataOutput *videoOutput;

@property(nonatomic,strong)BoltResult *boltResult;
@end

@implementation VideoDetectionVC

- (void)viewDidLoad {
    [super viewDidLoad];
    self.navigationItem.title=@"实时检测";
    _boltResult=[[BoltResult alloc]init];
    [_boltResult initBolt:_modelPathStr ResultPath:_resultImgPath];
    
    _resultImg.contentMode=UIViewContentModeScaleAspectFill;
    [self setupAVCapture];
    // Do any additional setup after loading the view.
}

-(void)setupAVCapture{
    NSError *error=nil;


    _session=[[AVCaptureSession alloc]init];
    [_session beginConfiguration];

    _device=[AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionFront];
    _deviceInput=[AVCaptureDeviceInput deviceInputWithDevice:_device error:&error];
    if ([_session canAddInput:_deviceInput]) {
        [_session addInput:_deviceInput];
    }

    _videoOutput=[[AVCaptureVideoDataOutput alloc]init];
    _videoOutput.alwaysDiscardsLateVideoFrames=YES;
    _videoOutput.videoSettings=[NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA] forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    if ([_session canAddOutput:_videoOutput]) {
        [_session addOutput:_videoOutput];
    }

    self.queue=dispatch_queue_create("myQueue", NULL);
    [_videoOutput setSampleBufferDelegate:self queue:self.queue];
    
    [_session commitConfiguration];
    [_session startRunning];

}


-(AVCaptureDevice *)cameraWithPosition:(AVCaptureDevicePosition)position{
    AVCaptureDeviceDiscoverySession *captureDeviceDiscoverySession = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera]
                                          mediaType:AVMediaTypeVideo
                                           position: AVCaptureDevicePositionUnspecified]; // here you pass AVCaptureDevicePositionUnspecified to find all capture devices

    NSArray *captureDevices = [captureDeviceDiscoverySession devices];
    for (AVCaptureDevice *device in captureDevices)
    {
        if ( device.position == position )
        {
            return device;
        }
    }

    return nil;
}



-(IBAction)changeCameraPosition:(id)sender{
    
    [self.videoOutput setSampleBufferDelegate:nil queue:self.queue];
    
    AVCaptureDeviceDiscoverySession *captureDeviceDiscoverySession = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera]
                                          mediaType:AVMediaTypeVideo
                                           position: AVCaptureDevicePositionUnspecified]; // here you pass AVCaptureDevicePositionUnspecified to find all capture devices

    NSArray *captureDevices = [captureDeviceDiscoverySession devices];
    NSInteger cameraCount = [captureDevices count];
        //摄像头的数量小于等于1的时候直接返回
    if (cameraCount <= 1) {
        return;
    }

    AVCaptureDevicePosition position=self.device.position;
    if (position==AVCaptureDevicePositionFront) {
        self.device=[self cameraWithPosition:AVCaptureDevicePositionBack];
    }else
    {
        self.device=[self cameraWithPosition:AVCaptureDevicePositionFront];
    }

    NSError *error;
    AVCaptureDeviceInput *deviceInput=[AVCaptureDeviceInput deviceInputWithDevice:_device error:&error];

    [self.session beginConfiguration];
    [self.session removeInput:self.deviceInput];
    if ([self.session canAddInput:deviceInput]) {
        [self.session addInput:deviceInput];
        self.deviceInput=deviceInput;
    }else{
        [self.session addInput:self.deviceInput];
    }
    [self.session commitConfiguration];
    
    [self.videoOutput setSampleBufferDelegate:self queue:self.queue];
}

-(void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection{

        // The first frame is dark
    connection.videoOrientation=AVCaptureVideoOrientationPortrait;
    __weak typeof(self)weakSelf=self;

    UIImage *image=[weakSelf imageFromSampleBuffer:sampleBuffer];
    [weakSelf.boltResult getResultImg:image];

    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        NSData *imgDate=[NSData dataWithContentsOfFile:weakSelf.resultImgPath];
        UIImage *resultImg=[UIImage imageWithData:imgDate];
        if (weakSelf.device.position==AVCaptureDevicePositionFront) {//镜像翻转
            resultImg=[UIImage imageWithCGImage:resultImg.CGImage scale:resultImg.scale orientation:UIImageOrientationUpMirrored];
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf.resultImg setImage:resultImg];
        });
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


-(void)dealloc{
    [self.session stopRunning];
    self.session=nil;
    
    [_boltResult destroy];
    _boltResult=nil;
    
}
/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
