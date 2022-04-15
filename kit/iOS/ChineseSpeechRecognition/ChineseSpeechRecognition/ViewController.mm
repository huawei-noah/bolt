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
#include "kit_flags.h"
#include "flow_asr.h"
#import <AVFoundation/AVFoundation.h>

@interface ViewController ()<AVAudioRecorderDelegate,AVAudioPlayerDelegate>

@property(nonatomic,weak)IBOutlet UILabel *audioTransContentLabel;
@property(nonatomic,weak)IBOutlet UIView *textBgView;
@property(nonatomic,weak)IBOutlet UIImageView *yuyinImg;
@property(nonatomic,strong)AVAudioRecorder *audioRecorder;
@property(nonatomic,strong)AVAudioPlayer *audioPlayer;

@property(nonatomic,strong)NSString *url;
@end

@implementation ViewController


- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor=[UIColor whiteColor];
    
    _textBgView.layer.masksToBounds=YES;
    _textBgView.layer.borderWidth=1;
    _textBgView.layer.borderColor=[UIColor colorWithRed:150/255.0 green:150/255.0 blue:150/255.0 alpha:1].CGColor;
    _textBgView.layer.cornerRadius=5;
    
    _yuyinImg.hidden=YES;
 
    flowRegisterFunction("encoderInferOutputSize", encoderInferOutputSize);
    flowRegisterFunction("encoderPreProcess", encoderPreProcess);
    flowRegisterFunction("predictionInferOutputSize", predictionInferOutputSize);
    flowRegisterFunction("jointInferOutputSize", jointInferOutputSize);
    flowRegisterFunction("pinyin2hanziInferOutputSize", pinyin2hanziInferOutputSize);
    flowRegisterFunction("pinyin2hanziPreProcess", pinyin2hanziPreProcess);

  
    NSMutableArray *pathArr=[[NSMutableArray alloc]init];
    NSArray *arr1=[NSArray arrayWithObjects:@"encoder_flow",@"prediction_flow",@"joint_flow",@"pinyin2hanzi_flow", nil];
    NSArray *arr2=[NSArray arrayWithObjects:@"asr_convolution_transformer_encoder_f32",@"asr_convolution_transformer_prediction_net_f32",@"asr_convolution_transformer_joint_net_f32",@"cnn_pinyin_lm_b7h512e4_cn_en_20200518_cloud_fp32_f32", nil];
    for (int i=0; i<4; i++) {
         NSString *graphPathStr=[[NSBundle mainBundle]pathForResource:arr1[i] ofType:@"prototxt"];
         NSArray *path = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
         NSString *docDirectory = [path objectAtIndex:0];
         NSString *realPath = [docDirectory stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.prototxt",arr1[i]]];
         [[NSFileManager defaultManager] copyItemAtPath:graphPathStr toPath:realPath error:nil];
         NSString *myStr=[[NSString alloc]initWithContentsOfFile:realPath encoding:NSUTF8StringEncoding error:nil];
         NSMutableArray *arr=[NSMutableArray arrayWithArray:[myStr componentsSeparatedByString:@"inference_parameter:"] ];
         NSString *boltPath=[[NSBundle mainBundle]pathForResource:arr2[i] ofType:@"bolt"];
         NSString *changeStr=[NSString stringWithFormat:@"%@inference_parameter:\"%@\"\ninference_parameter:\"\"\n}",arr[0],boltPath];
         if (i==3) {
             NSString *binPath=[[NSBundle mainBundle]pathForResource:@"pinyin_lm_embedding" ofType:@"bin"];
             if([[NSUserDefaults standardUserDefaults]objectForKey:@"binPath"]==nil)
             {
                 changeStr=[changeStr stringByReplacingOccurrencesOfString:@"/data/local/tmp/CI/test/pinyin_lm_embedding.bin" withString:binPath];
                 
             }else
             {
                 changeStr=[changeStr stringByReplacingOccurrencesOfString:[[NSUserDefaults standardUserDefaults]objectForKey:@"binPath"] withString:binPath];
                 
             }
             [[NSUserDefaults standardUserDefaults]setObject:binPath forKey:@"binPath"];
         }
         NSError *error=nil;
         [changeStr writeToFile:realPath atomically:YES encoding:NSUTF8StringEncoding error:&error];
         [pathArr addObject:realPath];
    }
     
    encoderGraphPath = [pathArr[0] UTF8String];
    predictionGraphPath = [pathArr[1] UTF8String];
    jointGraphPath = [pathArr[2] UTF8String];
    pinyin2hanziGraphPath = [pathArr[3] UTF8String];
     
    NSString *labelPath=[[NSBundle mainBundle]pathForResource:@"asr_labels" ofType:@"txt"];
    labelFilePath = [labelPath UTF8String];

     
     initASRFlow();
    
    NSArray *path = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *docDirectory = [path objectAtIndex:0];
    self.url = [docDirectory stringByAppendingPathComponent:@"myWav.wav"];
    // Do any additional setup after loading the view.
}

-(IBAction)startRecord:(id)sender
{
    _yuyinImg.hidden=NO;
    

    NSMutableDictionary * settings = @{}.mutableCopy;
    [settings setObject:[NSNumber numberWithFloat:16000] forKey:AVSampleRateKey];
    [settings setObject:[NSNumber numberWithInt: kAudioFormatLinearPCM] forKey:AVFormatIDKey];
    [settings setObject:@1 forKey:AVNumberOfChannelsKey];//设置成一个通道，iPnone只有一个麦克风，一个通道已经足够了
    [settings setObject:@16 forKey:AVLinearPCMBitDepthKey];//采样的位数
    NSError *error=nil;
    self.audioRecorder = [[AVAudioRecorder  alloc] initWithURL:[NSURL fileURLWithPath:self.url] settings:settings error:&error];
    self.audioRecorder.delegate = self;
    [[AVAudioSession sharedInstance] setCategory:AVAudioSessionCategoryRecord error:nil];
    self.audioRecorder.meteringEnabled = YES;
    BOOL success = [self.audioRecorder record];
    if (success) {
        NSLog(@"录音开始成功");
    }else{
        NSLog(@"录音开始失败");
    }
  
}

-(IBAction)stopRecord:(id)sender
{
    _yuyinImg.hidden=YES;
    [self.audioRecorder stop];
}

-(IBAction)playRecord:(id)sender
{
    NSError * error;
    [[AVAudioSession sharedInstance] setCategory:AVAudioSessionCategoryPlayback error:nil];
    _audioPlayer = [[AVAudioPlayer alloc] initWithContentsOfURL:[NSURL fileURLWithPath:self.url] error:&error];
    _audioPlayer.volume = 0.5;//范围为（0到1）；
    _audioPlayer.delegate = self;
    [_audioPlayer prepareToPlay];
    BOOL success = [_audioPlayer play];
    if (success) {
        NSLog(@"播放成功");
    }else{
        NSLog(@"播放失败");
    }
}

-(IBAction)translateAudio:(id)sender
{
    std::string myWavPath = [self.url UTF8String];
    std::string hanzi=runASRFlow(myWavPath);
    self.audioTransContentLabel.text=[NSString stringWithCString:hanzi.c_str() encoding:NSUTF8StringEncoding];
}
@end
