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
#import "VideoDetectionVC.h"
#import "PhotoDetectionVC.h"
@interface ViewController ()

@property(nonatomic,strong)NSString *modelPathStr;
@property(nonatomic,strong)NSString *resultImgPath;//结果图片路径
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.view.backgroundColor=[UIColor whiteColor];
    self.navigationItem.title=@"FaceDetection";
    _modelPathStr=[[NSBundle mainBundle]pathForResource:@"simplified_f32" ofType:@"bolt"];
    
    NSString * path =NSHomeDirectory();
    _resultImgPath=[path stringByAppendingString:@"/Documents/result_bolt.jpg"];
}

-(IBAction)selectBt:(UIButton *)sender{
    UIStoryboard *main=[UIStoryboard storyboardWithName:@"Main" bundle:nil];
    if (sender.tag==1) {
        VideoDetectionVC *videoVC=[main instantiateViewControllerWithIdentifier:@"VideoDetectionVC"];
        videoVC.modelPathStr=_modelPathStr;
        videoVC.resultImgPath=_resultImgPath;
        [self.navigationController pushViewController:videoVC animated:YES];
    }else
    {
        PhotoDetectionVC *photoVC=[main instantiateViewControllerWithIdentifier:@"PhotoDetectionVC"];
        photoVC.modelPathStr=_modelPathStr;
        photoVC.resultImgPath=_resultImgPath;
        [self.navigationController pushViewController:photoVC animated:YES];
    }
}

@end
