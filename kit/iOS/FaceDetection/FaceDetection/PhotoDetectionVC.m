// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#import "PhotoDetectionVC.h"
#import <AVFoundation/AVFoundation.h>
#import "BoltResult.h"
@interface PhotoDetectionVC ()<UIImagePickerControllerDelegate,UINavigationControllerDelegate>
@property(nonatomic,assign)IBOutlet UIImageView* resultImg;

@property(nonatomic,strong)BoltResult *boltResult;
@end

@implementation PhotoDetectionVC

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor=[UIColor whiteColor];
    self.navigationItem.title=@"图片检测";
    

    _boltResult=[[BoltResult alloc]init];
    [_boltResult initBolt:_modelPathStr ResultPath:_resultImgPath];
    
    UIButton *rightBt=[UIButton buttonWithType:UIButtonTypeCustom];
    rightBt.titleLabel.font=[UIFont systemFontOfSize:15];
    [rightBt setTitle:@"获取图片" forState:UIControlStateNormal];
    [rightBt addTarget:self action:@selector(clickRightBt) forControlEvents:UIControlEventTouchUpInside];
    UIBarButtonItem *rightItem=[[UIBarButtonItem alloc]initWithCustomView:rightBt];
    self.navigationItem.rightBarButtonItem=rightItem;
    // Do any additional setup after loading the view.
}

-(void)clickRightBt{
    UIAlertController *alertController=[UIAlertController alertControllerWithTitle:@"图片获取" message:nil preferredStyle:UIAlertControllerStyleActionSheet];
    UIAlertAction *cameraAction=[UIAlertAction actionWithTitle:@"拍照获取" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        [self selFromCamera];
    }];
    UIAlertAction *photoAction=[UIAlertAction actionWithTitle:@"相册获取" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        [self selFromAlbum];
    }];
   
    [alertController addAction:cameraAction];
    [alertController addAction:photoAction];
    [self presentViewController:alertController animated:YES completion:nil];
    
}

-(void)selFromCamera{
    UIImagePickerController *imagePicker = [[UIImagePickerController alloc] init];
    imagePicker.sourceType = UIImagePickerControllerSourceTypeCamera;
    imagePicker.delegate = self;
    [self presentViewController:imagePicker animated:YES completion:nil];
}

-(void)selFromAlbum{
    UIImagePickerController *imagePicker=[[UIImagePickerController alloc]init];
    imagePicker.sourceType=UIImagePickerControllerSourceTypePhotoLibrary;
    imagePicker.delegate=self;
    imagePicker.allowsEditing =YES;
    [self presentViewController:imagePicker animated:YES completion:nil];
    
}

-(void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<UIImagePickerControllerInfoKey,id> *)info{
   
    NSString *type = [info objectForKey:UIImagePickerControllerMediaType];

    //当选择的类型是图片
    if ([type isEqualToString:@"public.image"])
    {
        UIImage *image;
        if (info[UIImagePickerControllerEditedImage]!=nil) {
            image=info[UIImagePickerControllerEditedImage];
        }else{
            image=info[UIImagePickerControllerOriginalImage];
        }
        
        NSData *imageData=UIImageJPEGRepresentation(image, 1.0);
        image=[self fixImgOrientation:[UIImage imageWithData:imageData] ];
    
        [_boltResult getResultImg:image];
        [self showResultImage];
        [self dismissViewControllerAnimated:YES completion:nil];
    }
   
}

-(void)showResultImage{
    NSData *imgDate=[NSData dataWithContentsOfFile:self.resultImgPath];
    UIImage *resultImg=[UIImage imageWithData:imgDate];
    [self.resultImg setImage:resultImg];
}


-(UIImage *)fixImgOrientation:(UIImage *)img{
    if (img.imageOrientation!=UIImageOrientationUp) {
        UIGraphicsBeginImageContext(img.size);
        [img drawInRect:CGRectMake(0, 0, img.size.width, img.size.height)];
        img = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
    
    }
    
    return img;
}
-(void)dealloc{

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
