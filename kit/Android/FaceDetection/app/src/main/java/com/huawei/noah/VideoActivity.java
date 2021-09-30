// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
package com.huawei.noah;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public class VideoActivity extends AppCompatActivity implements SurfaceHolder.Callback{

    private String modelPath;
    private ImageView resultImg;

    private Camera mCamera;
    private SurfaceView surfaceView;
    private SurfaceHolder mSurfaceHolder;
    private int cameraId = 0;

    BoltResult boltResult;
    String resultImgPath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video);

        modelPath=getIntent().getStringExtra("modelPath");
        boltResult=new BoltResult();

        initView();
    }

    public void  initView(){
        surfaceView=findViewById(R.id.surfaceView);
        mSurfaceHolder=surfaceView.getHolder();
        mSurfaceHolder.addCallback((SurfaceHolder.Callback) this);

        findViewById(R.id.button).setBackgroundResource(R.drawable.rotate_camera_white3);
        resultImg=findViewById(R.id.imageView);
        resultImg.setScaleType(ImageView.ScaleType.CENTER_CROP);

        Button button=findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                cameraSwitch();
            }
        });

        findViewById(R.id.button5).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });

    }

    private void showResultImg(Bitmap bitmap, String boltPath){
        resultImgPath=boltResult.getDetectionImgPath(bitmap,boltPath);
        Bitmap resultBitmap= BitmapFactory.decodeFile(resultImgPath);

        Handler mainHandler = new Handler(Looper.getMainLooper());
        mainHandler.post(new Runnable() {
            @Override
            public void run() {
                //已在主线程中，可以更新UI
                resultImg.setImageBitmap(resultBitmap);
            }
        });
    }

    public void cameraOpen(){
        try {
            mCamera=Camera.open(cameraId);
            mCamera.setDisplayOrientation(90);
            //绑定Surface并开启预览
            mCamera.setPreviewDisplay(mSurfaceHolder);
            mCamera.startPreview();
            addCallBack();
        } catch (IOException e) {
            mCamera.release();
            mCamera=null;
            e.printStackTrace();
        }

    }

    @Override
    public void surfaceCreated(@NonNull SurfaceHolder holder) {
        cameraOpen();
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {
        Camera.Parameters parameters = mCamera.getParameters();
        parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
        List<Camera.Size> previewSizes = parameters.getSupportedPreviewSizes();
        mCamera.setParameters(parameters);
        mCamera.startPreview();
    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
        holder.removeCallback(this);
        mCamera.stopPreview();
        mCamera.setPreviewCallback(null);
        mCamera.release();
        mCamera = null;
    }

    private void addCallBack(){
        mCamera.setPreviewCallback(new Camera.PreviewCallback() {
            @Override
            public void onPreviewFrame(byte[] data, Camera camera) {

                Camera.Size size=camera.getParameters().getPreviewSize();
                data=rotateYUV420Degree90(data,size.width,size.height);
                try {
                    YuvImage image=new YuvImage(data, ImageFormat.NV21,size.height,size.width,null);
                    if (image!=null)
                    {

                        ByteArrayOutputStream stream = new ByteArrayOutputStream();
                        image.compressToJpeg(new Rect(0, 0, size.height, size.width), 80, stream);
                        Bitmap bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());

                        showResultImg(bmp,modelPath);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    //翻转摄像头
    public void cameraSwitch(){
        if (cameraId==0)
        {
            cameraId=1;

            resultImg.setRotation(180);
            resultImg.setScaleX(-1);
        }else{
            cameraId=0;

            resultImg.setRotation(0);
            resultImg.setScaleX(0);
        }

        mCamera.setPreviewCallback(null);
        mCamera.stopPreview();
        mCamera.release();
        mCamera=null;
        cameraOpen();
    }

    public static byte[] rotateYUV420Degree90(byte[] data, int imageWidth, int imageHeight) {
        byte[] yuv = new byte[imageWidth * imageHeight * 3 / 2];
        int i = 0;
        for (int x = 0; x < imageWidth; x++) {
            for (int y = imageHeight - 1; y >= 0; y--) {
                yuv[i] = data[y * imageWidth + x];
                i++;
            }
        }
        i = imageWidth * imageHeight * 3 / 2 - 1;
        for (int x = imageWidth - 1; x > 0; x = x - 2) {
            for (int y = 0; y < imageHeight / 2; y++) {
                yuv[i] = data[(imageWidth * imageHeight) + (y * imageWidth) + x];
                i--;
                yuv[i] = data[(imageWidth * imageHeight) + (y * imageWidth)
                        + (x - 1)];
                i--;
            }
        }
        return yuv;
    }

    private boolean bitMapToWritePath(Bitmap bitmap,String fileName) {
        try {
            File cacheDir=getCacheDir();
            if (!cacheDir.exists()) {
                cacheDir.mkdirs();
            }
            File outFile = new File(cacheDir, fileName);
            FileOutputStream fos = new FileOutputStream(outFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG,80,fos);
            fos.flush();
            fos.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        boltResult.destroyBolt();
    }
}
