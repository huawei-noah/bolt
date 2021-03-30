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
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity implements TextureView.SurfaceTextureListener {
    private TextureView textureView;
    private Camera mCamera;
    private String[] transArr;
    private TextView tv;
    private Object Tag;

    private String readTextFromSDcard(InputStream is) throws Exception {
        InputStreamReader reader = new InputStreamReader(is);
        BufferedReader bufferedReader = new BufferedReader(reader);
        StringBuffer buffer = new StringBuffer("");
        String str;
        while ((str = bufferedReader.readLine()) != null) {
            buffer.append(str);
            buffer.append("\n");
        }
        return buffer.toString();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        InputStream is = null;
        try {
            is = getAssets().open("imagenet_classes.txt");
            String text = readTextFromSDcard(is);
            transArr = text.split("\n");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, new String[] {android.Manifest.permission.CAMERA}, 50);
        }

        initView();
        initListener();
        copyAssetAndWrite("ghostnet_f32.bolt");
    }

    private boolean copyAssetAndWrite(String fileName)
    {
        try {
            File cacheDir = getCacheDir();
            if (!cacheDir.exists()) {
                cacheDir.mkdirs();
            }
            File outFile = new File(cacheDir, fileName);
            if (!outFile.exists()) {
                boolean res = outFile.createNewFile();
                if (!res) {
                    return false;
                }
            } else {
                if (outFile.length() > 0) {
                    return true;
                }
            }
            InputStream is = getAssets().open(fileName);
            FileOutputStream fos = new FileOutputStream(outFile);
            byte[] buffer = new byte[1024];
            int byteCount;
            while ((byteCount = is.read(buffer)) != -1) {
                fos.write(buffer, 0, byteCount);
            }
            fos.flush();
            is.close();
            fos.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }

    private void initView()
    {
        tv = findViewById(R.id.sample_text);
        textureView = (TextureView)findViewById(R.id.texture_view);
    }

    private void initListener()
    {
        textureView.setSurfaceTextureListener(this);
    }

    @Override
    public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height)
    {
        mCamera = Camera.open(0);
        mCamera.setDisplayOrientation(90);
        if (mCamera != null) {
            Camera.Parameters params = mCamera.getParameters();
            List<Camera.Size> PreviewSizes = params.getSupportedPreviewSizes();
            Camera.Size size = null;
            for (int i = 0; i < PreviewSizes.size(); i++) {
                Camera.Size tmpSize = PreviewSizes.get(i);

                if (tmpSize.height >= height && tmpSize.width >= width) {
                    size = tmpSize;
                } else {
                    break;
                }
            }

            params.setPreviewSize(size.width, size.height);

            List<String> focusModes = params.getSupportedFocusModes();
            if (focusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                mCamera.setParameters(params);
            }
            try {
                mCamera.setPreviewTexture(surface);
                mCamera.startPreview();
                addCallBack();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private int[] topResultArr(float[] array, int length)
    {
        int[] resultArray = new int[length];
        for (int i = 0; i < length; i++) {
            int max_index = 0;
            for (int j = 1; j < array.length; j ++) {
                if (array[j] > array[max_index]) {
                    max_index = j;
                }
            }
            resultArray[i] = max_index;
            array[max_index] = -100;
        }

        return resultArray;
    }

    private void addCallBack()
    {
        if (mCamera != null) {
            mCamera.setPreviewCallback(new Camera.PreviewCallback() {
                @RequiresApi(api = Build.VERSION_CODES.O)
                @Override
                public void onPreviewFrame(byte[] data, Camera camera)
                {
                    Camera.Size size = camera.getParameters().getPreviewSize();
                    data = rotateYUV420Degree90(data, size.width, size.height);

                    try {
                        YuvImage image =
                            new YuvImage(data, ImageFormat.NV21, size.height, size.width, null);
                        if (image != null) {
                            ByteArrayOutputStream stream = new ByteArrayOutputStream();
                            image.compressToJpeg(
                                new Rect(0, 0, size.height, size.width), 80, stream);
                            Bitmap bmp = BitmapFactory.decodeByteArray(
                                stream.toByteArray(), 0, stream.size());
                            bmp = Bitmap.createScaledBitmap(bmp, 224, 224, true);

                            byte[] bgrByteArr = bitmap2BGR(bmp);

                            float[] floatData = new float[bgrByteArr.length];
                            int[] intData = new int[bgrByteArr.length];
                            for (int i = 0; i < bgrByteArr.length; i++) {
                                intData[i] = Byte.toUnsignedInt(bgrByteArr[i]);
                                floatData[i] = intData[i];
                            }

                            int inputNum = 1;
                            String[] inputName = {"input:0"};
                            int[] inputN = {1};
                            int[] inputC = {224};
                            int[] inputH = {224};
                            int[] inputW = {3};
                            DataType[] intputDataType = {DataType.FP32};
                            DataFormat[] intputDataFormat = {DataFormat.NCHW};

                            int length = 224 * 224 * 3;
                            float[][] inputData = new float[1][length];
                            for (int i = 0; i < length; i++) {
                                inputData[0][i] = floatData[i];
                            }

                            BoltModel boltModel = new BoltModel(
                                "/data/user/0/com.huawei.noah/cache/ghostnet_f32.bolt",
                                AffinityType.CPU_HIGH_PERFORMANCE, inputNum, inputName, inputN,
                                inputC, inputH, inputW, intputDataType, intputDataFormat);
                            BoltResult boltResult = boltModel.run(inputNum, inputName, inputData);
                            if (null == boltResult) {
                                System.err.println("[ERROR] modelAddr is 0 in Java API test");
                                boltModel.destructor();
                                System.exit(1);
                            }

                            float[][] result = boltResult.getResultData();
                            int[] topArr = topResultArr(result[0], 5);

                            // model destroy
                            boltModel.destructor();
                            String resultStr = "1：" + transArr[topArr[0]];
                            for (int i = 1; i < 5; i++) {
                                resultStr = resultStr + "\n" + (i + 1) + "：" + transArr[topArr[i]];
                            }
                            tv.setText(resultStr);

                            stream.close();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        }
    }

    public static float[] byteArrayToFloatArray(byte[] bytes)
    {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        FloatBuffer fb = buffer.asFloatBuffer();
        float[] floatArray = new float[fb.limit()];
        fb.get(floatArray);
        return floatArray;
    }

    public static byte[] bitmap2BGR(Bitmap bitmap)
    {
        int bytes = bitmap.getByteCount();

        ByteBuffer buffer = ByteBuffer.allocate(bytes);
        bitmap.copyPixelsToBuffer(buffer);

        byte[] rgba = buffer.array();
        byte[] pixels = new byte[(rgba.length / 4) * 3];

        int count = rgba.length / 4;

        for (int i = 0; i < count; i++) {
            pixels[i * 3] = rgba[i * 4 + 2];      //B
            pixels[i * 3 + 1] = rgba[i * 4 + 1];  //G
            pixels[i * 3 + 2] = rgba[i * 4];      //R
        }
        return pixels;
    }

    public static byte[] rotateYUV420Degree90(byte[] data, int imageWidth, int imageHeight)
    {
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
                yuv[i] = data[(imageWidth * imageHeight) + (y * imageWidth) + (x - 1)];
                i--;
            }
        }
        return yuv;
    }

    @Override
    public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height)
    {}

    @Override public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface)
    {
        mCamera.stopPreview();
        mCamera.release();
        mCamera = null;
        return false;
    }

    @Override public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface)
    {}
}
