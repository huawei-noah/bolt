// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
package com.example.cameraenlarge;

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
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.TextureView;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;
import java.util.Optional;

public class MainActivity extends AppCompatActivity implements TextureView.SurfaceTextureListener {
    private TextureView textureView;
    private Camera mCamera;
    private ImageView imageView;
    private ScaleGestureDetector gestureDetector;

    public native void initFlow(String path);
    public native byte[] runFlow(byte[] bgrData, String path);

    int width = 32;
    int height = 32;
    private Object Tag;
    // Used to load the 'native-lib' library on application startup.
    static
    {
        System.loadLibrary("native-lib");
    }

    @Override protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, new String[] {android.Manifest.permission.CAMERA}, 50);
        }

        gestureDetector =
            new ScaleGestureDetector(this, new ScaleGestureDetector.OnScaleGestureListener() {
                float mScaleFactor;
                @Override public boolean onScale(ScaleGestureDetector detector)
                {
                    if (detector.getCurrentSpan() > mScaleFactor) {
                        zoomOut();
                    } else {
                        zoomIn();
                    }
                    mScaleFactor = detector.getCurrentSpan();
                    return false;
                }

                @Override public boolean onScaleBegin(ScaleGestureDetector detector)
                {
                    mScaleFactor = detector.getCurrentSpan();
                    return true;
                }

                @Override public void onScaleEnd(ScaleGestureDetector detector)
                {
                    mScaleFactor = detector.getCurrentSpan();
                }
            });

        copyAssetAndWrite("image_classification.prototxt");
        copyAssetAndWrite("esr_1_f32.bolt");
        File prototxt = new File(getCacheDir(), "image_classification.prototxt");
        if (prototxt.exists()) {
            Log.i((String)Tag, "Prototxt exists");
        } else {
            Log.i((String)Tag, "Prototxt does not exist");
        }
        initFlow(prototxt.getAbsolutePath());

        initView();
        initListener();
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
        textureView = findViewById(R.id.textureView);
        imageView = findViewById(R.id.imageView);
    }

    private void initListener()
    {
        textureView.setSurfaceTextureListener(this);
        textureView.setOnTouchListener(new View.OnTouchListener() {
            @Override public boolean onTouch(View v, MotionEvent event)
            {
                return gestureDetector.onTouchEvent(event);
            }
        });
    }

    @Override public boolean onTouchEvent(MotionEvent event)
    {
        gestureDetector.onTouchEvent(event);
        return super.onTouchEvent(event);
    }

    public void zoomOut()
    {
        Camera.Parameters parameters = mCamera.getParameters();
        if (!parameters.isZoomSupported())
            return;

        int zoom = parameters.getZoom() + 1;
        if (zoom < parameters.getMaxZoom()) {
            parameters.setZoom(zoom);
            mCamera.setParameters(parameters);
        }
    }

    public void zoomIn()
    {
        Camera.Parameters parameters = mCamera.getParameters();
        if (!parameters.isZoomSupported())
            return;

        int zoom = parameters.getZoom() - 1;
        if (zoom >= 0) {
            parameters.setZoom(zoom);
            mCamera.setParameters(parameters);
        }
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

    @Override
    public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height)
    {
        mCamera = Camera.open();
        mCamera.setDisplayOrientation(90);
        if (mCamera != null) {
            Camera.Parameters parameters = mCamera.getParameters();
            List<Camera.Size> previewSizes = parameters.getSupportedPreviewSizes();
            Camera.Size size = null;
            for (int i = 0; i < previewSizes.size(); i++) {
                Camera.Size tmpSize = previewSizes.get(i);
                if (tmpSize.height >= height && tmpSize.width >= width) {
                    size = tmpSize;
                } else {
                    break;
                }
            }

            parameters.setPreviewSize(size.width, size.height);

            List<String> focusModes = parameters.getSupportedFocusModes();
            if (focusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                mCamera.setParameters(parameters);
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
                            bmp = Bitmap.createScaledBitmap(bmp, width, height, true);

                            byte[] bgrByteArr = bitmap2BGR(bmp);

                            File prototxt = new File(getCacheDir(), "image_classification.prototxt");
                            byte[] result = runFlow(bgrByteArr, prototxt.getAbsolutePath());

                            Bitmap stitchBmp =
                                Bitmap.createBitmap(width * 2, height * 2, Bitmap.Config.ARGB_8888);
                            stitchBmp.copyPixelsFromBuffer(ByteBuffer.wrap(result));

                            imageView.setImageBitmap(stitchBmp);

                            stream.close();
                        }

                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });
        }
    }

    public static byte[] bitmap2BGR(Bitmap bitmap)
    {
        int bytes = bitmap.getByteCount();

        ByteBuffer buffer = ByteBuffer.allocate(bytes);
        bitmap.copyPixelsToBuffer(buffer);

        byte[] rgba = buffer.array();
        byte[] pixels = new byte[(rgba.length / 4) * 3];

        byte[] rArr = new byte[rgba.length / 4];
        byte[] gArr = new byte[rgba.length / 4];
        byte[] bArr = new byte[rgba.length / 4];
        int count = rgba.length / 4;

        for (int i = 0; i < count; i++) {
            rArr[i] = rgba[i * 4];      //R
            gArr[i] = rgba[i * 4 + 1];  //G
            bArr[i] = rgba[i * 4 + 2];  //B
        }
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < count; j++) {
                if (k == 0) {
                    pixels[j] = rArr[j];
                } else if (k == 1) {
                    pixels[count + j] = gArr[j];
                } else {
                    pixels[2 * count + j] = bArr[j];
                }
            }
        }
        return pixels;
    }

    @Override
    public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height)
    {}

    @Override public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface)
    {
        mCamera.stopPreview();
        mCamera.release();
        return false;
    }

    @Override public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface)
    {}

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
}
