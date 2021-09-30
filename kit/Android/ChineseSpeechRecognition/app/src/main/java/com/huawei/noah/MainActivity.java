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

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements View.OnClickListener{

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    public native void initFlow(String encoderPath,String predicPath,String jointPath,String pinyinPath,String labelPath);
    public native String runFlow(String wavFilePath);

    private ImageView yuyinImg;
    private TextView chineseWord;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        yuyinImg=findViewById(R.id.imageView);
        chineseWord=findViewById(R.id.textView2);
        findViewById(R.id.button).setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {

                AudioRecorder audioRecorder=AudioRecorder.getInstance();
                if(v.getId()==R.id.button)
                {
                    if (event.getAction()==MotionEvent.ACTION_DOWN){
                        Log.i("1","开始录音");
                        audioRecorder.startRecording();
                        yuyinImg.setVisibility(View.VISIBLE);
                    }else if (event.getAction()==MotionEvent.ACTION_UP){
                        Log.i("2","结束录音");
                        audioRecorder.stopRecording();
                        yuyinImg.setVisibility(View.GONE);
                    }
                }

                return false;
            }
        });

        findViewById(R.id.button2).setOnClickListener(this);
        findViewById(R.id.button3).setOnClickListener(this);


        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.RECORD_AUDIO) !=
                PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this, new String[] {android.Manifest.permission.RECORD_AUDIO}, 50);
        }


        copyAssetAndWrite("encoder_flow.prototxt");
        copyAssetAndWrite("joint_flow.prototxt");
        copyAssetAndWrite("pinyin2hanzi_flow.prototxt");
        copyAssetAndWrite("prediction_flow.prototxt");
        copyAssetAndWrite("cnn_pinyin_lm_b7h512e4_cn_en_20200518_cloud_fp32_f32.bolt");
        copyAssetAndWrite("asr_convolution_transformer_encoder_f32.bolt");
        copyAssetAndWrite("asr_convolution_transformer_prediction_net_f32.bolt");
        copyAssetAndWrite("asr_convolution_transformer_joint_net_f32.bolt");
        copyAssetAndWrite("asr_labels.txt");
        copyAssetAndWrite("pinyin_lm_embedding.bin");
        // Example of a call to a native method

        initFlow(getCacheDir()+"/encoder_flow.prototxt",getCacheDir()+"/prediction_flow.prototxt",
                getCacheDir()+"/joint_flow.prototxt",getCacheDir()+"/pinyin2hanzi_flow.prototxt",getCacheDir()+"/asr_labels.txt");
    }

    private boolean copyAssetAndWrite(String fileName) {
        try {
            File cacheDir=getCacheDir();
            if (!cacheDir.exists()) {
                cacheDir.mkdirs();
            }
            File outFile = new File(cacheDir, fileName);
            if (!outFile.exists()) {
                boolean res=outFile.createNewFile();
                if (!res) {
                    return false;
                }
            } else {
                if (outFile.length() > 0) {
                    return true;
                }
            }
            InputStream is=getAssets().open(fileName);
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

    @Override
    public void onClick(View v) {
        switch (v.getId()){
            case R.id.button2:
            {
                File dir = new File(MyApplication.getContext().getExternalFilesDir(null)
                        .getAbsolutePath(), "yuyin");
                String wavFileName=dir+"/my.wav";
                MediaPlayer mediaPlayer=new MediaPlayer();
                try {
                    mediaPlayer.setDataSource(wavFileName);
                    mediaPlayer.prepare();
                    mediaPlayer.start();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

                break;

            case R.id.button3:
            {
                File dir = new File(MyApplication.getContext().getExternalFilesDir(null)
                        .getAbsolutePath(), "yuyin");
                String wavFileName=dir+"/my.wav";

                chineseWord.setText(runFlow(wavFileName));

            }

                break;
        }
    }


    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

}