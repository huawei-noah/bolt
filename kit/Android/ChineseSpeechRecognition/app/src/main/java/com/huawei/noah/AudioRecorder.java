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

import android.content.Context;
import android.content.ContextWrapper;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Environment;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AudioRecorder {

    private  static AudioRecorder mInstance;
    private AudioRecord audioRecord;

    // 音频源：音频输入-麦克风
    private  final  static int audioSource= MediaRecorder.AudioSource.MIC;
    //采样率
    private  static  int audioRate=16000;
    //录音的声道，单声道
    private  static int audioChannel= AudioFormat.CHANNEL_IN_MONO;
    //量化的深度
    private  static int audioFormat=AudioFormat.ENCODING_PCM_16BIT;
    //缓存的大小
    private  static  int bufferSize=AudioRecord.getMinBufferSize(audioRate,audioChannel,audioFormat);
    //录音状态
    private Status status = Status.STATUS_NO_READY;


    //WAV文件
    private File wavFile;
    //文件输出流
    private OutputStream os;
    //线程池
    private ExecutorService mExecutorService;

    public String pcmFileName;
    //wav文件目录
    public String wavFileName;


    private AudioRecorder(){
        audioRecord=new AudioRecord(audioSource,audioRate,audioChannel,audioFormat,bufferSize);
        status = Status.STATUS_READY;
        mExecutorService= Executors.newCachedThreadPool();
    }

    public synchronized static AudioRecorder getInstance(){
        if (mInstance==null){
            mInstance=new AudioRecorder();
        }
        return mInstance;
    }


    public void startRecording(){
        if (status == Status.STATUS_NO_READY) {
            throw new IllegalStateException("录音尚未初始化,请检查是否禁止了录音权限~");
        }
        if (status == Status.STATUS_START) {
            throw new IllegalStateException("正在录音");
        }
        audioRecord.startRecording();

        mExecutorService.execute(new Runnable() {
            @Override
            public void run() {
                writeDataTOFile();
            }
        });

    }

    public void stopRecording(){
        if (status == Status.STATUS_NO_READY || status == Status.STATUS_READY) {
            Log.i("2","录音尚未开始");
        } else {
            audioRecord.stop();
            WavHeader wavHeader=new WavHeader();

            mExecutorService.execute(new Runnable() {
                @Override
                public void run() {

                    PcmToWav.makePCMFileToWAVFile(pcmFileName,wavFileName,true);
                }
            });
            status = Status.STATUS_STOP;
        }
    }

    public void writeDataTOFile(){

        File dir = new File(MyApplication.getContext().getExternalFilesDir(null)
                .getAbsolutePath(), "yuyin");
        if (!dir.exists()) {
            dir.mkdirs();
        }

        pcmFileName=dir+"/my.pcm";
        wavFileName=dir+"/my.wav";

        Log.i("1",pcmFileName);

        // new一个byte数组用来存一些字节数据，大小为缓冲区大小
        byte[] audiodata = new byte[bufferSize];

        FileOutputStream fos = null;
        int readsize = 0;
        try {
            File file = new File(pcmFileName);
            if (file.exists()) {
                file.delete();
            }
            fos = new FileOutputStream(file);// 建立一个可存取字节的文件
        } catch (IllegalStateException e) {
            Log.e("AudioRecorder", e.getMessage());
            throw new IllegalStateException(e.getMessage());
        } catch (FileNotFoundException e) {
            Log.e("AudioRecorder", e.getMessage());

        }
        //将录音状态设置成正在录音状态
        status = Status.STATUS_START;
        while (status == Status.STATUS_START) {
            readsize = audioRecord.read(audiodata, 0, bufferSize);
            if (AudioRecord.ERROR_INVALID_OPERATION != readsize && fos != null) {
                try {
                    fos.write(audiodata);

                } catch (IOException e) {
                    Log.e("AudioRecorder", e.getMessage());
                }
            }
        }
        try {
            if (fos != null) {
                fos.close();// 关闭写入流
            }
        } catch (IOException e) {
            Log.e("AudioRecorder", e.getMessage());
        }
    }


    public byte[] getWavHeader(long totalAudioLen){
        int mChannels = 1;
        long totalDataLen = totalAudioLen + 36;
        long longSampleRate = audioRate;
        long byteRate = audioRate * 2 * mChannels;

        byte[] header = new byte[44];
        header[0] = 'R';  // RIFF/WAVE header
        header[1] = 'I';
        header[2] = 'F';
        header[3] = 'F';
        header[4] = (byte) (totalDataLen & 0xff);
        header[5] = (byte) ((totalDataLen >> 8) & 0xff);
        header[6] = (byte) ((totalDataLen >> 16) & 0xff);
        header[7] = (byte) ((totalDataLen >> 24) & 0xff);
        header[8] = 'W';
        header[9] = 'A';
        header[10] = 'V';
        header[11] = 'E';
        header[12] = 'f';  // 'fmt ' chunk
        header[13] = 'm';
        header[14] = 't';
        header[15] = ' ';
        header[16] = 16;  // 4 bytes: size of 'fmt ' chunk
        header[17] = 0;
        header[18] = 0;
        header[19] = 0;
        header[20] = 1;  // format = 1
        header[21] = 0;
        header[22] = (byte) mChannels;
        header[23] = 0;
        header[24] = (byte) (longSampleRate & 0xff);
        header[25] = (byte) ((longSampleRate >> 8) & 0xff);
        header[26] = (byte) ((longSampleRate >> 16) & 0xff);
        header[27] = (byte) ((longSampleRate >> 24) & 0xff);
        header[28] = (byte) (byteRate & 0xff);
        header[29] = (byte) ((byteRate >> 8) & 0xff);
        header[30] = (byte) ((byteRate >> 16) & 0xff);
        header[31] = (byte) ((byteRate >> 24) & 0xff);
        header[32] = (byte) (2 * mChannels);  // block align
        header[33] = 0;
        header[34] = 16;  // bits per sample
        header[35] = 0;
        header[36] = 'd';
        header[37] = 'a';
        header[38] = 't';
        header[39] = 'a';
        header[40] = (byte) (totalAudioLen & 0xff);
        header[41] = (byte) ((totalAudioLen >> 8) & 0xff);
        header[42] = (byte) ((totalAudioLen >> 16) & 0xff);
        header[43] = (byte) ((totalAudioLen >> 24) & 0xff);

        return header;
    }



    public enum Status {
        //未开始
        STATUS_NO_READY,
        //预备
        STATUS_READY,
        //录音
        STATUS_START,
        //暂停
        STATUS_PAUSE,
        //停止
        STATUS_STOP
    }



}
