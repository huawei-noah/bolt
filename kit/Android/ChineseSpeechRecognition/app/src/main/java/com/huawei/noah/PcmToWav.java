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

import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class PcmToWav {

    /**
     * 将一个pcm文件转化为wav文件
     *
     * @param pcmPath         pcm文件路径
     * @param destinationPath 目标文件路径(wav)
     * @param deletePcmFile   是否删除源文件
     * @return
     */

     public static boolean makePCMFileToWAVFile(String pcmPath,String destinationPath,boolean deletePcmFile){
         byte buffer[] = null;
         int total_size=0;
         File file=new File(pcmPath);
         if (!file.exists()){
             return false;
         }
         total_size=(int)file.length();

         WavHeader header=new WavHeader();
         header.fileLength=total_size+(44-8);
         header.FmtHdrLeth=16;
         header.BitsPerSample=16;
         header.Channels=1;
         header.FormatTag=0x0001;
         header.SamplesPerSec=16000;
         header.BlockAlign=(short)(header.Channels*header.BitsPerSample/8);
         header.AvgBytesPerSec=header.BlockAlign*header.SamplesPerSec;
         header.DataHdrLeth=total_size;

         byte[] h=null;
         try{
             h=header.getHeader();
         }catch (IOException e){
             Log.e("PcmToWav",e.getMessage());
             return false;
         }

         if (h.length!=44)//WAV标准，头部应该是44字节,如果不是44个字节则不进行转换文件
         {
             return false;
         }

         File destfile=new File(destinationPath);
         if (destfile.exists()){
             destfile.delete();
         }

         try {
             buffer=new byte[1024*4];
             InputStream inputStream=null;
             OutputStream outputStream=null;

             outputStream=new BufferedOutputStream(new FileOutputStream(destinationPath));
             outputStream.write(h,0,h.length);
             inputStream=new BufferedInputStream(new FileInputStream(file));
             int size=inputStream.read(buffer);
             while (size!=-1){
                 outputStream.write(buffer);
                 size=inputStream.read(buffer);
             }
             inputStream.close();
             outputStream.close();

         } catch (FileNotFoundException e) {
             e.printStackTrace();
             return false;
         } catch (IOException e) {
             e.printStackTrace();
             return false;
         }

         if (deletePcmFile){
             file.delete();
         }
         Log.i("PcmToWav", "makePCMFileToWAVFile  success!");
         return true;

     }
}
