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

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.content.ContentUris;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.io.File;

public class PhotoActivity extends AppCompatActivity {

    private ImageView resultImg;
    private String modelPath;
    File cameraSavePath;
    String inputImgPath;
    String resultImgPath;
    BoltResult boltResult;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo);

        modelPath=getIntent().getStringExtra("modelPath");
        initView();
        boltResult=new BoltResult();
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

    private void initView(){
        resultImg=findViewById(R.id.imageView4);

        findViewById(R.id.button6).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });

        Button cameraBt=findViewById(R.id.button7);
        cameraBt.setBackgroundResource(R.drawable.camera_white1);
        cameraBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AlertDialog alertDialog;
                AlertDialog.Builder builder=new AlertDialog.Builder(PhotoActivity.this);
                builder.setTitle("图片来源");
                builder.setNegativeButton("取消",null);
                builder.setItems(new String[]{"拍照", "相册"}, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which){
                        switch (which){
                            case 0:
                                cameraPhoto();
                                break;
                            case 1:
                                selMyPhoto();
                                break;
                        }
                    }
                });
                alertDialog=builder.create();
                alertDialog.show();
            }
        });
    }

    //    拍照获取
    private void cameraPhoto(){
        cameraSavePath=new File(getExternalCacheDir(),"output_image"+String.valueOf(System.currentTimeMillis())+".jpg");

        Uri imageUri;
        if (Build.VERSION.SDK_INT>=Build.VERSION_CODES.N){
            //大于等于版本24（7.0）的场合
            imageUri= FileProvider.getUriForFile(this,"com.huawei.noah.fileProvider",cameraSavePath);
        }else
        {
            imageUri=Uri.fromFile(cameraSavePath);
        }

        Intent intent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT,imageUri);
        startActivityForResult(intent,103);
    }

    //    从系统相册选择
    private void  selMyPhoto(){
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},100);
        }else if (ContextCompat.checkSelfPermission(this,Manifest.permission.READ_EXTERNAL_STORAGE)!=PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},102);
        }else{
            Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent,101);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode==101&&data!=null){
            handleImageOnKitKat(data);
        }else if(requestCode==103&&resultCode!=RESULT_CANCELED){
            inputImgPath=cameraSavePath.getPath();
            Bitmap bitmap=BitmapFactory.decodeFile(inputImgPath);
            showResultImg(bitmap,modelPath);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    private void handleImageOnKitKat(Intent data){
        Uri uri=data.getData();
        if (DocumentsContract.isDocumentUri(this,uri)){
            String docId=DocumentsContract.getDocumentId(uri);
            if ("com.android.providers.media.documents".equals(uri.getAuthority())){
                String id=docId.split(":")[1];
                String selection= MediaStore.Images.Media._ID+"="+id;
                inputImgPath=getImagePath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,selection);
            }else if ("com.android.providers.downloads.documents".equals(uri.getAuthority())){
                Uri contentUri= ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"),Long.valueOf(docId));
                inputImgPath=getImagePath(contentUri,null);
            }
        }else if("content".equalsIgnoreCase(uri.getScheme())){// 如果是content类型的Uri，则使用普通方式处理
            inputImgPath=getImagePath(uri,null);
        }else if("file".equalsIgnoreCase(uri.getScheme())){// 如果是file类型的Uri，直接获取图片路径即可
            inputImgPath=uri.getPath();
        }

        if (inputImgPath!=null){
            Bitmap bitmap=BitmapFactory.decodeFile(inputImgPath);
            showResultImg(bitmap,modelPath);
        }
    }

    private String getImagePath(Uri uri,String selection){
        String path=null;
        Cursor cursor=getContentResolver().query(uri,null,selection,null,null);
        if (cursor!=null){
            if (cursor.moveToFirst()){
                path=cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DATA));
            }
            cursor.close();
        }
        return path;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        boltResult.destroyBolt();
    }
}
