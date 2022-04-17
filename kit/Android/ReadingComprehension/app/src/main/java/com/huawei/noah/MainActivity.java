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

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.huawei.noah.bert.AppTokenizer;
import com.huawei.noah.bert.PredictionModel;
import com.huawei.noah.databinding.ActivityMainBinding;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private ActivityMainBinding binding;
    private EditText content;
    private EditText question;
    private TextView answer;
    private static final String VOCAB = "vocab.txt";
    private static final String MODEL = "bert_squad_10_f32.bolt";
    private String modelPath;
    private AppTokenizer appTokenizer;
    private ProgressBar progressBar;

    private ExecutorService executorService;
    @Override protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        executorService = Executors.newFixedThreadPool(1);
        content = binding.content;
        question = binding.question;
        answer = binding.answer;
        progressBar = binding.progress;

        findViewById(R.id.demo1).setOnClickListener(this);
        findViewById(R.id.demo2).setOnClickListener(this);
        findViewById(R.id.ask_button).setOnClickListener(this);

        String vocab = getCacheDir() + File.separator + VOCAB;
        modelPath = getCacheDir() + File.separator + MODEL;

        try {
            copyAssetResource2File(VOCAB, vocab);
            copyAssetResource2File(MODEL, modelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        appTokenizer = new AppTokenizer(vocab);
    }

    private void copyAssetResource2File(String assetsFile, String outFile) throws IOException
    {
        File outF = new File(outFile);
        if (outF.exists())
            return;
        InputStream is = this.getAssets().open(assetsFile);
        FileOutputStream fos = new FileOutputStream(outF);
        int byteCount;
        byte[] buffer = new byte[1024];
        while ((byteCount = is.read(buffer)) != -1) {
            fos.write(buffer, 0, byteCount);
        }
        fos.flush();
        is.close();
        fos.close();
        outF.setReadable(true);
    }

    @Override public void onClick(View v)
    {
        switch (v.getId()) {
            case R.id.ask_button: {
                if (content.getText().toString().length() == 0) {
                    Toast
                        .makeText(
                            getApplicationContext(), "Content can not be null", Toast.LENGTH_LONG)
                        .show();
                    return;
                } else if (question.getText().toString().length() == 0) {
                    Toast
                        .makeText(
                            getApplicationContext(), "Question can not be null", Toast.LENGTH_LONG)
                        .show();
                    return;
                }

                progressBar.setVisibility(View.VISIBLE);
                executorService.submit(new Runnable() {
                    @Override public void run()
                    {
                        float[][] tokenizers = appTokenizer.runTokenizer(
                            content.getText().toString(), question.getText().toString());
                        int[] inputCActual = {
                            tokenizers[0].length, tokenizers[1].length, tokenizers[2].length};
                        int inputNum = 3;
                        int outputNum = 2;
                        String[] inputName = {"input_ids:0", "input_mask:0", "segment_ids:0"};
                        String[] outputName = {"unstack:0", "unstack:1"};
                        int[] inputN = {1, 1, 1};
                        int[] inputCMax = {256, 256, 256};
                        int[] inputH = {1, 1, 1};
                        int[] inputW = {1, 1, 1};
                        DataType[] inputDatatype = {DataType.INT32, DataType.INT32, DataType.INT32};
                        DataFormat[] inputDataFormat = {
                            DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
                        BoltModel boltModel = new BoltModel(modelPath,
                            AffinityType.CPU_HIGH_PERFORMANCE, inputNum, inputName, inputN,
                            inputCMax, inputH, inputW, inputDatatype, inputDataFormat, outputNum,
                            outputName);
                        BoltResult boltResult = boltModel.run(inputNum, inputName, inputN,
                            inputCActual, inputH, inputW, inputDatatype, inputDataFormat,
                            tokenizers);
                        float[][] result = boltResult.getResultData();
                        String resultStr = getResultAnswer(result);
                        boltModel.destructor();
                        doOnUiCode(resultStr);
                    }
                });

            } break;
            case R.id.demo1: {
                content.setText(getString(R.string.Demo1));
                question.setText(getString(R.string.Ques1));
                answer.setText("");
            } break;
            case R.id.demo2: {
                content.setText(getString(R.string.Demo2));
                question.setText(getString(R.string.Ques2));
                answer.setText("");
            } break;

            default:
                break;
        }
    }

    private void doOnUiCode(String string)
    {
        Handler uiThread = new Handler(Looper.getMainLooper());
        uiThread.post(new Runnable() {
            @Override public void run()
            {
                answer.setText(string);
                progressBar.setVisibility(View.GONE);
            }
        });
    }

    private String getResultAnswer(float[][] result)
    {
        ArrayList<Integer> start_index = getBestIndexs(result[0], 20);
        ArrayList<Integer> end_index = getBestIndexs(result[1], 20);
        ArrayList<PredictionModel> predictionModels = new ArrayList<>();
        for (int start : start_index) {
            for (int end : end_index) {
                predictionModels.add(
                    new PredictionModel(start, end, result[0][start], result[1][end]));
            }
        }
        Collections.sort(predictionModels, new Comparator<PredictionModel>() {
            @Override public int compare(PredictionModel o1, PredictionModel o2)
            {
                if ((o1.start_logit + o1.end_logit) >= (o2.start_logit + o2.end_logit)) {
                    return -1;
                } else
                    return 1;
            }
        });

        PredictionModel predictionModel = predictionModels.get(2);
        String tok = "";
        for (int i = predictionModel.start; i <= predictionModel.end; i++) {
            if (appTokenizer.features_.get(i).contains("##")) {
                String s = appTokenizer.features_.get(i).substring(
                    appTokenizer.features_.get(i).indexOf("##") + 2);
                tok += s;
            } else {
                if (i == predictionModel.start) {
                    tok += appTokenizer.features_.get(i);
                } else {
                    tok += " " + appTokenizer.features_.get(i);
                }
            }
        }
        return tok;
    }

    private ArrayList<Integer> getBestIndexs(float[] datas, int bestSize)
    {
        ArrayList<Integer> results = new ArrayList<>();
        Map<Float, Integer> unstack = new TreeMap<Float, Integer>(new Comparator<Float>() {
            @Override public int compare(Float o1, Float o2)
            {
                return o2.compareTo(o1);
            }
        });

        for (int i = 0; i < 256; i++) {
            unstack.put(datas[i], i);
        }

        int index = 0;
        for (Iterator i = unstack.values().iterator(); i.hasNext();) {
            if (index >= bestSize)
                break;
            Object obj = i.next();
            results.add((int)obj);
            index++;
        }
        return results;
    }
}
