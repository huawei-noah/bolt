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
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.app.ActivityManager;
import android.content.Context;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Debug;
import android.text.TextUtils;
import android.text.format.Formatter;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Adapter;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.huawei.noah.bert.AppTokenizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private static final String TAG = "MainActivity";
    private static final String VOCAB = "vocab.txt";
    private static final String MODEL = "tinybert_f32.bolt";

    int selIndex = 0;

    private EditText etInput;
    private TextView tvIntent;
    private TextView tvTime;

    private AppTokenizer appTokenizer;
    private RecyclerView recyclerView;
    private String[] quickList;
    private ImageView toggle;

    private String modelPath;

    @Override protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        etInput = findViewById(R.id.et_input);
        tvIntent = findViewById(R.id.tv_intent);
        tvTime = findViewById(R.id.tv_time);
        findViewById(R.id.btn_detect).setOnClickListener(this);

        findViewById(R.id.ic_clear).setOnClickListener(this);
        toggle = findViewById(R.id.ic_toggle);
        toggle.setOnClickListener(this);

        String vocab = getCacheDir() + File.separator + VOCAB;
        modelPath = getCacheDir() + File.separator + MODEL;
        try {
            copyAssetResource2File(VOCAB, vocab);
            copyAssetResource2File(MODEL, modelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Log.i(TAG, "onCreate: vocab = " + vocab + ", model path = " + modelPath);
        appTokenizer = new AppTokenizer(vocab);

        quickList = getResources().getStringArray(R.array.quick_list);
        recyclerView = findViewById(R.id.recyclerView);
        LinearLayoutManager layoutManager = new LinearLayoutManager(this);
        layoutManager.setOrientation(LinearLayoutManager.VERTICAL);
        recyclerView.setLayoutManager(layoutManager);
        TipAdapter tipAdapter = new TipAdapter();
        recyclerView.setAdapter(tipAdapter);

        etInput.setText(quickList[0]);
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

    class TipAdapter extends RecyclerView.Adapter<TipAdapter.ViewHolder> {
        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType)
        {
            View view =
                LayoutInflater.from(parent.getContext()).inflate(R.layout.item, parent, false);
            return new ViewHolder(view);
        }

        private View lastSelView;
        @Override public void onBindViewHolder(@NonNull final ViewHolder holder, final int position)
        {
            holder.textView.setText(position + ". " + quickList[position]);

            if (selIndex == position) {
                if (lastSelView != null && lastSelView != holder.textView) {
                    lastSelView.setBackgroundColor(Color.rgb(255, 255, 255));
                }
                holder.textView.setBackgroundColor(Color.rgb(0, 230, 0));
                lastSelView = holder.textView;
            } else {
                holder.textView.setBackgroundColor(Color.rgb(255, 255, 255));
            }
            holder.textView.setOnClickListener(new View.OnClickListener() {
                @Override public void onClick(View v)
                {
                    etInput.setText(quickList[position]);
                    etInput.setSelection(quickList[position].length());
                    recyclerView.setVisibility(View.GONE);
                    toggle.setImageResource(R.drawable.ic_find_next_holo_light);

                    selIndex = position;
                    notifyItemChanged(position);
                }
            });
        }

        @Override public int getItemCount()
        {
            return quickList.length;
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView textView;

            ViewHolder(@NonNull View itemView)
            {
                super(itemView);
                textView = itemView.findViewById(R.id.text);
            }
        }
    }

    private void detect(String sentence)
    {
        int inputNum = 3;
        int outputNum = 1;
        String[] inputName = {"input_ids", "position_ids", "token_type_ids"};
        String[] outputName = {"logit"};
        int[] inputN = {1, 1, 1};
        int[] inputCMax = {64, 64, 64};
        int[] inputH = {1, 1, 1};
        int[] inputW = {1, 1, 1};
        DataType[] intputDataType = {DataType.INT32, DataType.INT32, DataType.INT32};
        DataFormat[] intputDataFormat = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel boltModel = new BoltModel(modelPath, AffinityType.CPU_HIGH_PERFORMANCE, inputNum,
            inputName, inputN, inputCMax, inputH, inputW, intputDataType, intputDataFormat,
            outputNum, outputName);

        float[][] tokenizers = appTokenizer.runTokenizer(sentence);
        int[] inputCActual = {tokenizers[0].length, tokenizers[1].length, tokenizers[2].length};

        long start = System.currentTimeMillis();
        BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputCActual, inputH,
            inputW, intputDataType, intputDataFormat, tokenizers);
        float[][] result = boltResult.getResultData();

        if (result[0][0] > result[0][1]) {
            tvIntent.setText("negative");
        } else {
            tvIntent.setText("positive");
        }

        String memory = getRunningAppProcessInfo();
        long time = System.currentTimeMillis() - start;
        tvTime.setText("time：" + time + "ms\nstorage：" + memory);
    }

    @Override public void onClick(View v)
    {
        switch (v.getId()) {
            case R.id.btn_detect:
                // trim to remove blank
                String input = etInput.getText().toString().trim();
                if (TextUtils.isEmpty(input)) {
                    Toast.makeText(this, etInput.getHint(), Toast.LENGTH_SHORT).show();
                    return;
                }
                detect(input);
                break;

            case R.id.ic_clear:
                etInput.setText("");
                break;
            case R.id.ic_toggle:
                int visbility = recyclerView.getVisibility() == View.VISIBLE ? View.GONE
                                                                             : View.VISIBLE;
                recyclerView.setVisibility(visbility);
                if (visbility == View.VISIBLE) {
                    toggle.setImageResource(R.drawable.ic_find_previous_holo_light);
                } else {
                    toggle.setImageResource(R.drawable.ic_find_next_holo_light);
                }
                break;
        }
    }

    private String getRunningAppProcessInfo()
    {
        ActivityManager manager = (ActivityManager)getSystemService(Context.ACTIVITY_SERVICE);
        List<ActivityManager.RunningAppProcessInfo> appProcessList =
            manager.getRunningAppProcesses();

        for (ActivityManager.RunningAppProcessInfo appProcessInfo : appProcessList) {
            String processName = appProcessInfo.processName;
            if (!TextUtils.equals(processName, getPackageName()))
                continue;
            // get memory info
            int[] myMempid = new int[] {appProcessInfo.pid};
            Debug.MemoryInfo[] memoryInfo = manager.getProcessMemoryInfo(myMempid);
            // memory usgae(KB)
            int dalvikPss = memoryInfo[0].dalvikPss;
            int nativePss = memoryInfo[0].nativePss;
            int otherPss = memoryInfo[0].otherPss;
            return Formatter.formatFileSize(this, (dalvikPss + nativePss + otherPss) * 1024);
        }

        return null;
    }
}
