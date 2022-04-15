// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
package com.huawei.noah.bert;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

public class AppTokenizer {
    private static final String TAG = "AppTokenizer";

    private int maxSeqLength;
    private int tokenSize;
    private List<String> tokens;
    private FullTokenizer tokenizer;

    public AppTokenizer(String vocab)
    {
        this.maxSeqLength = 64;
        tokenizer = new FullTokenizer(vocab, true);
    }

    public float[][] runTokenizer(String words)
    {
        List<String> tokens = tokenizer.tokenize(words);
        Log.i(TAG, "runTokenizer: " + tokens.toString());
        return getExampleSingle(tokenizer, tokens);
    }

    private float[][] getExampleSingle(FullTokenizer tokenizer, List<String> input_tokens)
    {
        while (true) {
            int totalLength = input_tokens.size();
            if (totalLength <= maxSeqLength - 2) {
                break;
            } else {
                input_tokens.remove(input_tokens.size() - 1);
            }
        }

        tokens = new ArrayList<>();
        List<Integer> segmentIds = new ArrayList<>();

        tokens.add("[CLS]");
        segmentIds.add(0);
        for (String token : input_tokens) {
            tokens.add(token);
            segmentIds.add(0);
        }
        tokens.add("[SEP]");

        segmentIds.add(0);

        List<Integer> inputIds = tokenizer.convertTokensToIds(tokens);

        while (inputIds.size() < maxSeqLength) {
            inputIds.add(0);
            segmentIds.add(0);
        }
        List<Integer> positions = new ArrayList<Integer>();
        for (int i = 0; i < inputIds.size(); i++) {
            positions.add(i);
        }

        float[][] outputs = new float[3][maxSeqLength];
        for (int i = 0; i < inputIds.size(); i++) {
            outputs[0][i] = inputIds.get(i);
        }
        for (int i = 0; i < positions.size(); i++) {
            outputs[1][i] = positions.get(i);
        }

        for (int i = 0; i < segmentIds.size(); i++) {
            outputs[2][i] = segmentIds.get(i);
        }

        tokenSize = input_tokens.size() + 2;
        Log.i(TAG, "getExampleSingle: tokenSize = " + tokenSize);
        return outputs;
    }

    public String getTokens()
    {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 1; i < tokens.size() - 1; i++) {
            stringBuilder.append(tokens.get(i)).append(" ");
        }
        return stringBuilder.toString();
    }

    public int getTokenSize()
    {
        return tokenSize;
    }
}
