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
    public List<String> features_ = new ArrayList<>();

    public AppTokenizer(String vocab)
    {
        this.maxSeqLength = 256;
        tokenizer = new FullTokenizer(vocab);
    }

    public float[][] runTokenizer(String paragraph, String question)
    {
        List<String> paragraph_tokens = tokenizer.tokenize(paragraph);
        List<String> feture1 = tokenizer.getFeaturesList();
        List<String> question_tokens = tokenizer.tokenize(question);
        List<String> feture2 = tokenizer.getFeaturesList();

        return getExampleSingle(tokenizer, paragraph_tokens, question_tokens, feture1, feture2);
    }

    private float[][] getExampleSingle(FullTokenizer tokenizer,
        List<String> paragraph_tokens,
        List<String> question_tokens,
        List<String> feature1,
        List<String> feature2)
    {
        tokens = new ArrayList<>();
        List<Integer> segmentIds = new ArrayList<>();
        List<Integer> positions = new ArrayList<Integer>();

        features_.clear();
        features_.add("[CLS]");
        tokens.add("[CLS]");

        for (int i = 0; i < question_tokens.size(); i++) {
            tokens.add(question_tokens.get(i));
            features_.add(feature2.get(i));
        }
        tokens.add("[SEP]");
        features_.add("[SEP]");

        for (int i = 0; i < paragraph_tokens.size(); i++) {
            tokens.add(paragraph_tokens.get(i));
            features_.add(feature1.get(i));
        }
        tokens.add("[SEP]");
        features_.add("[SEP]");

        List<Integer> inputIds = tokenizer.convertTokensToIds(tokens);
        for (int i = 0; i < maxSeqLength; i++) {
            if (i < inputIds.size()) {
                if (i < question_tokens.size() + 2) {
                    segmentIds.add(0);
                } else {
                    segmentIds.add(1);
                }
                positions.add(1);
            } else {
                inputIds.add(0);
                segmentIds.add(0);
                positions.add(0);
            }
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

        tokenSize = paragraph_tokens.size() + 2 + question_tokens.size() + 1;
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
