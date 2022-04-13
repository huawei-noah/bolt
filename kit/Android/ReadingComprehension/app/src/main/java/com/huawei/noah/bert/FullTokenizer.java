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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class WordpieceTokenizer {
    private Map<String, Integer> vocab;
    private String unkToken = "[UNK]";
    private int maxInputCharsPerWord = 200;
    private List<String> featuresList = new ArrayList<>();

    public WordpieceTokenizer(Map<String, Integer> vocab)
    {
        this.vocab = vocab;
    }

    /*
        For example:
        input = "unaffable"
        output = ["un", "##aff", "##able"]
    */
    public List<String> tokenize(String text)
    {
        String lowText = text.toLowerCase();

        featuresList.clear();
        List<String> outputTokens = new ArrayList<String>();

        int length = lowText.length();
        if (length > this.maxInputCharsPerWord) {
            outputTokens.add(this.unkToken);
        }

        boolean isBad = false;
        int start = 0;
        List<String> subTokens = new ArrayList<String>();
        List<String> featureTokens = new ArrayList<String>();

        while (start < length) {
            int end = length;
            String curSubStr = null;
            String featureSubStr = null;
            while (start < end) {
                String subStr = lowText.substring(start, end);
                String featureStr = text.substring(start, end);
                if (start > 0) {
                    subStr = "##" + subStr;
                    featureStr = "##" + featureStr;
                }
                if (this.vocab.containsKey(subStr)) {
                    curSubStr = subStr;
                    featureSubStr = featureStr;
                    break;
                }
                end -= 1;
            }
            if (null == curSubStr) {
                isBad = true;
                break;
            }
            subTokens.add(curSubStr);
            featureTokens.add(featureSubStr);
            start = end;
        }

        if (isBad) {
            outputTokens.add(this.unkToken);
            featuresList.add(this.unkToken);
        } else {
            outputTokens.addAll(subTokens);
            featuresList.addAll(featureTokens);
        }

        return outputTokens;
    }

    public List<String> getFeaturesList()
    {
        return featuresList;
    }
}
