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
        List<String> tokens = whiteSpaceTokenize(text);

        List<String> outputTokens = new ArrayList<String>();
        for (String token : tokens) {
            int length = token.length();
            if (length > this.maxInputCharsPerWord) {
                outputTokens.add(this.unkToken);
                continue;
            }

            boolean isBad = false;
            int start = 0;
            List<String> subTokens = new ArrayList<String>();

            while (start < length) {
                int end = length;
                String curSubStr = null;
                while (start < end) {
                    String subStr = token.substring(start, end);
                    if (start > 0) {
                        subStr = "##" + subStr;
                    }
                    if (this.vocab.containsKey(subStr)) {
                        curSubStr = subStr;
                        break;
                    }
                    end -= 1;
                }
                if (null == curSubStr) {
                    isBad = true;
                    break;
                }
                subTokens.add(curSubStr);
                start = end;
            }

            if (isBad) {
                outputTokens.add(this.unkToken);
            } else {
                outputTokens.addAll(subTokens);
            }
        }
        return outputTokens;
    }

    private List<String> whiteSpaceTokenize(String text)
    {
        List<String> result = new ArrayList<String>();

        text = text.trim();
        if (null == text) {
            return result;
        }
        String[] tokens = text.split(" ");
        result = Arrays.asList(tokens);

        return result;
    }
}
