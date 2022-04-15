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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FullTokenizer {
    private Map<String, Integer> vocab;
    private boolean doLowerCase;
    private BasicTokenizer basicTokenizer;
    private WordpieceTokenizer wordpieceTokenizer;

    public FullTokenizer(String vocab_file, boolean doLowerCase)
    {
        this.vocab = load(vocab_file);
        this.doLowerCase = doLowerCase;
        this.basicTokenizer = new BasicTokenizer();
        this.wordpieceTokenizer = new WordpieceTokenizer(vocab);
    }

    private Map<String, Integer> load(String filePath)
    {
        Map<String, Integer> map = new HashMap<String, Integer>();
        try {
            BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(new File(filePath)), "UTF-8"));
            int index = 0;
            String token = null;
            while ((token = br.readLine()) != null) {
                map.put(token, index);
                index += 1;
            }
            br.close();
        } catch (Exception e) {
            System.err.println("read errors :" + e);
        }
        return map;
    }

    public List<String> tokenize(String text)
    {
        if (this.doLowerCase) {
            text = text.toLowerCase();
        }
        List<String> splitTopkens = new ArrayList<String>();

        for (String token : basicTokenizer.tokenize(text)) {
            for (String subToken : wordpieceTokenizer.tokenize(token)) {
                splitTopkens.add(subToken);
            }
        }

        return splitTopkens;
    }

    public List<Integer> convertTokensToIds(List<String> tokens)
    {
        List<Integer> outputIds = new ArrayList<Integer>();
        for (String token : tokens) {
            outputIds.add(this.vocab.get(token));
        }
        return outputIds;
    }
}
