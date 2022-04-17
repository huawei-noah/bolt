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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class BasicTokenizer {
    public List<String> tokenize(String text)
    {
        String cleanText = cleanText(text);

        String chineseTokens = tokenizeChineseChars(cleanText);

        List<String> origTokens = whiteSpaceTokenize(chineseTokens);

        String str = "";
        for (String token : origTokens) {
            List<String> list = runSplitOnPunc(token);
            for (int i = 0; i < list.size(); i++) {
                str += list.get(i) + " ";
            }
        }

        List<String> resTokens = whiteSpaceTokenize(str);

        return resTokens;
    }

    private List<String> runSplitOnPunc(String token)
    {
        List<List<Character>> result = new ArrayList<List<Character>>();

        int length = token.length();
        int i = 0;
        boolean startNewWord = true;
        while (i < length) {
            char c = token.charAt(i);
            if (isPunctuation(c)) {
                List<Character> list = Arrays.asList(c);
                result.add(list);
                startNewWord = true;
            } else {
                if (startNewWord) {
                    result.add(new ArrayList<Character>());
                }
                startNewWord = false;
                result.get(result.size() - 1).add(c);
            }
            i += 1;
        }

        List<String> res = new ArrayList<String>();
        for (int j = 0; j < result.size(); j++) {
            String str = "";
            for (int k = 0; k < result.get(j).size(); k++) {
                str += result.get(j).get(k);
            }
            res.add(str);
        }
        return res;
    }

    private boolean isPunctuation(char c)
    {
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) ||
            (c >= 123 && c <= 126)) {
            return true;
        }

        if (c == '“' || c == '”' || c == '、' || c == '《' || c == '》' || c == '。' || c == '；' ||
            c == '【' || c == '】') {
            return true;
        }

        return false;
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

    private String tokenizeChineseChars(String cleanText)
    {
        StringBuffer outStrBuf = new StringBuffer();

        for (int i = 0; i < cleanText.length(); i++) {
            char c = cleanText.charAt(i);
            if (isChineseChar(c)) {
                outStrBuf.append(" ");
                outStrBuf.append(c);
                outStrBuf.append(" ");
            } else {
                outStrBuf.append(c);
            }
        }

        return outStrBuf.toString();
    }

    private boolean isChineseChar(char c)
    {
        String s = String.valueOf(c);
        String regex = "[\u4e00-\u9fa5]";
        Pattern p = Pattern.compile(regex);

        Matcher m = p.matcher(s);
        return m.matches();
    }

    private String cleanText(String text)
    {
        StringBuffer outStrBuf = new StringBuffer("");

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (isWhiteSpace(c)) {
                outStrBuf.append(" ");
            } else {
                outStrBuf.append(c);
            }
        }
        return outStrBuf.toString();
    }

    private boolean isWhiteSpace(char c)
    {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            return true;
        }

        return false;
    }

    public static void main(String[] args)
    {
        System.out.print("hello world");
    }
}
