// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import com.huawei.noah.BoltResult;

public final class TestUtils {
    public static float[] readSequenceDataFromFile(String pathName, int lineNumber)
    {
        float[] array = {};
        try (FileReader reader = new FileReader(pathName);
             BufferedReader br = new BufferedReader(reader)) {
            String line;
            int lineIndex = 0;
            while ((line = br.readLine()) != null) {
                if (lineIndex == lineNumber) {
                    String[] strArray = line.split(" ");
                    int arraySize = Integer.valueOf(strArray[0]);
                    array = new float[arraySize];
                    for (int i = 0; i < arraySize; i++) {
                        array[i] = Float.valueOf(strArray[1 + i]);
                    }
                } else {
                    lineIndex++;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return array;
    }

    public static double getMillisTime()
    {
        return System.nanoTime() / 1000.0 / 1000.0;
    }

    public static void verify(int[] arrayA, int[] arrayB, int length)
    {
        for (int j = 0; j < length; j++) {
            if (arrayA[j] != arrayB[j]) {
                System.err.println("[ERROR] verify failed " + j + " @ " + arrayA[j] + " " +
                    arrayB[j] + ", in Java API test");
                System.exit(1);
            }
        }
    }

    public static void verify(float[] arrayA, float[] arrayB, int length, float threshold)
    {
        for (int j = 0; j < arrayA.length; j++) {
            if (Math.abs(arrayA[j] - arrayB[j]) > threshold) {
                System.err.println("[ERROR] verify failed " + j + " @ " + arrayA[j] + " " +
                    arrayB[j] + ", in Java API test");
                System.exit(1);
            }
        }
    }

    public static int verify(float[][] arrayA, float[][] arrayB, int[][] dimensions, float threshold)
    {
        if (arrayA.length != arrayB.length || arrayA.length != dimensions.length) {
            System.err.println("[ERROR] unmatch data to verify, in Java API test");
            System.exit(1);
        }

        int sum = 0;
        for (int i = 0; i < dimensions.length; i++) {
            int length = BoltResult.calculateLength(dimensions[i]);
            verify(arrayA[i], arrayB[i], length, threshold);
            sum += length;
        }
        return sum;
    }

    public static int top1(float[] array, int offset, int length)
    {
        int maxIndex = offset;
        for (int i = offset + 1; i < offset + length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
