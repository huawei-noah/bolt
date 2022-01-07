/**
 * @file
 * @brief Java BoltResult Class Document
 *
 * @copyright
 * @code
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * @endcode
 */
package com.huawei.noah;

public class BoltResult {
    public BoltResult(float[][] value, int[][] dimension, String[] name, String[] dataFormat)
    {
        this.value = value;
        this.dimension = dimension;
        this.name = name;
        this.dataFormat = dataFormat;
    }

    /**
     * @brief get result data name from BoltResult object
     *
     * @return 1d String array of output data in the inference result
     */
    public String[] getResultName()
    {
        return this.name;
    }

    /**
     * @brief get result data format from BoltResult object
     *
     * @return 1d String array of output data in the inference result
     */
    public String[] getResultDataFormat()
    {
        return this.dataFormat;
    }

    /**
     * @brief get result data dimension information from BoltResult object
     *
     * @return 2d int array of output data in the inference result
     */
    public int[][] getResultDimension()
    {
        return this.dimension;
    }

    /**
     * @brief get result data array from BoltResult object
     *
     * @return 2d float array of output data in the inference result
     */
    public float[][] getResultData()
    {
        return this.value;
    }

    /**
     * @brief print BoltResult object info
     * @param  num	the number of the result you want
     *
     * @return
     */
    public void print(int num)
    {
        for (int i = 0; i < name.length; i++) {
            System.out.println("[INFO] output name: " + name[i]);
            System.out.println("       data format: " + dataFormat[i]);
            int len = calculateLength(this.dimension[i]);
            System.out.println("       data number: " + len);
            if (num >= 0) {
                if (num < len) {
                    len = num;
                }
            }

            for (int j = 0; j < len; j++) {
                System.out.print(value[i][j] + " ");
            }
            System.out.println();
        }
    }

    /** 2d float array of output data in the inference result, the length of value is output size */
    private float[][] value;

    /** 2d int array of output dimension info in the inference result, the length of dimension is output size */
    private int[][] dimension;

    /** String array of output names info in the inference result, the length of name is output size */
    private String[] name;

    /** String array of output data info in the inference result, the length of dataFormat is output size */
    private String[] dataFormat;

    /** calculate product and skip 0 */
    public static int calculateLength(int[] array)
    {
        int num = array.length;
        int length = 0;
        for (int j = 0; j < num; j++) {
            if (array[j] == 0) {
                break;
            } else {
                if (length == 0) {
                    length = array[j];
                } else {
                    length *= array[j];
                }
            }
        }
        return length;
    }
}
