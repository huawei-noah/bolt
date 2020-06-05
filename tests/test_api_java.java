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

public final class test_api_java {
    public static float[] readSequenceDataFromFile(String pathName, int lineNumber) {
        float[] array = {};
        try (FileReader reader = new FileReader(pathName);
             BufferedReader br = new BufferedReader(reader)
        ) {
            String line;
            int lineIndex = 0;
            while ((line = br.readLine()) != null) {
                if (lineIndex == lineNumber) {
                    String[] strArray = line.split(" ");
                    int arraySize = Integer.valueOf(strArray[0]);
                    array = new float[arraySize];
                    for (int i = 0; i < arraySize; i++)
                        array[i] = Float.valueOf(strArray[1+i]);
                } else {
                    lineIndex++;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return array;
    }

    public static double getMillisTime() {
        return System.nanoTime() / 1000.0 / 1000.0;
    }

    public static void verify(int[] arrayA, int[] arrayB, int length) {
        for (int j = 0; j < length; j++) {
            if (arrayA[j] != arrayB[j]) {
                System.err.println("[ERROR] verify failed " + j + " @ "+ arrayA[j] + " "
                    + arrayB[j] + ", in Java API test");
                System.exit(1);
            }
        }
    }

    public static void verify(float[] arrayA, float[] arrayB, int length, float threshold) {
        for (int j = 0; j < arrayA.length; j++) {
            if (Math.abs(arrayA[j] - arrayB[j]) > threshold) {
                System.err.println("[ERROR] verify failed " + j + " @ "+ arrayA[j] + " "
                    + arrayB[j] + ", in Java API test");
                System.exit(1);
            }
        }
    }

    public static int verify(float[][] arrayA, float[][] arrayB, int[][] dimensions, float threshold) {
        if (arrayA.length != arrayB.length
            || arrayA.length != dimensions.length) {
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

    public static int top1(float[] array, int offset, int length) {
        int maxIndex = offset;
        for (int i = offset + 1; i < offset + length; i++) {
            if (array[i] > array[maxIndex])
                maxIndex = i;
        }
        return maxIndex;
    }

    public static void tinybert_intent_slot(String outputPrefix, DeviceType device, AffinityType affinity, String modelPath) {
        int num_input = 3;
        int num_output = 2;
        String[] input_names = {"tinybert_words", "tinybert_positions", "tinybert_token_type"};
        String[] output_names = {"intent_softmax", "slot_softmax"};
        int[] n = {1, 1, 1};
        int[] c_max = {32, 32, 32};
        int[] h = {1, 1, 1};
        int[] w = {1, 1, 1};
        DataType[] dts = {DataType.UINT32, DataType.UINT32, DataType.UINT32};
        DataFormat[] dfs = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel bolt_model = new BoltModel(modelPath, affinity, device,
            num_input, input_names, n, c_max, h, w, dts, dfs,
            num_output, output_names);

        int[] c_act = {9, 9, 9};
        float[][] inputData = {
            {101, 2224, 8224, 7341, 2000, 22149, 2000, 2899, 102},
            {0, 1, 2, 3, 4, 5, 6, 7, 8},
            {0, 0, 0, 0, 0, 0, 0, 0, 0}};
        float[][] resultData = {
            {22, 0.999023f},
            {44, 44, 1, 23, 44, 44, 44, 8, 44}};

        double startTime = getMillisTime();
        BoltResult bolt_result = bolt_model.Run(num_input, input_names,
            n, c_act, h, w, dts, dfs,
            inputData);
        double endTime = getMillisTime();
        System.out.println(outputPrefix + bolt_model.DeviceMapping(device) + ", " + bolt_model.AffinityMapping(affinity)
            + ", tinybert " + String.format("%.3f", endTime - startTime)
            + " ms/sequence, model " + modelPath);
        float[][] result = bolt_result.getResultData();
        int[][] dimension = bolt_result.getResultDimension();
        int intentIndex = top1(result[0], 0, result[0].length);
        float[][] finalResult = new float[2][dimension[1][1]];
        finalResult[0][0] = intentIndex;
        finalResult[0][1] = result[0][intentIndex];
        for (int i = 0; i < dimension[1][1]; i++) {
            finalResult[1][i] = top1(result[1], i*dimension[1][2], dimension[1][2]) - i*dimension[1][2];
        }
        int[][] finalDimension = {{1, 2}, {1, dimension[1][1]}};
        int length = verify(resultData, finalResult, finalDimension, 0.1f);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in tinybert, in Java API test");
            System.exit(1);
        }

        // model destroy
        bolt_model.Destructor();
    }

    public static void tinybert_disambiguate(String outputPrefix, DeviceType device, AffinityType affinity, String modelPath, DataType dt) {
        int num_input = 5;
        int num_output = 1;
        String[] input_names = {"tinybert_words", "tinybert_positions", "tinybert_token_type", "tinybert_words_mask", "tinybert_dict_type"};
        String[] output_names = {"slot_softmax"};
        int[] n = {1, 1, 1, 1, 1};
        int[] c_max = {32, 32, 32, 511, 511};
        int[] h = {1, 1, 1, 32, 1};
        int[] w = {1, 1, 1, 1, 1};
        DataType[] dts = {DataType.UINT32, DataType.UINT32, DataType.UINT32, dt, DataType.UINT32};
        DataFormat[] dfs = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.MTK, DataFormat.NORMAL};
        BoltModel bolt_model = new BoltModel(modelPath, affinity, device,
            num_input, input_names, n, c_max, h, w, dts, dfs,
            num_output, output_names);

        int[] c_act = {27, 27, 27, 1, 1};
        float[][] inputData = {
            {101, 3017, 5164, 678, 5341, 5686, 5688, 4680, 5564, 6577, 1920, 1104, 2773, 5018, 671, 2108,
                2001, 3813, 3924, 2193, 4028, 3330, 3247, 712, 2898, 4638, 102},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
            {5}};
        float[][] resultData = {{0.796903967857f, 0.203096017241f}};

        double startTime = getMillisTime();
        BoltResult bolt_result = bolt_model.Run(num_input, input_names,
            n, c_act, h, w, dts, dfs,
            inputData);
        double endTime = getMillisTime();
        System.out.println(outputPrefix + bolt_model.DeviceMapping(device) + ", " + bolt_model.AffinityMapping(affinity)
            + ", tinybert " + String.format("%.3f", endTime - startTime)
            + " ms/sequence, model " + modelPath);
        float[][] result = bolt_result.getResultData();
        int[][] dimension = bolt_result.getResultDimension();
        int length = verify(resultData, result, dimension, 0.1f);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in tinybert, in Java API test");
            System.exit(1);
        }

        // model destroy
        bolt_model.Destructor();
    }

    public static void nmt(String outputPrefix, DeviceType device, AffinityType affinity, String modelPath) {
        int num_input = 2;
        int num_output = 1;
        String[] input_names = {"nmt_words", "nmt_positions"};
        String[] output_names = {"decoder_output"};
        int[] n = {1, 1};
        int[] c_max = {128, 128};
        int[] h = {1, 1, 1};
        int[] w = {1, 1, 1};
        DataType[] dts = {DataType.UINT32, DataType.UINT32, DataType.UINT32};
        DataFormat[] dfs = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel bolt_model = new BoltModel(modelPath, affinity, device,
            num_input, input_names, n, c_max, h, w, dts, dfs,
            num_output, output_names);

        int[] c_act = {28, 28};
        float[][] inputData = {
            {1977, 1788, 2061, 3911, 248, 734, 1330, 1111, 1307, 729, 411, 383, 101, 713,
                5640, 627, 1330, 37, 282, 352, 438, 94, 1111, 729, 1103, 72, 133, 2},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}};
        float[][] resultData = {
            {7456, 40, 1788, 2061, 3911, 248, 734, 140, 4667, 1307, 5365, 411, 383, 1244,
             206, 2669, 5640, 627, 50, 236, 37, 63, 48, 352, 94, 4667, 53, 287, 1763, 72,
             133, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

        double startTime = getMillisTime();
        BoltResult bolt_result = bolt_model.Run(num_input, input_names,
            n, c_act, h, w, dts, dfs,
            inputData);
        double endTime = getMillisTime();
        System.out.println(outputPrefix + bolt_model.DeviceMapping(device) + ", " + bolt_model.AffinityMapping(affinity)
            + ", machine translation " + String.format("%.3f", endTime - startTime)
            + " ms/sequence, model " + modelPath);
        int length = verify(resultData, bolt_result.getResultData(), bolt_result.getResultDimension(), 0);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in machine translation, in Java API test");
            System.exit(1);
        }

        // model destroy
        bolt_model.Destructor();
    }

    public static void nmt_tsc(String outputPrefix, DeviceType device, AffinityType affinity, DataType dataType,
        String encoderModelPath, String decoderModelPath)
    {
        int encoderInputNum = 2;
        String[] encoderInputNames = {"encoder_words", "encoder_positions"};
        int[] encoderNs = {1, 1};
        int[] encoderCMaxs = {128, 128};
        int[] encoderHs = {1, 1, 1};
        int[] encoderWs = {1, 1, 1};
        DataType[] encoderDataTypes = {DataType.UINT32, DataType.UINT32, DataType.UINT32};
        DataFormat[] encoderDataFormats = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel encoderModel = new BoltModel(encoderModelPath, affinity, device,
            encoderInputNum, encoderInputNames, encoderNs, encoderCMaxs, encoderHs, encoderWs, encoderDataTypes, encoderDataFormats);

        int[] encoderCActs = {4, 4};
        float[][] encoderInputData = {{13024, 1657, 35399, 0}, {0, 1, 2, 3}};
        int[] result = {6160, 3057, 113, 157, 0};

        double startTime = getMillisTime();
        BoltResult encoderResult = encoderModel.Run(encoderInputNum, encoderInputNames,
            encoderNs, encoderCActs, encoderHs, encoderWs, encoderDataTypes, encoderDataFormats,
            encoderInputData);
        double endTime = getMillisTime();
        double encoderTime = endTime - startTime;

        int decoderInputNum = 26;
        int decoderOutputNum = 13;
        int maxDecodeLength = 128;
        String[] decoderInputNames = {"decoder_words", "decoder_positions",
                "decoder_layer0_multihead_k", "decoder_layer0_multihead_v",
                "decoder_layer1_multihead_k", "decoder_layer1_multihead_v",
                "decoder_layer2_multihead_k", "decoder_layer2_multihead_v",
                "decoder_layer3_multihead_k", "decoder_layer3_multihead_v",
                "decoder_layer4_multihead_k", "decoder_layer4_multihead_v",
                "decoder_layer5_multihead_k", "decoder_layer5_multihead_v",
                "decoder_layer0_kmem", "decoder_layer0_vmem",
                "decoder_layer1_kmem", "decoder_layer1_vmem",
                "decoder_layer2_kmem", "decoder_layer2_vmem",
                "decoder_layer3_kmem", "decoder_layer3_vmem",
                "decoder_layer4_kmem", "decoder_layer4_vmem",
                "decoder_layer5_kmem", "decoder_layer5_vmem"
            };
        String[] decoderOutputNames = {
                "transformer_decoder_embedding_argmax",
                "transformer_decoder_layer_0_self_attention_multihead_k_cache", "transformer_decoder_layer_0_self_attention_multihead_v_cache",
                "transformer_decoder_layer_1_self_attention_multihead_k_cache", "transformer_decoder_layer_1_self_attention_multihead_v_cache",
                "transformer_decoder_layer_2_self_attention_multihead_k_cache", "transformer_decoder_layer_2_self_attention_multihead_v_cache",
                "transformer_decoder_layer_3_self_attention_multihead_k_cache", "transformer_decoder_layer_3_self_attention_multihead_v_cache",
                "transformer_decoder_layer_4_self_attention_multihead_k_cache", "transformer_decoder_layer_4_self_attention_multihead_v_cache",
                "transformer_decoder_layer_5_self_attention_multihead_k_cache", "transformer_decoder_layer_5_self_attention_multihead_v_cache",
            };
        int[] decoderNs = new int[decoderInputNum];
        int[] decoderCMaxs = new int[decoderInputNum];
        int[] decoderHs = new int[decoderInputNum];
        int[] decoderWs = new int[decoderInputNum];
        DataType[] decoderDataTypes = new DataType[decoderInputNum];
        DataFormat[] decoderDataFormats = new DataFormat[decoderInputNum];
        double decoderTime = 0;
        for (int i = 0; i < 2; i++) {
            decoderNs[i] = 1;
            decoderCMaxs[i] = 1;
            decoderHs[i] = 1;
            decoderWs[i] = 1;
            decoderDataTypes[i] = DataType.UINT32;
            decoderDataFormats[i] = DataFormat.NORMAL;
        }
        for (int i = 2; i < decoderInputNum; i++) {
            decoderNs[i] = 1;
            if (i - 2 < 12)
                decoderCMaxs[i] = 4;
            else
                decoderCMaxs[i] = maxDecodeLength - 1;
            decoderHs[i] = 512;
            decoderWs[i] = 1;
            decoderDataTypes[i] = dataType;
            decoderDataFormats[i] = DataFormat.MTK;
        }
        BoltModel decoderModel = new BoltModel(decoderModelPath, affinity, device,
            decoderInputNum, decoderInputNames, decoderNs, decoderCMaxs, decoderHs, decoderWs,
            decoderDataTypes, decoderDataFormats, decoderOutputNum, decoderOutputNames);
        float[][] encoderResultData = encoderResult.getResultData();
        float[][] decoderStates = {{}, {}, {}, {}, {}, {}, {},
                          {}, {}, {}, {}, {}, {}, {}};
        int word = 0, i;
        int[] words = new int[maxDecodeLength];
        for (i = 0; i < maxDecodeLength; i++) {
            int[] decoderCActs = {1, 1,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    i, i, i, i, i, i, i, i, i, i, i, i
                };
            float[][] decoderInputData = {{word}, {i},
                 encoderResultData[0], encoderResultData[1],
                 encoderResultData[2], encoderResultData[3],
                 encoderResultData[4], encoderResultData[5],
                 encoderResultData[6], encoderResultData[7],
                 encoderResultData[8], encoderResultData[9],
                 encoderResultData[10], encoderResultData[11],
                 decoderStates[0], decoderStates[1],
                 decoderStates[2], decoderStates[3],
                 decoderStates[4], decoderStates[5],
                 decoderStates[6], decoderStates[7],
                 decoderStates[8], decoderStates[9],
                 decoderStates[10], decoderStates[11],
            };
            startTime = getMillisTime();
            BoltResult decoderResult = decoderModel.Run(decoderInputNum, decoderInputNames,
                decoderNs, decoderCActs, decoderHs, decoderWs, decoderDataTypes, decoderDataFormats,
                decoderInputData);
            endTime = getMillisTime();
            decoderTime += endTime - startTime;
            float[][] decoderResultData = decoderResult.getResultData();
            for (int j = 0; j < 12; j++)
                decoderStates[j] = decoderResultData[j+1];
            word = (int)decoderResultData[0][0];
            words[i] = word;
            if (word == 0)
                break;
        }
        System.out.println(outputPrefix + encoderModel.DeviceMapping(device) + ", " + encoderModel.AffinityMapping(affinity)
            + ", machine translation " + String.format("%.3f", encoderTime+decoderTime)
            + " ms/sequence, encoder model " + encoderModelPath
            + ", decoder model " + decoderModelPath);
        verify(result, words, result.length);

        // model destroy
        encoderModel.Destructor();
        decoderModel.Destructor();
    }

    public static void tts(String outputPrefix, DeviceType device, AffinityType affinity,
        String encoderDecoderModelPath, String postnetModelPath, String melganModelPath, DataType dataType) {
        int numMels = 80;
        int maxResult = 2000 * 3;
        int encoderDecoderInputNum = 2;
        int encoderDecoderOutputNum = 2;
        String[] encoderDecoderInputNames = {"tts_words", "tts_alignments"};
        String[] encoderDecoderOutputNames = {"decoder_position", "decoder_result"};
        int[] encoderDecoderNs = {1, 1};
        int[] encoderDecoderCMaxs = {128, 128};
        int[] encoderDecoderHs = {1, 1};
        int[] encoderDecoderWs = {1, 1};
        DataType[] encoderDecoderDataTypes = {DataType.UINT32, dataType};
        DataFormat[] encoderDecoderDataFormats = {DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel encoderDecoderModel = new BoltModel(encoderDecoderModelPath, affinity, device,
            encoderDecoderInputNum, encoderDecoderInputNames,
            encoderDecoderNs, encoderDecoderCMaxs, encoderDecoderHs, encoderDecoderWs,
            encoderDecoderDataTypes, encoderDecoderDataFormats,
            encoderDecoderOutputNum, encoderDecoderOutputNames);
        int[] encoderDecoderCActs = {50, 50};
        float[][] encoderDecoderInputData = {{4, 25, 14, 33, 11, 20, 1, 9, 14, 33,
                27, 2, 20, 35, 15, 1, 10, 37, 11, 2,
                30, 34, 15, 7, 21, 1, 25, 14, 35, 21,
                27, 3, 25, 14, 34, 27, 1, 25, 14, 35,
                27, 1, 17, 36, 7, 20, 1, 37, 7, 0},
               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

        int postnetInputNum = 1;
        int postnetOutputNum = 1;
        String[] postnetInputNames = {"tts_decoder"};
        String[] postnetOutputNames = {"mel"};
        int[] postnetNs = {1};
        int[] postnetCMaxs = {maxResult};
        int[] postnetHs = {numMels};
        int[] postnetWs = {1};
        DataType[] postnetDataTypes = {dataType};
        DataFormat[] postnetDataFormats = {DataFormat.MTK};
        BoltModel postnetModel = new BoltModel(postnetModelPath, affinity, device,
            postnetInputNum, postnetInputNames,
            postnetNs, postnetCMaxs, postnetHs, postnetWs,
            postnetDataTypes, postnetDataFormats,
            postnetOutputNum, postnetOutputNames);

        int melganInputNum = 1;
        int melganOutputNum = 1;
        String[] melganInputNames = {"input"};
        String[] melganOutputNames = {"output"};
        int[] melganNs = {1};
        int[] melganCs = {numMels};
        int[] melganHMaxs = {maxResult};
        int[] melganWs = {1};
        DataType[] melganDataTypes = {dataType};
        DataFormat[] melganDataFormats = {DataFormat.NCHW};
        BoltModel melganModel = new BoltModel(melganModelPath, affinity, device,
            melganInputNum, melganInputNames,
            melganNs, melganCs, melganHMaxs, melganWs,
            melganDataTypes, melganDataFormats,
            melganOutputNum, melganOutputNames);

        double startTime = getMillisTime();
        BoltResult encoderDecoderResult = encoderDecoderModel.Run(encoderDecoderInputNum, encoderDecoderInputNames,
            encoderDecoderNs, encoderDecoderCActs, encoderDecoderHs, encoderDecoderWs,
            encoderDecoderDataTypes, encoderDecoderDataFormats,
            encoderDecoderInputData);
        float[][] encoderDecoderResultData = encoderDecoderResult.getResultData();

        int frameNum = ((int)encoderDecoderResultData[0][0] + 1) * 3;
        int[] postnetCActs = {frameNum};
        float[][] postnetInputData = {encoderDecoderResultData[1]};
        BoltResult postnetResult = postnetModel.Run(postnetInputNum, postnetInputNames,
            postnetNs, postnetCActs, postnetHs, postnetWs,
            postnetDataTypes, postnetDataFormats,
            postnetInputData);
        int[][] postnetResultDimension = postnetResult.getResultDimension();
        float[][] postnetResultData = postnetResult.getResultData();

        if (postnetResultDimension[0][0] != 1 || postnetResultDimension[0][1] != numMels
            || postnetResultDimension[0][2] != frameNum) {
            System.out.println("[ERROR] unmatched dimension of postnet");
            System.exit(1);
        }
        int[] melganHActs = {frameNum};
        float[][] melganInputData = {postnetResultData[0]};
        BoltResult melganResult = melganModel.Run(melganInputNum, melganInputNames,
            melganNs, melganCs, melganHActs, melganWs,
            melganDataTypes, melganDataFormats,
            melganInputData);
        int[][] melganResultDimension = melganResult.getResultDimension();
        float[][] melganResultData = melganResult.getResultData();
        int length = (int)melganResultDimension[0][2];
        float[] resultSum = {180.83719f};
        float[] result = new float[length];
        float[] sum = {0};
        for (int i = 0; i < length; i++) {
            result[i] = melganResultData[0][i*8];
            sum[0] += result[i];
        }
        double endTime = getMillisTime();
        System.out.println(outputPrefix + encoderDecoderModel.DeviceMapping(device) + ", " + encoderDecoderModel.AffinityMapping(affinity)
            + ", text to speech " + String.format("%.3f", endTime - startTime)
            + " ms/sequence, encoder decoder model " + encoderDecoderModelPath
            + ", postnet model " + postnetModelPath
            + ", melgan vocoder model " + melganModelPath);
        verify(sum, resultSum, 1, 8);

        // model destroy
        encoderDecoderModel.Destructor();
        postnetModel.Destructor();
        melganModel.Destructor();
    }

    public static void asr(String outputPrefix, DeviceType device, AffinityType affinity, String modelPath, DataType dataType) {
        int num_input = 1;
        int num_output = 1;
        String[] input_names = {"sounds"};
        String[] output_names = {"labels"};
        int[] n = {1};
        int[] c_max = {128};
        int[] h = {240};
        int[] w = {1};
        DataType[] dts = {dataType};
        DataFormat[] dfs = {DataFormat.NCHW};
        BoltModel bolt_model = new BoltModel(modelPath, affinity, device,
            num_input, input_names, n, c_max, h, w, dts, dfs,
            num_output, output_names);

        String soundDataPath  = "/data/local/tmp/CI/testing_data/nlp/asr/asr_rnnt/input/1.seq";
        String resultDataPath = "/data/local/tmp/CI/testing_data/nlp/asr/asr_rnnt/result/1.seq";
        float[] sound = readSequenceDataFromFile(soundDataPath, 0);
        float[] result = readSequenceDataFromFile(resultDataPath, 0);
        int[] c_act = {sound.length / h[0]};
        float[][] inputData = {sound};
        float[][] resultData = {result};

        double startTime = getMillisTime();
        BoltResult bolt_result = bolt_model.Run(num_input, input_names,
            n, c_act, h, w, dts, dfs,
            inputData);
        double endTime = getMillisTime();
        System.out.println(outputPrefix + bolt_model.DeviceMapping(device) + ", " + bolt_model.AffinityMapping(affinity)
            + ", speech recognization " + String.format("%.3f", endTime - startTime)
            + " ms/sequence, model " + modelPath);
        int length = verify(resultData, bolt_result.getResultData(), bolt_result.getResultDimension(), 0);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in speech recognize, in Java API test");
            System.exit(1);
        }

        // model destroy
        bolt_model.Destructor();
    }

    public static void classification(String outputPrefix, DeviceType device, AffinityType affinity, String modelPath,
        String inputName, DataType dataType, int[] imageSize, float initValue, int topIndex)
    {
        int num_input = 1;
        String[] input_names = {inputName};
        int[] n = {1};
        int[] c = {imageSize[0]};
        int[] h = {imageSize[1]};
        int[] w = {imageSize[2]};
        DataType[] dts = {dataType};
        DataFormat[] dfs = {DataFormat.NCHW};
        // constructor(modelCreate + ready)
        BoltModel bolt_model = new BoltModel(modelPath, affinity, device,
            num_input, input_names, n, c, h, w, dts, dfs);

        int length = imageSize[0]*imageSize[1]*imageSize[2];
        float[][] inputData = new float[1][length];
        for (int i = 0; i < length; i++) {
            inputData[0][i] = initValue;
        }
        // model run
        double startTime = getMillisTime();
        BoltResult bolt_result = bolt_model.Run(num_input, input_names, inputData);
        double endTime = getMillisTime();
        System.out.println(outputPrefix + bolt_model.DeviceMapping(device) + ", " + bolt_model.AffinityMapping(affinity)
            + ", classification " + String.format("%.3f", endTime - startTime)
            + " ms/image, model " + modelPath);
        
        float[][] result = bolt_result.getResultData();
        int labelIndex = top1(result[0], 0, result[0].length);
        if (labelIndex != topIndex) {
            System.err.println("[ERROR] verify data classfication label failed " + labelIndex
                + " " + topIndex + ", in Java API test");
            System.exit(1);
        }

        // model destroy
        bolt_model.Destructor();
    }

    public static void testSuites(String outputPrefix, DeviceType device, AffinityType affinity) {
        String prefix = "/data/local/tmp/CI/java/tmp";

        int[] image_3x224x224 = {3, 224, 224};
        int[] image_2x188x188 = {2, 188, 188};
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/mobilenet_v1/mobilenet_v1_f16.bolt",
            "data", DataType.FP16, image_3x224x224, 1, 499);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/mobilenet_v2/mobilenet_v2_f16.bolt",
            "data", DataType.FP16, image_3x224x224, 1, 813);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/mobilenet_v3/mobilenet_v3_f16.bolt",
            "data", DataType.FP16, image_3x224x224, 1, 892);
        if (device == DeviceType.GPU)
            return;
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/resnet50/resnet50_f16.bolt",
            "data", DataType.FP16, image_3x224x224, 255, 506);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/squeezenet/squeezenet_f16.bolt",
             "data", DataType.FP16, image_3x224x224, 255, 310);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/mobilenet_v1/mobilenet_v1_f32.bolt",
            "data", DataType.FP32, image_3x224x224, 1, 499);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/mobilenet_v2/mobilenet_v2_f32.bolt",
            "data", DataType.FP32, image_3x224x224, 1, 813);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/mobilenet_v3/mobilenet_v3_f32.bolt",
            "data", DataType.FP32, image_3x224x224, 1, 892);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/resnet50/resnet50_f32.bolt",
            "data", DataType.FP32, image_3x224x224, 255, 506);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/squeezenet/squeezenet_f32.bolt",
             "data", DataType.FP32, image_3x224x224, 255, 310);
        classification(outputPrefix, device, affinity, prefix+"/onnx_models/ghostnet/ghostnet_f16.bolt",
            "MobileNetV2/MobileNetV2/Conv2d_0/Conv2D__6:0", DataType.FP16, image_3x224x224, 255, 789);
        classification(outputPrefix, device, affinity, prefix+"/onnx_models/ghostnet/ghostnet_f32.bolt",
            "MobileNetV2/MobileNetV2/Conv2d_0/Conv2D__6:0", DataType.FP32, image_3x224x224, 255, 789);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/fingerprint_resnet18/fingerprint_resnet18_f16.bolt",
             "Data", DataType.FP16, image_2x188x188, 1, 0);
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/fingerprint_resnet18/fingerprint_resnet18_f32.bolt",
             "Data", DataType.FP32, image_2x188x188, 1, 0);

        tinybert_intent_slot(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert384/tinybert384_int8_q.bolt");
        tinybert_intent_slot(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert384/tinybert384_f16.bolt");
        tinybert_intent_slot(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert384/tinybert384_f32.bolt");

        tinybert_intent_slot(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert/tinybert_f16.bolt");
        tinybert_intent_slot(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert/tinybert_f32.bolt");
        tinybert_disambiguate(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert_disambiguate/tinybert_disambiguate_f16.bolt", DataType.FP16);
        tinybert_disambiguate(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert_disambiguate/tinybert_disambiguate_f32.bolt", DataType.FP32);
        nmt(outputPrefix, device, affinity, prefix+"/caffe_models/nmt/nmt_f16.bolt");
        nmt(outputPrefix, device, affinity, prefix+"/caffe_models/nmt/nmt_f32.bolt");
        nmt_tsc(outputPrefix, device, affinity, DataType.FP16, prefix+"/caffe_models/nmt_tsc_encoder/nmt_tsc_encoder_f16.bolt",
            prefix+"/caffe_models/nmt_tsc_decoder/nmt_tsc_decoder_f16.bolt");
        nmt_tsc(outputPrefix, device, affinity, DataType.FP32, prefix+"/caffe_models/nmt_tsc_encoder/nmt_tsc_encoder_f32.bolt",
            prefix+"/caffe_models/nmt_tsc_decoder/nmt_tsc_decoder_f32.bolt");
        
        classification(outputPrefix, device, affinity, prefix+"/caffe_models/squeezenet/squeezenet_int8_q.bolt",
             "data", DataType.FP16, image_3x224x224, 255, 310);
        classification(outputPrefix, device, affinity, prefix+"/onnx_models/birealnet18/birealnet18_f16.bolt",
            "0", DataType.FP16, image_3x224x224, 255, 565);

        asr(outputPrefix, device, affinity, prefix+"/caffe_models/asr_rnnt/asr_rnnt_f16.bolt", DataType.FP16);
        asr(outputPrefix, device, affinity, prefix+"/caffe_models/asr_rnnt/asr_rnnt_f32.bolt", DataType.FP32);
        tts(outputPrefix, device, affinity,
            prefix+"/caffe_models/tts_encoder_decoder/tts_encoder_decoder_f16.bolt",
            prefix+"/caffe_models/tts_postnet/tts_postnet_f16.bolt",
            prefix+"/onnx_models/tts_melgan_vocoder/tts_melgan_vocoder_f16.bolt",
            DataType.FP16);
        tts(outputPrefix, device, affinity,
            prefix+"/caffe_models/tts_encoder_decoder/tts_encoder_decoder_f32.bolt",
            prefix+"/caffe_models/tts_postnet/tts_postnet_f32.bolt",
            prefix+"/onnx_models/tts_melgan_vocoder/tts_melgan_vocoder_f32.bolt",
            DataType.FP32);
    }

    public static void main(String[] args) {
        String outputPrefix = "[INFO] ";
        if (args.length > 0) {
            outputPrefix += args[0] + ", ";
        }
        testSuites(outputPrefix, DeviceType.CPU, AffinityType.HIGH_PERFORMANCE);
        testSuites(outputPrefix, DeviceType.CPU, AffinityType.LOW_POWER);
        testSuites(outputPrefix, DeviceType.GPU, AffinityType.LOW_POWER);
    }
}
