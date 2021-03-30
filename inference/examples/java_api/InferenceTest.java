// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import com.huawei.noah.*;

public final class InferenceTest {
    public static void GSR(
        String rootPath, String outputPrefix, AffinityType affinity, String modelPath)
    {
        int inputNum = 4;
        int outputNum = 1;
        String[] inputName = {"words", "positions", "token_type", "mask"};
        String[] outputName = {"normalized_token_embeddings_name"};
        int[] inputN = {1, 1, 1, 1};
        int[] inputC = {128, 128, 128, 128};
        int[] inputH = {1, 1, 1, 1};
        int[] inputW = {1, 1, 1, 1};
        DataType[] inputDataType = {
            DataType.UINT32, DataType.UINT32, DataType.UINT32, DataType.FP32};
        DataFormat[] inputDataFormat = {
            DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel boltModel = new BoltModel(modelPath, affinity, inputNum, inputName, inputN,
            inputC, inputH, inputW, inputDataType, inputDataFormat, outputNum, outputName);
        String wordsPath = rootPath + "/testing_data/nlp/gsr/64.seq";
        float[] words = TestUtils.readSequenceDataFromFile(wordsPath, 0);
        float[] positions = new float[words.length];
        float[] token_type = new float[words.length];
        float[] mask = new float[words.length];
        for (int j = 0; j < words.length; j++) {
            positions[j] = (float)j;
            token_type[j] = (float)0;
            mask[j] = (float)1;
        }
        for (int j = 0; j < inputW.length; j++) {
            inputC[j] = words.length;
        }
        float[][] inputData = {words, positions, token_type, mask};
        double startTime = TestUtils.getMillisTime();
        BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputC, inputH, inputW,
            inputDataType, inputDataFormat, inputData);
        if (null == boltResult) {
            System.err.println("[ERROR] modelAddr is 0 in Java API test");
            boltModel.destructor();
            System.exit(1);
        }
        double endTime = TestUtils.getMillisTime();
        System.out.println(outputPrefix + affinity + ", GSR " +
            String.format("%.3f", endTime - startTime) + " ms/sequence, model " + modelPath);
        float[][] result = boltResult.getResultData();
        int[][] dimension = boltResult.getResultDimension();
        String resultPath = rootPath + "/testing_data/nlp/gsr/64_out.seq";
        float[][] resultData = {TestUtils.readSequenceDataFromFile(resultPath, 0)};
        int length = TestUtils.verify(resultData, result, dimension, 0.1f);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in GSR, in Java API test");
            System.exit(1);
        }

        // model destroy
        boltModel.destructor();
    }

    public static void tinybertIntentSlot(
        String outputPrefix, AffinityType affinity, String modelPath)
    {
        int inputNum = 3;
        int outputNum = 2;
        String[] inputName = {"tinybert_words", "tinybert_positions", "tinybert_token_type"};
        String[] outputName = {"intent_softmax", "slot_softmax"};
        int[] inputN = {1, 1, 1};
        int[] inputCMax = {32, 32, 32};
        int[] inputH = {1, 1, 1};
        int[] inputW = {1, 1, 1};
        DataType[] intputDataType = {DataType.UINT32, DataType.UINT32, DataType.UINT32};
        DataFormat[] intputDataFormat = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel boltModel = new BoltModel(modelPath, affinity, inputNum, inputName, inputN,
            inputCMax, inputH, inputW, intputDataType, intputDataFormat, outputNum, outputName);

        int[] inputCActual = {9, 9, 9};
        float[][] inputData = {{101, 2224, 8224, 7341, 2000, 22149, 2000, 2899, 102},
            {0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 0, 0, 0, 0, 0, 0, 0, 0}};
        float[][] resultData = {{22, 0.999023f}, {44, 44, 1, 23, 44, 44, 44, 8, 44}};

        double startTime = TestUtils.getMillisTime();
        BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputCActual, inputH,
            inputW, intputDataType, intputDataFormat, inputData);
        if (null == boltResult) {
            System.err.println("[ERROR] modelAddr is 0 in Java API test");
            boltModel.destructor();
            System.exit(1);
        }
        double endTime = TestUtils.getMillisTime();
        System.out.println(outputPrefix + affinity + ", tinybert " +
            String.format("%.3f", endTime - startTime) + " ms/sequence, model " + modelPath);
        float[][] result = boltResult.getResultData();
        int[][] dimension = boltResult.getResultDimension();
        int intentIndex = TestUtils.top1(result[0], 0, result[0].length);
        float[][] finalResult = new float[2][dimension[1][1]];
        finalResult[0][0] = intentIndex;
        finalResult[0][1] = result[0][intentIndex];
        for (int i = 0; i < dimension[1][1]; i++) {
            finalResult[1][i] = TestUtils.top1(result[1], i * dimension[1][2], dimension[1][2]) -
                i * dimension[1][2];
        }
        int[][] finalDimension = {{1, 2}, {1, dimension[1][1]}};
        int length = TestUtils.verify(resultData, finalResult, finalDimension, 0.1f);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in tinybert, in Java API test");
            System.exit(1);
        }

        // model destroy
        boltModel.destructor();
    }

    public static void tinybertDisambiguate(
        String outputPrefix, AffinityType affinity, String modelPath, DataType dt)
    {
        int inputNum = 5;
        int outputNum = 1;
        String[] inputName = {"tinybert_words", "tinybert_positions", "tinybert_token_type",
            "tinybert_words_mask", "tinybert_dict_type"};
        String[] outputName = {"slot_softmax"};
        int[] inputN = {1, 1, 1, 1, 1};
        int[] inputCMax = {32, 32, 32, 511, 511};
        int[] inputH = {1, 1, 1, 32, 1};
        int[] inputW = {1, 1, 1, 1, 1};
        DataType[] intputDataType = {
            DataType.UINT32, DataType.UINT32, DataType.UINT32, dt, DataType.UINT32};
        DataFormat[] intputDataFormat = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL,
            DataFormat.MTK, DataFormat.NORMAL};
        BoltModel boltModel = new BoltModel(modelPath, affinity, inputNum, inputName, inputN,
            inputCMax, inputH, inputW, intputDataType, intputDataFormat, outputNum, outputName);

        int[] inputCActual = {27, 27, 27, 1, 1};
        float[][] inputData = {
            {101, 3017, 5164, 678, 5341, 5686, 5688, 4680, 5564, 6577, 1920, 1104, 2773, 5018, 671,
                2108, 2001, 3813, 3924, 2193, 4028, 3330, 3247, 712, 2898, 4638, 102},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0}, {5}};
        float[][] resultData = {{0.796903967857f, 0.203096017241f}};

        double startTime = TestUtils.getMillisTime();
        BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputCActual, inputH,
            inputW, intputDataType, intputDataFormat, inputData);
        if (null == boltResult) {
            System.err.println("[ERROR] modelAddr is 0 in Java API test");
            boltModel.destructor();
            System.exit(1);
        }
        double endTime = TestUtils.getMillisTime();
        System.out.println(outputPrefix + affinity + ", tinybert " +
            String.format("%.3f", endTime - startTime) + " ms/sequence, model " + modelPath);
        float[][] result = boltResult.getResultData();
        int[][] dimension = boltResult.getResultDimension();
        int length = TestUtils.verify(resultData, result, dimension, 0.1f);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in tinybert, in Java API test");
            System.exit(1);
        }

        // model destroy
        boltModel.destructor();
    }

    public static void nmt(String outputPrefix, AffinityType affinity, String modelPath)
    {
        int inputNum = 2;
        int outputNum = 1;
        String[] inputName = {"nmt_words", "nmt_positions"};
        String[] outputName = {"decoder_output"};
        int[] inputN = {1, 1};
        int[] inputCMax = {128, 128};
        int[] inputH = {1, 1};
        int[] inputW = {1, 1};
        DataType[] intputDataType = {DataType.UINT32, DataType.UINT32, DataType.UINT32};
        DataFormat[] intputDataFormat = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel boltModel = new BoltModel(modelPath, affinity, inputNum, inputName, inputN,
            inputCMax, inputH, inputW, intputDataType, intputDataFormat, outputNum, outputName);

        int[] inputCActual = {28, 28};
        float[][] inputData = {
            {1977, 1788, 2061, 3911, 248, 734, 1330, 1111, 1307, 729, 411, 383, 101, 713, 5640, 627,
                1330, 37, 282, 352, 438, 94, 1111, 729, 1103, 72, 133, 2},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27}};
        float[][] resultData = {{7456, 40, 1788, 2061, 3911, 248, 734, 140, 4667, 1307, 5365, 411,
            383, 1244, 206, 2669, 5640, 627, 50, 236, 37, 63, 48, 352, 94, 4667, 53, 287, 1763, 72,
            133, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

        double startTime = TestUtils.getMillisTime();
        BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputCActual, inputH,
            inputW, intputDataType, intputDataFormat, inputData);
        if (null == boltResult) {
            System.err.println("[ERROR] modelAddr is 0 in Java API test");
            boltModel.destructor();
            System.exit(1);
        }
        double endTime = TestUtils.getMillisTime();
        System.out.println(outputPrefix + affinity + ", machine translation " +
            String.format("%.3f", endTime - startTime) + " ms/sequence, model " + modelPath);
        int length = TestUtils.verify(
            resultData, boltResult.getResultData(), boltResult.getResultDimension(), 0);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in machine translation, in Java API test");
            System.exit(1);
        }

        // model destroy
        boltModel.destructor();
    }

    public static void nmtTSC(String outputPrefix,
        AffinityType affinity,
        DataType dataType,
        String encoderModelPath,
        String decoderModelPath)
    {
        int encoderInputNum = 2;
        String[] encoderInputNames = {"encoder_words", "encoder_positions"};
        int[] encoderNs = {1, 1};
        int[] encoderCMaxs = {128, 128};
        int[] encoderHs = {1, 1};
        int[] encoderWs = {1, 1};
        DataType[] encoderDataTypes = {DataType.UINT32, DataType.UINT32, DataType.UINT32};
        DataFormat[] encoderDataFormats = {DataFormat.NORMAL, DataFormat.NORMAL, DataFormat.NORMAL};
        BoltModel encoderModel = new BoltModel(encoderModelPath, affinity, encoderInputNum,
            encoderInputNames, encoderNs, encoderCMaxs, encoderHs, encoderWs, encoderDataTypes,
            encoderDataFormats);

        int[] encoderCActs = {4, 4};
        float[][] encoderInputData = {{13024, 1657, 35399, 0}, {0, 1, 2, 3}};
        int[] result = {6160, 3057, 113, 157, 0};

        double startTime = TestUtils.getMillisTime();
        BoltResult encoderResult = encoderModel.run(encoderInputNum, encoderInputNames, encoderNs,
            encoderCActs, encoderHs, encoderWs, encoderDataTypes, encoderDataFormats,
            encoderInputData);
        if (null == encoderResult) {
            System.err.println("[ERROR] modelAddr is 0 in Java API test");
            encoderModel.destructor();
            System.exit(1);
        }
        double endTime = TestUtils.getMillisTime();
        double encoderTime = endTime - startTime;

        int decoderInputNum = 26;
        int decoderOutputNum = 13;
        int maxDecodeLength = 128;
        String[] decoderInputNames = {"decoder_words", "decoder_positions",
            "decoder_layer0_multihead_k", "decoder_layer0_multihead_v", "decoder_layer1_multihead_k",
            "decoder_layer1_multihead_v", "decoder_layer2_multihead_k", "decoder_layer2_multihead_v",
            "decoder_layer3_multihead_k", "decoder_layer3_multihead_v", "decoder_layer4_multihead_k",
            "decoder_layer4_multihead_v", "decoder_layer5_multihead_k", "decoder_layer5_multihead_v",
            "decoder_layer0_kmem", "decoder_layer0_vmem", "decoder_layer1_kmem",
            "decoder_layer1_vmem", "decoder_layer2_kmem", "decoder_layer2_vmem",
            "decoder_layer3_kmem", "decoder_layer3_vmem", "decoder_layer4_kmem",
            "decoder_layer4_vmem", "decoder_layer5_kmem", "decoder_layer5_vmem"};
        String[] decoderOutputNames = {
            "transformer_decoder_embedding_argmax",
            "transformer_decoder_layer_0_self_attention_multihead_k_cache",
            "transformer_decoder_layer_0_self_attention_multihead_v_cache",
            "transformer_decoder_layer_1_self_attention_multihead_k_cache",
            "transformer_decoder_layer_1_self_attention_multihead_v_cache",
            "transformer_decoder_layer_2_self_attention_multihead_k_cache",
            "transformer_decoder_layer_2_self_attention_multihead_v_cache",
            "transformer_decoder_layer_3_self_attention_multihead_k_cache",
            "transformer_decoder_layer_3_self_attention_multihead_v_cache",
            "transformer_decoder_layer_4_self_attention_multihead_k_cache",
            "transformer_decoder_layer_4_self_attention_multihead_v_cache",
            "transformer_decoder_layer_5_self_attention_multihead_k_cache",
            "transformer_decoder_layer_5_self_attention_multihead_v_cache",
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
            if (i - 2 < 12) {
                decoderCMaxs[i] = 4;
            } else {
                decoderCMaxs[i] = maxDecodeLength - 1;
            }
            decoderHs[i] = 512;
            decoderWs[i] = 1;
            decoderDataTypes[i] = dataType;
            decoderDataFormats[i] = DataFormat.MTK;
        }
        BoltModel decoderModel = new BoltModel(decoderModelPath, affinity, decoderInputNum,
            decoderInputNames, decoderNs, decoderCMaxs, decoderHs, decoderWs, decoderDataTypes,
            decoderDataFormats, decoderOutputNum, decoderOutputNames);
        float[][] encoderResultData = encoderResult.getResultData();
        float[][] decoderStates = {{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
        int word = 0, i;
        int[] words = new int[maxDecodeLength];
        for (i = 0; i < maxDecodeLength; i++) {
            int[] decoderCActs = {
                1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, i, i, i, i, i, i, i, i, i, i, i, i};
            float[][] decoderInputData = {
                {word},
                {i},
                encoderResultData[0],
                encoderResultData[1],
                encoderResultData[2],
                encoderResultData[3],
                encoderResultData[4],
                encoderResultData[5],
                encoderResultData[6],
                encoderResultData[7],
                encoderResultData[8],
                encoderResultData[9],
                encoderResultData[10],
                encoderResultData[11],
                decoderStates[0],
                decoderStates[1],
                decoderStates[2],
                decoderStates[3],
                decoderStates[4],
                decoderStates[5],
                decoderStates[6],
                decoderStates[7],
                decoderStates[8],
                decoderStates[9],
                decoderStates[10],
                decoderStates[11],
            };
            startTime = TestUtils.getMillisTime();
            BoltResult decoderResult = decoderModel.run(decoderInputNum, decoderInputNames,
                decoderNs, decoderCActs, decoderHs, decoderWs, decoderDataTypes, decoderDataFormats,
                decoderInputData);
            if (null == decoderResult) {
                System.err.println("[ERROR] modelAddr is 0 in Java API test");
                decoderModel.destructor();
                encoderModel.destructor();
                System.exit(1);
            }
            endTime = TestUtils.getMillisTime();
            decoderTime += endTime - startTime;
            float[][] decoderResultData = decoderResult.getResultData();
            for (int j = 0; j < 12; j++) {
                decoderStates[j] = decoderResultData[j + 1];
            }
            word = (int)decoderResultData[0][0];
            words[i] = word;
            if (word == 0) {
                break;
            }
        }
        System.out.println(outputPrefix + affinity + ", machine translation " +
            String.format("%.3f", encoderTime + decoderTime) + " ms/sequence, encoder model " +
            encoderModelPath + ", decoder model " + decoderModelPath);
        TestUtils.verify(result, words, result.length);

        // model destroy
        encoderModel.destructor();
        decoderModel.destructor();
    }

    public static void tts(String outputPrefix,
        AffinityType affinity,
        String encoderDecoderModelPath,
        String postnetModelPath,
        String melganModelPath,
        DataType dataType)
    {
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
        BoltModel encoderDecoderModel = new BoltModel(encoderDecoderModelPath, affinity,
            encoderDecoderInputNum, encoderDecoderInputNames, encoderDecoderNs, encoderDecoderCMaxs,
            encoderDecoderHs, encoderDecoderWs, encoderDecoderDataTypes, encoderDecoderDataFormats,
            encoderDecoderOutputNum, encoderDecoderOutputNames);
        int[] encoderDecoderCActs = {50, 50};
        float[][] encoderDecoderInputData = {
            {4, 25, 14, 33, 11, 20, 1, 9, 14, 33, 27, 2, 20, 35, 15, 1, 10, 37, 11, 2, 30, 34, 15,
                7, 21, 1, 25, 14, 35, 21, 27, 3, 25, 14, 34, 27, 1, 25, 14, 35, 27, 1, 17, 36, 7,
                20, 1, 37, 7, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

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
        BoltModel postnetModel = new BoltModel(postnetModelPath, affinity, postnetInputNum,
            postnetInputNames, postnetNs, postnetCMaxs, postnetHs, postnetWs, postnetDataTypes,
            postnetDataFormats, postnetOutputNum, postnetOutputNames);

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
        BoltModel melganModel = new BoltModel(melganModelPath, affinity, melganInputNum,
            melganInputNames, melganNs, melganCs, melganHMaxs, melganWs, melganDataTypes,
            melganDataFormats, melganOutputNum, melganOutputNames);

        double startTime = TestUtils.getMillisTime();
        BoltResult encoderDecoderResult = encoderDecoderModel.run(encoderDecoderInputNum,
            encoderDecoderInputNames, encoderDecoderNs, encoderDecoderCActs, encoderDecoderHs,
            encoderDecoderWs, encoderDecoderDataTypes, encoderDecoderDataFormats,
            encoderDecoderInputData);
        float[][] encoderDecoderResultData = encoderDecoderResult.getResultData();

        int frameNum = ((int)encoderDecoderResultData[0][0] + 1) * 3;
        int[] postnetCActs = {frameNum};
        float[][] postnetInputData = {encoderDecoderResultData[1]};
        BoltResult postnetResult = postnetModel.run(postnetInputNum, postnetInputNames, postnetNs,
            postnetCActs, postnetHs, postnetWs, postnetDataTypes, postnetDataFormats,
            postnetInputData);
        int[][] postnetResultDimension = postnetResult.getResultDimension();
        float[][] postnetResultData = postnetResult.getResultData();

        if (postnetResultDimension[0][0] != 1 || postnetResultDimension[0][1] != numMels ||
            postnetResultDimension[0][2] != frameNum) {
            System.out.println("[ERROR] unmatched dimension of postnet");
            System.exit(1);
        }
        int[] melganHActs = {frameNum};
        float[][] melganInputData = {postnetResultData[0]};
        BoltResult melganResult = melganModel.run(melganInputNum, melganInputNames, melganNs,
            melganCs, melganHActs, melganWs, melganDataTypes, melganDataFormats, melganInputData);
        int[][] melganResultDimension = melganResult.getResultDimension();
        float[][] melganResultData = melganResult.getResultData();
        int length = (int)melganResultDimension[0][2];
        float[] resultSum = {180.595139f};
        if (DataType.FP16 == dataType) {
            resultSum[0] = 222.57361f;
        }
        float[] sum = {0};
        for (int i = 0; i < length; i++) {
            sum[0] += melganResultData[0][i];
        }
        double endTime = TestUtils.getMillisTime();
        System.out.println(outputPrefix + affinity + ", text to speech " +
            String.format("%.3f", endTime - startTime) + " ms/sequence, encoder decoder model " +
            encoderDecoderModelPath + ", postnet model " + postnetModelPath +
            ", melgan vocoder model " + melganModelPath);
        TestUtils.verify(sum, resultSum, 1, 1);

        // model destroy
        encoderDecoderModel.destructor();
        postnetModel.destructor();
        melganModel.destructor();
    }

    public static void asr(String rootPath,
        String outputPrefix,
        AffinityType affinity,
        String modelPath,
        DataType dataType)
    {
        int inputNum = 1;
        int outputNum = 1;
        String[] inputName = {"sounds"};
        String[] outputName = {"labels"};
        int[] inputN = {1};
        int[] inputCMax = {128};
        int[] inputH = {240};
        int[] inputW = {1};
        DataType[] intputDataType = {dataType};
        DataFormat[] intputDataFormat = {DataFormat.NCHW};
        BoltModel boltModel = new BoltModel(modelPath, affinity, inputNum, inputName, inputN,
            inputCMax, inputH, inputW, intputDataType, intputDataFormat, outputNum, outputName);

        String soundDataPath = rootPath + "/testing_data/nlp/asr/asr_rnnt/input/1.seq";
        String resultDataPath = rootPath + "/testing_data/nlp/asr/asr_rnnt/result/1.seq";
        float[] sound = TestUtils.readSequenceDataFromFile(soundDataPath, 0);
        float[] result = TestUtils.readSequenceDataFromFile(resultDataPath, 0);
        int[] inputCActual = {sound.length / inputH[0]};
        float[][] inputData = {sound};
        float[][] resultData = {result};

        double startTime = TestUtils.getMillisTime();
        BoltResult boltResult = boltModel.run(inputNum, inputName, inputN, inputCActual, inputH,
            inputW, intputDataType, intputDataFormat, inputData);
        if (null == boltResult) {
            System.err.println("[ERROR] modelAddr is 0 in Java API test");
            boltModel.destructor();
            System.exit(1);
        }
        double endTime = TestUtils.getMillisTime();
        System.out.println(outputPrefix + affinity + ", speech recognization " +
            String.format("%.3f", endTime - startTime) + " ms/sequence, model " + modelPath);
        int length = TestUtils.verify(
            resultData, boltResult.getResultData(), boltResult.getResultDimension(), 0);
        if (length == 0) {
            System.err.println("[ERROR] verify null data in speech recognize, in Java API test");
            System.exit(1);
        }

        // model destroy
        boltModel.destructor();
    }

    public static void classification(String outputPrefix,
        AffinityType affinity,
        String modelPath,
        String name,
        DataType dataType,
        int[] imageSize,
        float initValue,
        int topIndex)
    {
        int inputNum = 1;
        String[] inputName = {name};
        int[] inputN = {1};
        int[] inputC = {imageSize[0]};
        int[] inputH = {imageSize[1]};
        int[] inputW = {imageSize[2]};
        DataType[] intputDataType = {dataType};
        DataFormat[] intputDataFormat = {DataFormat.NCHW};
        // constructor(modelCreate + ready)
        BoltModel boltModel = new BoltModel(modelPath, affinity, inputNum, inputName, inputN,
            inputC, inputH, inputW, intputDataType, intputDataFormat);

        int length = imageSize[0] * imageSize[1] * imageSize[2];
        float[][] inputData = new float[1][length];
        for (int i = 0; i < length; i++) {
            inputData[0][i] = initValue;
        }
        // warm up
        boltModel.run(inputNum, inputName, inputData);

        // model run
        double startTime = TestUtils.getMillisTime();
        BoltResult boltResult = boltModel.run(inputNum, inputName, inputData);
        if (null == boltResult) {
            System.err.println("[ERROR] modelAddr is 0 in Java API test");
            boltModel.destructor();
            System.exit(1);
        }
        double endTime = TestUtils.getMillisTime();
        System.out.println(outputPrefix + affinity + ", classification " +
            String.format("%.3f", endTime - startTime) + " ms/image, model " + modelPath);

        float[][] result = boltResult.getResultData();
        int labelIndex = TestUtils.top1(result[0], 0, result[0].length);
        if (labelIndex != topIndex) {
            System.out.println("[ERROR] verify data classfication label failed " + labelIndex +
                " " + topIndex + ", in Java API test");
            System.exit(1);
        }

        // model destroy
        boltModel.destructor();
    }

    public static void testSuites0(
        String rootPath, String outputPrefix, AffinityType affinity, DataType dt)
    {
        String prefix = rootPath + "/model_zoo";
        String modelSuffix = "";
        if (dt == DataType.FP16)
            modelSuffix = "_f16.bolt";
        else if (dt == DataType.FP32)
            modelSuffix = "_f32.bolt";
        int[] image_3x224x224 = {3, 224, 224};
        int[] image_2x188x188 = {2, 188, 188};
        classification(outputPrefix, affinity,
            prefix + "/caffe_models/mobilenet_v1/mobilenet_v1" + modelSuffix, "data", dt,
            image_3x224x224, 1, 499);
        classification(outputPrefix, affinity,
            prefix + "/caffe_models/mobilenet_v2/mobilenet_v2" + modelSuffix, "data", dt,
            image_3x224x224, 1, 813);
        classification(outputPrefix, affinity,
            prefix + "/caffe_models/mobilenet_v3/mobilenet_v3" + modelSuffix, "data", dt,
            image_3x224x224, 1, 892);
        classification(outputPrefix, affinity,
            prefix + "/caffe_models/squeezenet/squeezenet" + modelSuffix, "data", dt,
            image_3x224x224, 255, 310);
        classification(outputPrefix, affinity,
            prefix + "/caffe_models/resnet50/resnet50" + modelSuffix, "data", dt, image_3x224x224,
            255, 506);
        classification(outputPrefix, affinity,
            prefix + "/caffe_models/fingerprint_resnet18/fingerprint_resnet18" + modelSuffix,
            "Data", dt, image_2x188x188, 1, 0);
        int ghostnetTopIndex = 623;
        if (dt == DataType.FP16) {
            ghostnetTopIndex = 512;
            if (affinity == AffinityType.GPU) {
                // TODO: verify this data, classification top1 accuracy 0.90
                ghostnetTopIndex = 549;
            }
        }
        classification(outputPrefix, affinity, prefix + "/onnx_models/ghostnet/ghostnet" + modelSuffix,
            "input:0", dt, image_3x224x224, 255, ghostnetTopIndex);
        tinybertIntentSlot(
            outputPrefix, affinity, prefix + "/caffe_models/tinybert384/tinybert384" + modelSuffix);
        if (affinity == AffinityType.GPU) {
            return;
        }
        tinybertIntentSlot(
            outputPrefix, affinity, prefix + "/caffe_models/tinybert/tinybert" + modelSuffix);
        tinybertDisambiguate(outputPrefix, affinity,
            prefix + "/caffe_models/tinybert_disambiguate/tinybert_disambiguate" + modelSuffix, dt);
        GSR(rootPath, outputPrefix, affinity, prefix + "/caffe_models/gsr/gsr" + modelSuffix);
        nmt(outputPrefix, affinity, prefix + "/caffe_models/nmt/nmt" + modelSuffix);
        nmtTSC(outputPrefix, affinity, dt,
            prefix + "/caffe_models/nmt_tsc_encoder/nmt_tsc_encoder" + modelSuffix,
            prefix + "/caffe_models/nmt_tsc_decoder/nmt_tsc_decoder" + modelSuffix);
        asr(rootPath, outputPrefix, affinity,
            prefix + "/caffe_models/asr_rnnt/asr_rnnt" + modelSuffix, dt);
        tts(outputPrefix, affinity,
            prefix + "/caffe_models/tts_encoder_decoder/tts_encoder_decoder" + modelSuffix,
            prefix + "/caffe_models/tts_postnet/tts_postnet" + modelSuffix,
            prefix + "/onnx_models/tts_melgan_vocoder/tts_melgan_vocoder" + modelSuffix, dt);
    }

    public static void testSuites1(String rootPath, String outputPrefix, AffinityType affinity)
    {
        String prefix = rootPath + "/model_zoo";
        int[] image_3x224x224 = {3, 224, 224};
        int[] image_2x188x188 = {2, 188, 188};
        classification(outputPrefix, affinity,
            prefix + "/onnx_models/birealnet18/birealnet18_f16.bolt", "0", DataType.FP16,
            image_3x224x224, 255, 565);
        classification(outputPrefix, affinity,
            prefix + "/caffe_models/squeezenet/squeezenet_int8_q.bolt", "data", DataType.FP16,
            image_3x224x224, 255, 310);
        tinybertIntentSlot(
            outputPrefix, affinity, prefix + "/caffe_models/tinybert384/tinybert384_int8_q.bolt");
    }

    public static void main(String[] args)
    {
        String outputPrefix = "[INFO] ";
        if (args.length > 0) {
            outputPrefix += args[0] + ", ";
        }
        String rootPath = args[1];
        if (args[0].equals("x86_HOST")) {
            testSuites0(rootPath, outputPrefix, AffinityType.CPU_HIGH_PERFORMANCE, DataType.FP32);
        } else {
            testSuites0(rootPath, outputPrefix, AffinityType.CPU_HIGH_PERFORMANCE, DataType.FP32);
            testSuites0(rootPath, outputPrefix, AffinityType.CPU_HIGH_PERFORMANCE, DataType.FP16);
            testSuites1(rootPath, outputPrefix, AffinityType.CPU_HIGH_PERFORMANCE);
            testSuites0(rootPath, outputPrefix, AffinityType.CPU_LOW_POWER, DataType.FP32);
            testSuites0(rootPath, outputPrefix, AffinityType.CPU_LOW_POWER, DataType.FP16);
            testSuites1(rootPath, outputPrefix, AffinityType.CPU_LOW_POWER);
            testSuites0(rootPath, outputPrefix, AffinityType.GPU, DataType.FP16);
        }
    }
}
