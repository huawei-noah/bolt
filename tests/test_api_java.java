// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


public final class test_api_java {
    public static double getMillisTime() {
        return System.nanoTime() / 1000.0 / 1000.0;
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
            sum += length;
            for (int j = 0; j < length; j++) {
                 if (Math.abs(arrayA[i][j] - arrayB[i][j]) > threshold) {
                     System.err.println("[ERROR] verify failed " + arrayA[i][j] + " "
                         + arrayB[i][j] + ", in Java API test");
                     System.exit(1);
                 }
            }
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

    public static void tinybert(String outputPrefix, DeviceType device, AffinityType affinity, String modelPath) {
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
        String prefix = "/data/local/CI/java/tmp";

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

        tinybert(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert/tinybert_f16.bolt");
        tinybert(outputPrefix, device, affinity, prefix+"/caffe_models/tinybert/tinybert_f32.bolt");
        nmt(outputPrefix, device, affinity, prefix+"/caffe_models/nmt/nmt_f16.bolt");
        nmt(outputPrefix, device, affinity, prefix+"/caffe_models/nmt/nmt_f32.bolt");

        classification(outputPrefix, device, affinity, prefix+"/caffe_models/squeezenet/squeezenet_int8_q.bolt",
             "data", DataType.FP16, image_3x224x224, 255, 310);
        classification(outputPrefix, device, affinity, prefix+"/onnx_models/birealnet18/birealnet18_f16.bolt",
            "0", DataType.FP16, image_3x224x224, 255, 565);
    }

    public static void main(String[] args) {
        String outputPrefix = "[INFO] ";
        if (args.length > 0) {
            outputPrefix += args[0] + ", ";
        }
        testSuites(outputPrefix, DeviceType.CPU, AffinityType.HIGH_PERFORMANCE);
        testSuites(outputPrefix, DeviceType.CPU, AffinityType.LOW_POWER);
        // testSuites(outputPrefix, DeviceType.GPU, AffinityType.LOW_POWER);
    }
}
