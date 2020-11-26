// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef AUDIO_FEATURE_H_
#define AUDIO_FEATURE_H_

#include <vector>
#include <complex>

using cd = std::complex<double>;

class AudioFeatureExtractor {
public:
    static std::vector<std::vector<std::vector<float>>> getEncoderInputFromWav(
        std::string wavFilePath);

    static std::vector<std::vector<std::vector<float>>> getEncoderInput(std::vector<short> signal,
        std::vector<short> lastPoints,
        bool padding);  // padding false

private:
    static constexpr int FRAME_STEP = 160;
    static constexpr int W_LENGTH = 400;  // window length
    static constexpr int N_FFT = 512;     // Num of FFT length
    static constexpr int N_DIM = N_FFT / 2 + 1;
    static constexpr int N_FILTERS = 128;  // N_FILTERS = 41;

    static constexpr int SAMPLE_RATE = 16000;
    static constexpr double LOWER_HERZ_FREQ = 0;
    static constexpr double UPPER_HERZ_FREQ = 8000;
    static constexpr float EPSILON = 2.2204460492503131e-16F;
    static constexpr float LOG_EPSILON = -36.043653389F;

    static constexpr float _MEL_BREAK_FREQUENCY_HERTZ = 700.0F;
    static constexpr float _MEL_HIGH_FREQUENCY_Q = 1127.0F;

    static void PreEmphasis(std::vector<short> &signal, short lastPoint, std::vector<float> &output);

    static void SplitToFrames(
        std::vector<float> &signal, std::vector<std::vector<float>> &output, int nFrames);

    static void CentralPadding(std::vector<float> &signal, std::vector<float> &output);

    static std::vector<float> GetMelBankForSingleFrame(std::vector<float> frame);

    static void AddHammingWindow(std::vector<float> &data);

    static void fft(std::vector<cd> &a, bool invert);

    static std::vector<float> ComputePowerSpec(std::vector<double> fft);

    static std::vector<float> GetHammingWindow(bool periodic);

    static std::vector<std::vector<float>> GetLinearToMelMatrix();

    static std::vector<double> LineSpace(double lower, double upper, int number);

    static std::vector<double> HerzToMel(std::vector<double> herzVec);

    static double HerzToMel(double herz);

    static int getWavHead(FILE *file);

    static std::vector<short> readWav(const std::string &wavName);
};

#endif  // AUDIO_FEATURE_H_
