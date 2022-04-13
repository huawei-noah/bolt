// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>
#include <tests/GTestExtensions.h>

#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/trainable/Batchnorm.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

namespace UT
{

using namespace raul;

using dtype = raul::dtype;
using shape = raul::shape;
using dvec = std::vector<dtype>;

struct TestCNNBatchNormUnit : public testing::TestWithParam<std::tuple<shape, NetworkMode, dvec, dvec>>
{
    static constexpr dtype EPSILON = 1e-4_dt;

    const shape inputShape = std::get<0>(GetParam());
    const NetworkMode networkMode = std::get<1>(GetParam());
    const dvec& inputData = std::get<2>(GetParam());
    const dvec& forwardPassResult = std::get<3>(GetParam());
};

INSTANTIATE_TEST_SUITE_P(
    TestCNNBatchNorm,
    TestCNNBatchNormUnit,
    testing::Values(
        std::make_tuple(shape(1u, 3u, 3u, 1u),
                        NetworkMode::Train,
                        dvec{ 3_dt, 7_dt, 4_dt, 8_dt, 9_dt, 5_dt, 3_dt, 15_dt, 15_dt },
                        dvec{ -0.98058051_dt, 1.37281271_dt, -0.3922322_dt, 0.3922322_dt, 0.98058051_dt, -1.37281271_dt, -1.41421354_dt, 0.70710677_dt, 0.70710677_dt }),
        std::make_tuple(shape(1u, 5u, 5u, 1u),
                        NetworkMode::Train,
                        dvec{ 0.2992_dt, 0.0614_dt, 0.3442_dt, 0.4992_dt, 0.1848_dt, 0.3404_dt, 0.3627_dt, 0.6232_dt, 0.5426_dt, 0.1261_dt, 0.9982_dt, 0.7149_dt, 0.8062_dt,
                              0.6040_dt, 0.0333_dt, 0.3870_dt, 0.2276_dt, 0.0830_dt, 0.0222_dt, 0.9375_dt, 0.9395_dt, 0.4894_dt, 0.4846_dt, 0.3932_dt, 0.3220_dt },
                        dvec{ 0.1449_dt,  -1.4626_dt, 0.4491_dt, 1.4969_dt,  -0.6284_dt, -0.3381_dt, -0.2095_dt, 1.2937_dt, 0.8286_dt,  -1.5747_dt, 1.1262_dt,  0.2566_dt, 0.5368_dt,
                              -0.0839_dt, -1.8358_dt, 0.1692_dt, -0.3165_dt, -0.7570_dt, -0.9423_dt, 1.8466_dt,  1.9153_dt, -0.1682_dt, -0.1904_dt, -0.6135_dt, -0.9431_dt }),
        std::make_tuple(shape(5u, 5u, 1u, 1u),
                        NetworkMode::Train,
                        dvec{ 0.2992_dt, 0.0614_dt, 0.3442_dt, 0.4992_dt, 0.1848_dt, 0.3404_dt, 0.3627_dt, 0.6232_dt, 0.5426_dt, 0.1261_dt, 0.9982_dt, 0.7149_dt, 0.8062_dt,
                              0.6040_dt, 0.0333_dt, 0.3870_dt, 0.2276_dt, 0.0830_dt, 0.0222_dt, 0.9375_dt, 0.9395_dt, 0.4894_dt, 0.4846_dt, 0.3932_dt, 0.3220_dt },
                        dvec{ -0.9509_dt, -1.3887_dt, -0.5044_dt, 0.4205_dt,  -0.4218_dt, -0.8175_dt, -0.0381_dt, 0.6302_dt, 0.6303_dt, -0.6039_dt, 1.3125_dt,  1.5407_dt, 1.3744_dt,
                              0.9272_dt,  -0.8918_dt, -0.6666_dt, -0.6437_dt, -1.5666_dt, -1.8859_dt, 1.9135_dt,  1.1225_dt, 0.5299_dt, 0.0665_dt,  -0.0921_dt, 0.0039_dt }),
        std::make_tuple(
            shape(3u, 4u, 5u, 6u),
            NetworkMode::Train,
            dvec{ 9.77800555e-02_dt, 3.53264511e-01_dt, 2.66772072e-01_dt, 2.30766091e-01_dt, 5.32899343e-01_dt, 6.20913405e-01_dt, 7.86335246e-01_dt, 2.74646282e-01_dt, 3.12343771e-01_dt,
                  4.02963045e-01_dt, 4.85350802e-01_dt, 2.89835029e-01_dt, 1.70157161e-01_dt, 1.00855877e-01_dt, 5.98972123e-01_dt, 8.27687293e-01_dt, 4.88252325e-01_dt, 1.68515994e-01_dt,
                  5.70838646e-01_dt, 5.30862343e-01_dt, 1.31262474e-01_dt, 6.30811189e-01_dt, 9.13114234e-01_dt, 6.81991949e-01_dt, 3.23860465e-01_dt, 2.06175012e-02_dt, 2.03504954e-01_dt,
                  6.40338708e-01_dt, 5.53627927e-01_dt, 5.40967746e-02_dt, 1.58681187e-01_dt, 6.56620605e-01_dt, 4.42853989e-01_dt, 5.08290012e-01_dt, 3.22312184e-01_dt, 7.21623362e-01_dt,
                  3.49950677e-01_dt, 8.75714009e-01_dt, 8.63942188e-01_dt, 6.82456763e-01_dt, 2.73577567e-01_dt, 8.48642819e-01_dt, 3.65199037e-01_dt, 1.05601541e-02_dt, 9.80572690e-01_dt,
                  8.60514799e-01_dt, 4.84604094e-01_dt, 7.60105140e-01_dt, 4.66072587e-01_dt, 2.47301122e-01_dt, 6.40744327e-02_dt, 7.28925523e-01_dt, 8.53794478e-01_dt, 6.68397252e-01_dt,
                  6.84656544e-01_dt, 8.09576923e-01_dt, 8.25028327e-01_dt, 6.65535876e-01_dt, 4.29366847e-01_dt, 2.77251618e-01_dt, 7.91509393e-01_dt, 8.41398446e-01_dt, 2.28866685e-01_dt,
                  2.29016293e-01_dt, 2.22881412e-01_dt, 2.69195090e-01_dt, 8.00508533e-01_dt, 7.41912269e-02_dt, 7.55123748e-01_dt, 2.75609209e-01_dt, 9.48978733e-01_dt, 3.30928431e-01_dt,
                  5.77432716e-01_dt, 3.49660816e-01_dt, 9.31502288e-01_dt, 5.16939480e-01_dt, 9.10499887e-01_dt, 1.73638747e-01_dt, 3.88130457e-01_dt, 3.45477286e-01_dt, 6.86662669e-01_dt,
                  7.92149137e-01_dt, 3.90295754e-02_dt, 1.10321019e-01_dt, 4.97502057e-02_dt, 8.32182702e-01_dt, 7.64158188e-01_dt, 6.96318842e-01_dt, 9.49956434e-01_dt, 5.03834540e-02_dt,
                  7.86579335e-02_dt, 6.30414127e-01_dt, 2.22023199e-02_dt, 3.94991261e-01_dt, 6.50833365e-01_dt, 8.60519904e-01_dt, 2.96603298e-01_dt, 9.47043113e-02_dt, 6.35972032e-01_dt,
                  8.68504949e-01_dt, 4.91585448e-01_dt, 6.99195166e-01_dt, 8.71770936e-01_dt, 3.74931195e-01_dt, 5.01519928e-01_dt, 1.83905789e-01_dt, 6.51205570e-01_dt, 3.66360260e-01_dt,
                  1.36366328e-01_dt, 4.05065553e-02_dt, 8.60385871e-01_dt, 2.65456852e-01_dt, 6.56418341e-01_dt, 8.94826597e-01_dt, 2.48995226e-02_dt, 4.37947070e-01_dt, 5.85282099e-01_dt,
                  6.71394233e-01_dt, 6.02241152e-01_dt, 9.65663736e-01_dt, 4.00695162e-01_dt, 9.93102079e-01_dt, 4.02632450e-01_dt, 4.22737765e-01_dt, 1.73848008e-01_dt, 1.76786305e-01_dt,
                  8.66916864e-01_dt, 5.66841437e-03_dt, 8.03129096e-01_dt, 4.42671085e-01_dt, 9.16227084e-01_dt, 7.00328453e-01_dt, 9.44741760e-02_dt, 9.34049081e-01_dt, 2.03394853e-01_dt,
                  6.23594385e-01_dt, 5.85712888e-01_dt, 4.08449001e-01_dt, 3.68527464e-01_dt, 2.08109771e-01_dt, 3.45570101e-01_dt, 1.83936504e-02_dt, 1.56312847e-01_dt, 8.87980434e-01_dt,
                  9.37872032e-02_dt, 4.28870517e-01_dt, 5.50659311e-01_dt, 4.32980322e-02_dt, 8.18403199e-01_dt, 7.09507867e-01_dt, 3.46883176e-01_dt, 3.78409741e-01_dt, 8.77498840e-01_dt,
                  2.45052710e-01_dt, 4.22903936e-01_dt, 6.44538239e-01_dt, 4.99365558e-01_dt, 3.69989594e-01_dt, 7.50699293e-02_dt, 9.56065916e-01_dt, 9.51984030e-01_dt, 2.78706080e-01_dt,
                  2.69315753e-01_dt, 6.01302626e-01_dt, 7.02577001e-01_dt, 6.64154408e-01_dt, 7.14875033e-01_dt, 3.27179933e-01_dt, 5.91972917e-02_dt, 7.14381080e-01_dt, 8.87892953e-01_dt,
                  3.20857017e-01_dt, 8.24997813e-01_dt, 6.86640613e-01_dt, 9.24991170e-02_dt, 5.08710774e-01_dt, 3.86655566e-01_dt, 9.48000320e-03_dt, 6.70939201e-01_dt, 8.41139957e-01_dt,
                  5.14979669e-01_dt, 7.15719938e-01_dt, 1.92648993e-01_dt, 5.03256477e-01_dt, 4.63148905e-01_dt, 6.85503830e-01_dt, 1.98912970e-01_dt, 5.84024824e-01_dt, 8.88997997e-01_dt,
                  5.31634905e-02_dt, 6.06487929e-01_dt, 4.29464322e-01_dt, 2.36725764e-01_dt, 2.28998152e-01_dt, 6.25303905e-01_dt, 1.50221802e-01_dt, 3.37435605e-01_dt, 5.36270640e-01_dt,
                  8.64611451e-01_dt, 9.78729876e-01_dt, 1.93830601e-01_dt, 5.17394174e-01_dt, 2.29681530e-01_dt, 5.78217062e-01_dt, 8.62663963e-01_dt, 3.21392055e-01_dt, 8.45685916e-01_dt,
                  5.50340659e-01_dt, 6.55309423e-02_dt, 5.65544950e-01_dt, 7.15176673e-01_dt, 5.79537144e-01_dt, 8.33700710e-01_dt, 4.74959880e-01_dt, 2.54744694e-01_dt, 7.42147328e-01_dt,
                  1.58136040e-01_dt, 3.85223493e-01_dt, 9.85215764e-01_dt, 5.16292160e-01_dt, 1.01594892e-01_dt, 9.20983496e-01_dt, 8.60057920e-01_dt, 8.47588110e-01_dt, 3.08111721e-01_dt,
                  5.05987773e-01_dt, 4.65895734e-01_dt, 2.90976326e-01_dt, 8.56196221e-02_dt, 7.32879282e-01_dt, 6.84766994e-01_dt, 5.69435709e-01_dt, 1.88649486e-01_dt, 1.61360028e-01_dt,
                  3.99288566e-01_dt, 1.49470396e-02_dt, 3.89089149e-01_dt, 5.26952393e-01_dt, 4.16219183e-01_dt, 9.52534523e-01_dt, 4.60849520e-01_dt, 2.26519364e-01_dt, 2.63218338e-01_dt,
                  5.53670207e-01_dt, 8.41728726e-01_dt, 1.42615284e-01_dt, 3.14047941e-01_dt, 9.47645952e-01_dt, 5.68324155e-01_dt, 2.92418953e-01_dt, 5.85206850e-01_dt, 7.16189851e-01_dt,
                  2.79699618e-01_dt, 8.08198921e-01_dt, 1.31956514e-01_dt, 5.74294152e-01_dt, 6.14908312e-01_dt, 9.42539483e-01_dt, 9.35341614e-01_dt, 6.83345064e-01_dt, 7.55447454e-01_dt,
                  5.78529630e-01_dt, 8.18808225e-01_dt, 4.70827923e-01_dt, 1.54525369e-01_dt, 4.73310652e-01_dt, 5.81352818e-01_dt, 6.58587084e-01_dt, 9.35398824e-01_dt, 9.30131433e-01_dt,
                  9.57851946e-01_dt, 4.22153639e-01_dt, 1.98542014e-01_dt, 5.80882907e-01_dt, 8.65020864e-01_dt, 3.95721425e-01_dt, 7.18954571e-01_dt, 1.15222564e-01_dt, 3.97037035e-01_dt,
                  1.96768928e-01_dt, 9.32734804e-01_dt, 9.93195003e-01_dt, 1.30222986e-01_dt, 6.67470852e-01_dt, 2.29211835e-01_dt, 6.18951999e-01_dt, 9.21457211e-01_dt, 5.57187707e-01_dt,
                  9.45881969e-01_dt, 2.84178130e-02_dt, 8.78387170e-01_dt, 1.10789595e-02_dt, 3.54636486e-01_dt, 2.91366752e-01_dt, 1.64180135e-01_dt, 5.42968507e-01_dt, 7.21547476e-03_dt,
                  6.14725367e-01_dt, 1.96544825e-01_dt, 3.44890029e-01_dt, 8.08677890e-02_dt, 4.79162747e-02_dt, 6.77029300e-01_dt, 3.62922112e-01_dt, 2.93552370e-02_dt, 7.51271351e-01_dt,
                  8.86820681e-01_dt, 3.13008928e-01_dt, 4.78607774e-01_dt, 1.75519517e-04_dt, 4.71257801e-01_dt, 6.27473921e-01_dt, 4.00250703e-01_dt, 1.22122798e-01_dt, 5.89811442e-01_dt,
                  2.85812992e-02_dt, 2.46674273e-01_dt, 7.55656793e-01_dt, 2.59860216e-01_dt, 7.53702495e-02_dt, 1.64286066e-01_dt, 2.12510764e-01_dt, 3.78010682e-01_dt, 6.76076391e-01_dt,
                  4.51130171e-01_dt, 5.39201458e-01_dt, 4.96618901e-01_dt, 2.51728547e-01_dt, 1.70370586e-01_dt, 7.22909479e-01_dt, 5.15302168e-01_dt, 4.94214077e-01_dt, 7.77286501e-01_dt,
                  8.66233252e-01_dt, 7.17398895e-01_dt, 8.74810652e-01_dt, 6.11908038e-01_dt, 3.57020557e-01_dt, 5.99800063e-01_dt, 4.47885923e-01_dt, 3.08365632e-01_dt, 7.87660046e-01_dt,
                  5.53478768e-01_dt, 9.89349385e-01_dt, 6.24752792e-01_dt, 5.83571353e-01_dt, 1.24808488e-01_dt, 2.08414956e-01_dt, 2.27309689e-01_dt, 1.11749298e-01_dt, 8.30051922e-01_dt,
                  2.20119350e-01_dt, 2.60861680e-01_dt, 8.93376200e-01_dt, 1.40987484e-01_dt, 3.73134442e-01_dt, 7.27785481e-01_dt, 1.11244021e-01_dt, 8.54631643e-01_dt, 9.52460160e-01_dt },
            dvec{
                -1.3869_dt, -0.4622_dt, -0.7753_dt, -0.9056_dt, 0.1880_dt,  0.5065_dt,  1.1053_dt,  -0.7468_dt, -0.6103_dt, -0.2823_dt, 0.0159_dt,  -0.6918_dt, -1.1250_dt, -1.3758_dt, 0.4271_dt,
                1.2550_dt,  0.0264_dt,  -1.1309_dt, 0.3253_dt,  0.1806_dt,  -1.2657_dt, 0.5424_dt,  1.5642_dt,  0.7276_dt,  -0.5686_dt, -1.6662_dt, -1.0043_dt, 0.5769_dt,  0.2630_dt,  -1.5450_dt,
                -1.2474_dt, 0.4880_dt,  -0.2570_dt, -0.0290_dt, -0.6771_dt, 0.7145_dt,  -0.5808_dt, 1.2515_dt,  1.2105_dt,  0.5780_dt,  -0.8470_dt, 1.1572_dt,  -0.5277_dt, -1.7636_dt, 1.6170_dt,
                1.1986_dt,  -0.1115_dt, 0.8486_dt,  -0.1761_dt, -0.9385_dt, -1.5771_dt, 0.7400_dt,  1.1751_dt,  0.5290_dt,  0.5857_dt,  1.0210_dt,  1.0749_dt,  0.5190_dt,  -0.3040_dt, -0.8342_dt,
                1.2089_dt,  1.3868_dt,  -0.7975_dt, -0.7969_dt, -0.8188_dt, -0.6537_dt, 1.2410_dt,  -1.3490_dt, 1.0791_dt,  -0.6308_dt, 1.7704_dt,  -0.4335_dt, 0.4455_dt,  -0.3667_dt, 1.7081_dt,
                0.2298_dt,  1.6332_dt,  -0.9944_dt, -0.2295_dt, -0.3816_dt, 0.8350_dt,  1.2112_dt,  -1.4744_dt, -1.2202_dt, -1.4362_dt, 1.3539_dt,  1.1114_dt,  0.8694_dt,  1.7739_dt,  -1.4339_dt,
                -1.5415_dt, 0.4238_dt,  -1.7426_dt, -0.4148_dt, 0.4965_dt,  1.2434_dt,  -0.7652_dt, -1.4844_dt, 0.4436_dt,  1.2719_dt,  -0.0707_dt, 0.6688_dt,  1.2835_dt,  -0.4862_dt, -0.0353_dt,
                -1.1666_dt, 0.4978_dt,  -0.5168_dt, -1.3360_dt, -1.6774_dt, 1.2429_dt,  -0.8762_dt, 0.5164_dt,  1.3656_dt,  -1.7330_dt, -0.2618_dt, 0.2630_dt,  0.5698_dt,  0.3234_dt,  1.6179_dt,
                -0.2905_dt, 1.8537_dt,  -0.2835_dt, -0.2107_dt, -1.1116_dt, -1.1010_dt, 1.3970_dt,  -1.7203_dt, 1.1661_dt,  -0.1386_dt, 1.5754_dt,  0.7940_dt,  -1.3989_dt, 1.6399_dt,  -1.0047_dt,
                0.5163_dt,  0.3791_dt,  -0.2625_dt, -0.4070_dt, -0.9876_dt, -0.4901_dt, -1.6743_dt, -1.1751_dt, 1.4732_dt,  -1.4014_dt, -0.1886_dt, 0.2523_dt,  -1.5841_dt, 1.2214_dt,  0.8272_dt,
                -0.5915_dt, -0.4816_dt, 1.2577_dt,  -0.9464_dt, -0.3266_dt, 0.4459_dt,  -0.0601_dt, -0.5110_dt, -1.5388_dt, 1.5316_dt,  1.5173_dt,  -0.8291_dt, -0.8618_dt, 0.2952_dt,  0.6481_dt,
                0.5142_dt,  0.6910_dt,  -0.6602_dt, -1.5941_dt, 0.6893_dt,  1.2940_dt,  -0.6822_dt, 1.0748_dt,  0.5926_dt,  -1.4780_dt, -0.0275_dt, -0.4529_dt, -1.7674_dt, 0.5379_dt,  1.1310_dt,
                0.2228_dt,  0.9386_dt,  -0.9266_dt, 0.1810_dt,  0.0380_dt,  0.8309_dt,  -0.9043_dt, 0.4690_dt,  1.5565_dt,  -1.4240_dt, 0.5491_dt,  -0.0822_dt, -0.7695_dt, -0.7970_dt, 0.6162_dt,
                -1.0779_dt, -0.4103_dt, 0.2987_dt,  1.4696_dt,  1.8765_dt,  -0.9224_dt, 0.2314_dt,  -0.7946_dt, 0.4483_dt,  1.4626_dt,  -0.4675_dt, 1.4021_dt,  0.3489_dt,  -1.3799_dt, 0.4031_dt,
                0.7257_dt,  0.2426_dt,  1.1479_dt,  -0.1299_dt, -0.9143_dt, 0.8218_dt,  -1.2584_dt, -0.4496_dt, 1.6876_dt,  0.0173_dt,  -1.4598_dt, 1.4588_dt,  1.2418_dt,  1.1974_dt,  -0.7242_dt,
                -0.0194_dt, -0.1622_dt, -0.7853_dt, -1.5167_dt, 0.7888_dt,  0.6174_dt,  0.2066_dt,  -1.1498_dt, -1.2470_dt, -0.3995_dt, -1.7685_dt, -0.4358_dt, 0.0553_dt,  -0.3392_dt, 1.5712_dt,
                -0.0728_dt, -0.9210_dt, -0.7881_dt, 0.2632_dt,  1.3058_dt,  -1.2246_dt, -0.6042_dt, 1.6892_dt,  0.3162_dt,  -0.6824_dt, 0.3773_dt,  0.8514_dt,  -0.7285_dt, 1.1844_dt,  -1.2632_dt,
                0.3378_dt,  0.4848_dt,  1.6707_dt,  1.6446_dt,  0.7325_dt,  0.9935_dt,  0.3531_dt,  1.2228_dt,  -0.0367_dt, -1.1815_dt, -0.0277_dt, 0.3634_dt,  0.6429_dt,  1.6448_dt,  1.6258_dt,
                1.5378_dt,  -0.3292_dt, -1.1085_dt, 0.2240_dt,  1.2143_dt,  -0.4213_dt, 0.7052_dt,  -1.3988_dt, -0.4167_dt, -1.1146_dt, 1.4502_dt,  1.6610_dt,  -1.3466_dt, 0.5258_dt,  -1.0016_dt,
                0.3567_dt,  1.4109_dt,  0.1414_dt,  1.4961_dt,  -1.7014_dt, 1.2608_dt,  -1.7618_dt, -0.5645_dt, -0.7850_dt, -1.2282_dt, 0.0919_dt,  -1.7753_dt, 0.3420_dt,  -1.1154_dt, -0.5984_dt,
                -1.3252_dt, -1.4427_dt, 0.8007_dt,  -0.3194_dt, -1.5089_dt, 1.0654_dt,  1.5488_dt,  -0.4974_dt, 0.0931_dt,  -1.6130_dt, 0.0669_dt,  0.6239_dt,  -0.1863_dt, -1.1781_dt, 0.4896_dt,
                -1.5117_dt, -0.7340_dt, 1.0810_dt,  -0.6870_dt, -1.3448_dt, -1.0278_dt, -0.8558_dt, -0.2656_dt, 0.7973_dt,  -0.0049_dt, 0.3092_dt,  0.1573_dt,  -0.7160_dt, -1.0061_dt, 0.9643_dt,
                0.0138_dt,  -0.0613_dt, 0.9469_dt,  1.2638_dt,  0.7336_dt,  1.2943_dt,  0.3579_dt,  -0.5500_dt, 0.3147_dt,  -0.2264_dt, -0.7233_dt, 0.9839_dt,  0.1498_dt,  1.7023_dt,  0.4036_dt,
                0.2569_dt,  -1.3771_dt, -1.0793_dt, -1.0120_dt, -1.4237_dt, 1.1349_dt,  -1.0377_dt, -0.8925_dt, 1.3604_dt,  -1.3195_dt, -0.4926_dt, 0.7706_dt,  -1.4255_dt, 1.2224_dt,  1.5709_dt })));

TEST_P(TestCNNBatchNormUnit, Unit)
{
    PROFILE_TEST
    size_t batch = inputShape[0];

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in" }, inputShape[1], inputShape[2], inputShape[3] });
    BatchNormLayer bn("bn", BatchnormParams{ { "in" }, { "out" } }, networkParameters);
    TENSORS_CREATE(batch);
    bn.initNotBSTensors();
    std::copy(inputData.begin(), inputData.end(), &memory_manager["in"][0]);

    bn.forwardCompute(networkMode);
    const Tensor& out = memory_manager["out"];

    ASSERT_INTERVALS_NEAR(out.begin(), out.end(), forwardPassResult.begin(), forwardPassResult.end(), EPSILON);

    printf(" - BatchNorm forward [%zd, %zd, %zd, %zd] is Ok.\n", inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
}

}