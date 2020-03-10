// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_FP16
#include <iostream>
#include <algorithm>
#include "ut_util.h"
#include "type.h"
#include "tensor_desc.h"
#include "sequential_ocl.hpp"
#include "factory.hpp"
#include "ocl/factory_ocl.hpp"
#include "tensor.hpp"
#include "data_loader.hpp"
const int& min(const int& a, const int& b)
{
    return (b < a) ? b : a;
}

const int& max(const int& a, const int& b)
{
    return (b < a) ? a : b;
}
void print_help() {

    std::cout << "please set argvs:      "  <<std::endl;
    std::cout << "usage: argv[1]:  in"      <<std::endl;
    std::cout << "usage: argv[2]:  ic"      <<std::endl;
    std::cout << "usage: argv[3]:  ih"      <<std::endl;
    std::cout << "usage: argv[4]:  iw"      <<std::endl;
    std::cout << "usage: argv[5]:  dt"      <<std::endl;
}

inline void calWeight(F16* para){
    float ccm[3][3] = {{0.900616, -0.079311, -0.068347}, {-0.100600, 0.919760, -0.069032}, {-0.058384, -0.037624, 0.975032}};
    float ccm_bias[3] = {0.036360, 0.062180, 0.064861};
    float shifts[3] = {-0.036361, -0.062179, -0.064860};
    float slopes[3] = {0.003211,  0.007948, 0.046259};
    float cmix[3] = {0.249512, 0.274577, 0.324276};
    float cmix_bias = 0.078941;
    float x[3];
    for(int i = 0; i < 3; ++i) x[i] = (ccm_bias[i] - shifts[i]) * slopes[i] * 16;
    for(int i = 0 ; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            ccm[i][j] = ccm[i][j] * slopes[i] * 16;
        }
    }
    for(int i = 0; i < 3; i++) para[i] = (F16)(ccm[i][0] * cmix[0] + ccm[i][1] * cmix[1] + ccm[i][2] * cmix[2]);
    para[3] = (F16)(x[0] * cmix[0] + x[1] * cmix[1] + x[2] * cmix[2] + cmix_bias);
}

template <typename T>
inline void calGuide(const int w, const int h, const int c, F16* para, T* input, F16* guide, std::string DATA_DT){
    float in_val[3];
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++){
             if(DATA_DT == "UCHAR"){
                 in_val[0] = input[c * (j + w * i)] / 256.0;
                 in_val[1] = input[c * (j + w * i) + 1] / 256.0;
                 in_val[2] = input[c * (j + w * i) + 2] / 256.0;
             } else {
                 in_val[0] = input[c * (j + w * i)];
                 in_val[1] = input[c * (j + w * i) + 1];
                 in_val[2] = input[c * (j + w * i) + 2];
             }
             guide[j + w * i] = in_val[0] * para[0] + in_val[1] * para[1] + in_val[2] * para[2] + para[3];
        }
    }
    
}
template <typename T>
inline void bilateralSliceApply(const int w, const int h, const int gw, const int gh, const int gd, const int input_chans, const int output_chans, const bool has_offset,
                         F16* grid, F16* guide, T* input, T* output, std::string DATA_DT)
{
     int grid_chans = input_chans * output_chans;
     int coeff_stride = input_chans;
     if(has_offset){
         grid_chans += output_chans;
         coeff_stride += 1;
     }
     int sz = grid_chans;
     int sx = grid_chans * gd;
     int sy = grid_chans * gd * gw;
   
     float in_val[3];
     float out_val[3];
     for(int y = 0; y < h; ++y){
         float gy = (y + 0.5f) * gh / (1.0f * h);
         int   fy = static_cast<int>(floor(gy - 0.5));
         for(int x = 0; x < w; ++x){
             float gx = (x + 0.5f) * gw / (1.0f * w);
             float gz = guide[x + w * y] * gd;
             int   fx = static_cast<int>(floor(gx - 0.5f));
             int   fz = static_cast<int>(floor(gz - 0.5f));
             float coeff_sample[12] = {0.0f};
             for(int xx = fx; xx < fx + 2; ++xx){
                 int x_ = max(min(xx, gw - 1), 0);
                 float wx = fmax(1.0f- fabs(xx + 0.5 - gx), 0.0f);
                 for(int yy = fy; yy < fy + 2; ++yy){
                     int y_ = max(min(yy, gh - 1), 0);
                     float wy = fmax(1.0f - fabs(yy + 0.5 - gy), 0.0f);
                     for(int zz = fz; zz < fz + 2; ++zz){
                         int z_ = max(min(zz, gd-1), 0);
                         float wz = fmax(1.0f - fabs(zz + 0.5 - gz), 0.0f);
                         for(int in_c = 0; in_c < grid_chans; ++in_c){
                             int grid_idx = in_c + sz * z_ + sx * x_ + sy * y_;
                             coeff_sample[in_c] += grid[grid_idx] * wx * wy * wz;
                         }
                     }
                 }
             }
             if(DATA_DT == "UCHAR"){
                 in_val[0] = input[input_chans * (x + w * y)] / 256.0;
                 in_val[1] = input[input_chans * (x + w * y) + 1] / 256.0;
                 in_val[2] = input[input_chans * (x + w * y) + 2] / 256.0;
             } else {
                 in_val[0] = input[input_chans * (x + w * y)];
                 in_val[1] = input[input_chans * (x + w * y) + 1];
                 in_val[2] = input[input_chans * (x + w * y) + 2];
             }

             if(has_offset){
                 out_val[0] = in_val[0] * coeff_sample[0] + in_val[1] * coeff_sample[1] + in_val[2] * coeff_sample[2]  + coeff_sample[3];
                 out_val[1] = in_val[0] * coeff_sample[4] + in_val[1] * coeff_sample[5] + in_val[2] * coeff_sample[6]  + coeff_sample[7];
                 out_val[2] = in_val[0] * coeff_sample[8] + in_val[1] * coeff_sample[9] + in_val[2] * coeff_sample[10] + coeff_sample[11];
             } else {
                 out_val[0] = in_val[0] * coeff_sample[0] + in_val[1] * coeff_sample[1] + in_val[2] * coeff_sample[2];
                 out_val[1] = in_val[0] * coeff_sample[3] + in_val[1] * coeff_sample[4] + in_val[2] * coeff_sample[5];
                 out_val[2] = in_val[0] * coeff_sample[6] + in_val[1] * coeff_sample[7] + in_val[2] * coeff_sample[8];
             }

             if(DATA_DT == "UCHAR"){
                 output[input_chans * (x + w * y)]     = (U8)(out_val[0] * 256.0);
                 output[input_chans * (x + w * y) + 1] = (U8)(out_val[1] * 256.0);
                 output[input_chans * (x + w * y) + 2] = (U8)(out_val[2] * 256.0);
             } else {
                 output[input_chans * (x + w * y)]     = out_val[0];
                 output[input_chans * (x + w * y) + 1] = out_val[1];
                 output[input_chans * (x + w * y) + 2] = out_val[2];
             }
         }
     }
}

template <typename T>
void HDR_CPU(const int w, const int h, const int gw, const int gh, const int gd, const int input_chans, const int output_chans, const bool has_offset,
             F16* grid, T* input, T* output, std::string DATA_DT){

    U8* guideptr = (U8*) operator new (w * h * bytesOf(DT_F16));
    F16* guide = (F16*) guideptr;
    F16  para[4];
    calWeight(para);
    calGuide<T>(w, h, input_chans, para, input, guide, DATA_DT);
    bilateralSliceApply<T>(w, h, gw, gh, gd, input_chans, output_chans, has_offset, grid, guide, input, output, DATA_DT);
}

template <typename T>
void buildInputTensor(DataType dt, DataFormat df, U32 n, U32 c, U32 h, U32 w, Vec<TensorDesc>* dims, Vec<Tensor>* inputTensors){
    TensorDesc inputDesc = tensor4df(dt, df, n, c, h, w);
    U32 inputNum  = tensorNumElements(inputDesc);
    U32 inputSize = tensorNumBytes(inputDesc); 
    U8* inputVal = new U8[inputSize];
   
    T* data = (T*) inputVal;
    if(dt == DT_F16){
        for(U32 i = 0; i < inputNum; i++) data[i] = (T)(rand() & 255) / (256.0);
    }
    if(dt == DT_U8){
        for(U32 i = 0; i < inputNum; i++) {
            data[i] = (T)(rand() & 255);
        }
    }
    std::shared_ptr<Tensor> inputTensor = std::shared_ptr<Tensor>(new Tensor());
    inputTensor->set_desc(inputDesc);
    inputTensor->set_val(inputVal);

    dims->push_back(inputDesc);
    inputTensors->push_back(*inputTensor.get());
}

int main(int argc, char* argv[]) {

    if(argc != 6 && argc != 5) {
        printf("%d\n", argc);
        print_help();
        return 0;
    }
    std::string INPUT_DT = "F16";
    U32 in     = atoi(argv[1]);
    U32 ic     = atoi(argv[2]);
    U32 ih     = atoi(argv[3]);
    U32 iw     = atoi(argv[4]);
    if(argc == 6) INPUT_DT = argv[5];
    U32 gw = 16;
    U32 gh = 16;
    U32 gc = 96;
    U32 gd = 8;
    U32 coe = gc / gd;
    bool has_offset = true;

    const Arch A = MALI;
    DataType dt = DT_F16;
    auto model = new SequentialOcl(A, dt, "OT_BilateralSliceApply");
    std::shared_ptr<SequentialOcl> model_ptr = std::shared_ptr<SequentialOcl>(model);

    Factory* factory_ocl = (Factory*)(new FactoryOCL());
    std::shared_ptr<Factory> factory;
    factory = std::shared_ptr<Factory>(factory_ocl);


    BilateralSliceApplyMode mode = BSliceApply_CONV;
    auto op = factory->createBilateralSliceApply(coe, has_offset, mode);
    model_ptr->add(op);

    Vec<TensorDesc> dims;
    Vec<Tensor> inputTensors;
    if(INPUT_DT=="UCHAR") {
        buildInputTensor<U8>(DT_U8, DF_NHWC, in, ic, ih, iw, &dims, &inputTensors);
    } else {
        buildInputTensor<F16>(dt, DF_NHWC, in, ic, ih, iw, &dims, &inputTensors);
    }
    buildInputTensor<F16>(dt, DF_NHWC, 1, gc, gh, gw, &dims, &inputTensors);//grid

    F16* grid_val = (F16*)inputTensors[1].get_val();
    for(U32 i = 0; i < tensorNumElements(dims[1]); i++) grid_val[i] = grid_val[i] / 8.0;
    U8* input = new U8[tensorNumBytes(dims[0])];
    U8* grid  = new U8[tensorNumBytes(dims[1])];
    memcpy((void*)input, inputTensors[0].get_val(), tensorNumBytes(dims[0]));
    memcpy((void*)grid,  inputTensors[1].get_val(), tensorNumBytes(dims[1]));

    model_ptr->ready(dims, NULL, 1);
    model_ptr->mark_input_output();
    model_ptr->mali_prepare();
    model_ptr->set_input_tensors(inputTensors);
    model_ptr->run();
    auto ocl_output = model_ptr->get_output_tensors_map();

    int e0, e1, e2, e3, e4, e5, e6;
    e0 = 0; e1 = 0; e2 = 0; e3 = 0; e4 = 0; e5 = 0; e6 = 0;
    float maxrel = 0;
    float maxabs = 0;
    if(INPUT_DT == "UCHAR") {
        U8* output = new U8[iw * ih * ic * sizeof(U8)];
        HDR_CPU<U8> (iw, ih, gw, gh, gd, ic, ic, has_offset, (F16*)grid, input, output, INPUT_DT);
        U8* ocl_res = ocl_output[0];
        for(U32 i = 0; i < ih; i++){
             for(U32 j = 0; j < iw; j++){
                 U8 c, g;
                 int d;
                 int index = (i * iw + j) * 3;
                     for(int k = 0 ; k < 3; k++){
                         c = output[index + k];
                         g = ocl_res[index + k];
                         d = c - g;
                         if(d < 0) d = -d;
                         maxabs = ((float)d > maxabs) ? (float)d : maxabs; 
                         maxrel = ((float)d * 2/ (c + g + 0.000001) > maxrel) ? (float)d * 2 / (c + g + 0.000001): maxrel;
                         if(d >= 30) {e0++; continue;}
                         if(d >= 20) {e1++; continue;}
                         if(d >= 10) {e2++; continue;}
                         if(d >= 5)  {e3++; continue;}
                         if(d >= 2)  {e4++; continue;}
                         if(d >= 1)  {e5++; continue;}
                         e6++;
                     }
             }
        }
        std::cout <<  "      abs(diff) >=30 number = " << e0 << std::endl;
        std::cout <<  "20 <= abs(diff) < 30 number = " << e1 << std::endl;
        std::cout <<  "10 <= abs(diff) < 20 number = " << e2 << std::endl;
        std::cout <<  "5  <= abs(diff) < 10 number = " << e3 << std::endl;
        std::cout <<  "2  <= abs(diff) < 5  number = " << e4 << std::endl;
        std::cout <<  "1  <= abs(diff) < 2  number = " << e5 << std::endl;
        std::cout <<  "0  <= abs(diff) < 1  number = " << e6 << std::endl;        
        std::cout <<  "maxabs = " << maxabs << std::endl;        
        std::cout <<  "maxrel = " << maxrel << std::endl;        
        delete[] output;
    } else {
        U8* output = new U8[iw * ih * ic * sizeof(F16)];
        HDR_CPU<F16>(iw, ih, gw, gh, gd, ic, ic, has_offset, (F16*)grid, (F16*)input, (F16*)output, INPUT_DT);
        F16* cpu_res = (F16*)output;
        F16* gpu_res = (F16*)ocl_output[0];
        for(U32 i = 0; i < ih; i++){
             for(U32 j = 0; j < iw; j++){
                 float c, g, d;
                 int index = (i * iw + j) * 3;
                     for(int k = 0 ; k < 3; k++){
                         c = cpu_res[index + k];
                         g = gpu_res[index + k];
                         d = c - g;
                         if(d < 0) d = -d;
                         maxabs = ((float)d > maxabs) ? (float)d : maxabs; 
                         maxrel = ((float)d * 2/ (c + g + 0.000001) > maxrel) ? (float)d * 2 / (c + g + 0.000001): maxrel;
                         if(d >= 1)       {e0++; continue;}
                         if(d >= 0.1)     {e1++; continue;}
                         if(d >= 0.01)    {e2++; continue;}
                         if(d >= 0.001)   {e3++; continue;}
                         if(d >= 0.0001)  {e4++; continue;}
                         if(d >= 0.00001) {e5++; continue;}
                         e6++;
                     }
             }
        }
        std::cout <<  "           abs(diff) >=1       number = " << e0 << std::endl;
        std::cout <<  "0.1     <= abs(diff) < 1       number = " << e1 << std::endl;
        std::cout <<  "0.01    <= abs(diff) < 0.1     number = " << e2 << std::endl;
        std::cout <<  "0.001   <= abs(diff) < 0.01    number = " << e3 << std::endl;
        std::cout <<  "0.0001  <= abs(diff) < 0.001   number = " << e4 << std::endl;
        std::cout <<  "0.00001 <= abs(diff) < 0.0001  number = " << e5 << std::endl;
        std::cout <<  "0       <= abs(diff) < 0.00001 number = " << e6 << std::endl;        
        std::cout <<  "maxabs = " << maxabs << std::endl;        
        std::cout <<  "maxrel = " << maxrel << std::endl;        
        delete[] output;
    }

    UTIL_TIME_STATISTICS

    delete[] input;
    delete[] grid;
    return 0;
}
#endif
