#ifndef GEMM_TN_OPT
#define GEMM_TN_OPT
#include "common_opt.h"

inline EE get_gemm_tn_pointwise_cal_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk, U32 fn, GCLMemType outputMemType)
{
    U32 k = 4;
    for (U32 i = 0; i < 2; i++) {
        for (U32 j = 1; j <= 8; j++) {
            if (outputMemType != GCL_MEM_BUF && (j % 4) != 0) {
                continue;
            }
            vh->push_back(j);
            vc->push_back(1);
            vk->push_back(k);
        }
        k = k << 1;
        if (fn % k != 0) {
            break;
        }
    }
    return SUCCESS;
}

inline EE get_gemm_tn_cal_scheme(std::vector<U32> *vh,
    std::vector<U32> *vc,
    std::vector<U32> *vk,
    GCLMemType matAMemType,
    GCLMemType matBMemType,
    GCLMemType matCMemType)
{
    for (U32 i = 1; i <= 8; i++) {
        for (U32 j = 1; j <= 8; j++) {
            if (matAMemType != GCL_MEM_BUF && (i % 4) != 0) {
                continue;
            }
            if ((matBMemType != GCL_MEM_BUF || matCMemType != GCL_MEM_BUF) && (j % 4) != 0) {
                continue;
            }
            if (i * j <= 2) {
                continue;
            }
            vh->push_back(j);
            vc->push_back(1);
            vk->push_back(i);
        }
    }
    return SUCCESS;
}

typedef enum {
    USE_BIAS_MATCH_A = 0,
    USE_BIAS_MATCH_B = 1,
    NO_BIAS = 2,
} OclGemmBiasMode;

inline EE set_gemm_tn_opt_mali(U32 item_m,
    U32 item_n,
    OclGemmBiasMode biasMode,
    bool usePointwiseNchwc4,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType matAMemType,
    GCLMemType matBMemType,
    GCLMemType matCMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char modeName[128];
    char *opt = kernelOpt->option;
    std::string biasName = "nobias_";
    std::string formatName = "";
    std::string matAMemName = "";
    std::string matBMemName = "";
    std::string matCMemName = "";
    char devName[128] = "";
    bool useQualcomm = false;
    if (!usePointwiseNchwc4) {
        useQualcomm = check_qualcomm_device(devName);
    }
    if (item_m > 8 || item_n > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    if (biasMode == USE_BIAS_MATCH_A) {
        biasName = "biasA_";
        CHECK_STATUS(set_chars_define_opt("USE_BIAS_MATCH_A", opt));
    } else if (biasMode == USE_BIAS_MATCH_B) {
        biasName = "biasB_";
        CHECK_STATUS(set_chars_define_opt("USE_BIAS_MATCH_B", opt));
    } else if (biasMode == NO_BIAS) {
        CHECK_STATUS(set_chars_define_opt("NO_BIAS", opt));
    } else {
        CHECK_STATUS(NOT_MATCH);
    }
    if (usePointwiseNchwc4) {
        formatName = "pointwise_nchwc4_";
        if (item_m != 4 && item_m != 8) {
            CHECK_STATUS(NOT_MATCH);
        }
        CHECK_STATUS(set_chars_define_opt("USE_POINTWISE_NCHWC4", opt));
    }
    if (matAMemType != GCL_MEM_BUF) {
        matAMemName = "am_";
        CHECK_STATUS(set_chars_define_opt("USE_INPUT_A_IMG", opt));
    }
    if (matBMemType != GCL_MEM_BUF) {
        matBMemName = "bm_";
        CHECK_STATUS(set_chars_define_opt("USE_INPUT_B_IMG", opt));
    }
    if (matCMemType != GCL_MEM_BUF) {
        matCMemName = "cm_";
        CHECK_STATUS(set_chars_define_opt("USE_OUTPUT_IMG", opt));
    }
    std::string source = std::string("gemm_tn") + devName;
    std::string kernel = source + std::string("_") + matAMemName + matBMemName + matCMemName +
        modeName + formatName + biasName + std::to_string(item_m) + std::to_string(item_n);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    kernelOpt->kernelDataType = dt;
    U32 UN = item_n - 1;
    CHECK_STATUS(set_value_define_opt(item_m, "LM", opt));
    CHECK_STATUS(set_value_define_opt(item_n, "LN", opt));
    CHECK_STATUS(set_value_define_opt(UN, "UN", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    return SUCCESS;
}

#endif
