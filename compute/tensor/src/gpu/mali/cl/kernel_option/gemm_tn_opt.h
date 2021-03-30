#ifndef GEMM_TN_OPT
#define GEMM_TN_OPT
#include "common_opt.h"
inline EE set_gemm_tn_opt_mali(U32 item_m,
    U32 item_n,
    bool useBiasMatchA,
    bool useBiasMatchB,
    bool usePointwiseNcwhc4,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    char modeName[128];
    std::string biasName = "nobias_";
    std::string formatName = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));
    if (useBiasMatchA) {
        biasName = "biasA_";
    } else if (useBiasMatchB) {
        biasName = "biasB_";
    }
    if (useBiasMatchA && useBiasMatchB) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (usePointwiseNcwhc4) {
        formatName = "pointwise_ncwhc4_";
        if (item_m != 4 && item_m != 8) {
            CHECK_STATUS(NOT_MATCH);
        }
    }
    if (item_m > 8 || item_n > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    sprintf(kernelName, "gemm_tn_%s%s%s%d%d", modeName, formatName.c_str(), biasName.c_str(),
        item_m, item_n);
    sprintf(kernelOpt->sourceName, "gemm_tn");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;

    U32 UN = item_n - 1;
    CHECK_STATUS(set_value_define_opt(item_m, "LM", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(item_n, "LN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(UN, "UN", opt, &len));
    opt += len;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    if (useBiasMatchA) {
        CHECK_STATUS(set_chars_define_opt("USE_BIAS_MATCH_A", opt, &len));
        opt += len;
    } else if (useBiasMatchB) {
        CHECK_STATUS(set_chars_define_opt("USE_BIAS_MATCH_B", opt, &len));
        opt += len;
    } else {
        CHECK_STATUS(set_chars_define_opt("NO_BIAS", opt, &len));
        opt += len;
    }
    if (usePointwiseNcwhc4) {
        CHECK_STATUS(set_chars_define_opt("USE_POINTWISE_NCWHC4", opt, &len));
        opt += len;
    }
    return SUCCESS;
}

#endif
