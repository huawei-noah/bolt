#ifndef GEMV_OPT
#define GEMV_OPT
#include "common_opt.h"

inline EE set_gemv_trans_mat_opt(U32 workMatChannelsPerThread,
    bool useReduceMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    U32 C = workMatChannelsPerThread;
    std::string transName = "";
    if (!useReduceMode) {
        CHECK_STATUS(set_chars_define_opt("USE_TRANS_CK", opt));
        transName = "kc_";
    }
    std::string kernel = std::string("gemv_trans_mat_") + transName + std::to_string(C);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "gemv_trans_mat");
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_value_define_opt(C, "C", opt));
    return SUCCESS;
}

inline EE get_gemv_cal_scheme(std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk)
{
    U32 c = 4;
    for (U32 i = 0; i < 4; i++) {
        vh->push_back(0);
        vc->push_back(c);
        vk->push_back(0);
        if (i == 2) {
            c = 8 << 4;
        } else {
            c = c << 1;
        }
    }
    return SUCCESS;
}

inline EE set_gemv_opt(U32 workMatChannelsPerThread,
    ActivationParamSpec activeMode,
    bool useBias,
    bool useReduceMode,
    bool useOutputNchwc4,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    U32 OC = workMatChannelsPerThread;
    char modeName[128];
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    std::string biasName = "";
    std::string outFormatName = "";
    std::string reduceName = "";
    if (useOutputNchwc4) {
        outFormatName = "oc4_";
    }
    if (!useBias) {
        biasName = "nobias_";
    }
    if (useReduceMode) {
        reduceName = "_reduce";
    }

    std::string source = "gemv" + reduceName;
    std::string kernel =
        source + std::string("_") + modeName + outFormatName + biasName + std::to_string(OC);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_value_define_opt(OC, "OC", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (!useBias) {
        CHECK_STATUS(set_chars_define_opt("NO_BIAS", opt));
    }
    if (useOutputNchwc4) {
        CHECK_STATUS(set_chars_define_opt("USE_OUTPUT_NCHWC4", opt));
    }
    return SUCCESS;
}

#endif
