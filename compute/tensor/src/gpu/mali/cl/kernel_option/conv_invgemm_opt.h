#ifndef CONV_INVGEMM_OPT
#define CONV_INVGEMM_OPT
#include "common_opt.h"
inline EE set_conv_invgemm_trans_flt_opt(
    U32 workFiltersPerThread, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    U32 item_k = workFiltersPerThread;
    std::string kernel = std::string("conv_invgemm_trans_flt_") + std::to_string(item_k);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_invgemm_trans_flt");
    CHECK_STATUS(add_macro(opt, "K", item_k));
    return SUCCESS;
}

inline EE set_conv_invgemm_col2img_opt(ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    kernelOpt->kernelDataType = dt;
    char modeName[128];
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string kernel = std::string("conv_invgemm_col2img_") + ioMemName + modeName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_invgemm_col2img");
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
