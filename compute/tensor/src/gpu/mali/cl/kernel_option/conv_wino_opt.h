#ifndef CONV_WINO_OPT
#define CONV_WINO_OPT
#include "common_opt.h"
inline EE set_conv_wino_rotate_flt(
    U32 fw, U32 fh, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    U32 fwh = fw * fh;
    std::string kernel = std::string("conv_wino_rotate_fltbuf_") + std::to_string(fwh);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_wino_rotate_fltbuf");
    CHECK_STATUS(set_value_define_opt(fwh, "FWH", opt));
    return SUCCESS;
}

inline EE set_conv_wino_preprocess_input_opt(DataType dt,
    bool useNchwFormat,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char ioMemName[128] = "";
    char *opt = kernelOpt->option;
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw";
    }
    std::string kernel = std::string("conv_wino_preprocess_input_") + ioMemName + formatName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_wino_preprocess_input");
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    return SUCCESS;
}

inline EE set_conv_wino_trans_outbuf_opt(bool useAlign,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char modeName[128];
    char *opt = kernelOpt->option;
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string alignName = "";
    if (useAlign) {
        alignName = "align";
    }
    std::string kernel = std::string("conv_wino_trans_outbuf_") + ioMemName + modeName + alignName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_wino_trans_outbuf");
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (useAlign) {
        CHECK_STATUS(set_chars_define_opt("USE_ALIGN", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
