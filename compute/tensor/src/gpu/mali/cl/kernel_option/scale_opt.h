#ifndef SCALE_OPT
#define SCALE_OPT
#include "common_opt.h"
inline EE set_scale_opt_mali(bool useAlpha,
    bool useBeta,
    bool useNchwFormat,
    bool useBroadCast,
    U32 axis,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    std::string formatName = "";
    std::string alphaName = "";
    std::string betaName = "";
    std::string broadName = "";
    std::string axisName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    if (useAlpha) {
        alphaName = "alpha_";
        CHECK_STATUS(set_chars_define_opt("USE_ALPHA", opt));
    }
    if (useBeta) {
        betaName = "beta_";
        CHECK_STATUS(set_chars_define_opt("USE_BETA", opt));
    }
    if (useBroadCast) {
        broadName = "broad_";
        CHECK_STATUS(set_chars_define_opt("USE_BROADCAST_MODE", opt));
    }
    if (axis == 0) {
        axisName = "w_";
        CHECK_STATUS(set_chars_define_opt("SCALE_ON_AXIS_W", opt));
    } else if (axis == 1) {
        axisName = "h_";
        CHECK_STATUS(set_chars_define_opt("SCALE_ON_AXIS_H", opt));
    } else if (axis == 2) {
        axisName = "c_";
        CHECK_STATUS(set_chars_define_opt("SCALE_ON_AXIS_C", opt));
    }

    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string kernel = std::string("scale_") + ioMemName + formatName + broadName + axisName +
        alphaName + betaName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "scale");
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

#endif
