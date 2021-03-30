#ifndef SCALE_OPT
#define SCALE_OPT
#include "common_opt.h"
inline EE set_scale_opt_mali(bool useAlpha,
    bool useBeta,
    bool useNchwFormat,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    std::string formatName = "";
    std::string alphaName = "";
    std::string betaName = "";
    if (useNchwFormat) {
        formatName = "_nchw";
    }
    if (useAlpha) {
        alphaName = "_alpha";
    }
    if (useBeta) {
        betaName = "_beta";
    }

    sprintf(kernelName, "scale%s%s%s", formatName.c_str(), alphaName.c_str(), betaName.c_str());
    sprintf(kernelOpt->sourceName, "scale");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt, &len));
        opt += len;
    }
    if (useAlpha) {
        CHECK_STATUS(set_chars_define_opt("USE_ALPHA", opt, &len));
        opt += len;
    }
    if (useBeta) {
        CHECK_STATUS(set_chars_define_opt("USE_BETA", opt, &len));
        opt += len;
    }
    return SUCCESS;
}

#endif
