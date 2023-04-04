#ifndef ACTIVATION_OPT
#define ACTIVATION_OPT
#include "common_opt.h"
inline EE set_activation_opt_mali(bool useNchwFormat,
    ActivationParamSpec p,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    char modeName[128] = "";
    CHECK_STATUS(set_activation_mode_name(p, modeName));
    std::string kernel = std::string("activation_") + ioMemName + formatName + modeName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "activation");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_activation_define_opt(p, opt));
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
