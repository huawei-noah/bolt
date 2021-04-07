#ifndef ACTIVATION_OPT
#define ACTIVATION_OPT
#include "common_opt.h"
inline EE set_activation_opt_mali(bool useNchwFormat,
    ActivationMode activeMode,
    DataType dt,
    char* kernelName,
    KernelOpt* kernelOpt)
{
    U32 len = 0;
    std::string formatName= "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));
    sprintf(kernelName, "activation_%s%s", formatName.c_str(), modeName);
    sprintf(kernelOpt->sourceName, "activation");
    kernelOpt->kernelDataType = dt;
    char* opt = kernelOpt->option;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt, &len));
        opt += len;
    }
    return SUCCESS;
}
#endif

