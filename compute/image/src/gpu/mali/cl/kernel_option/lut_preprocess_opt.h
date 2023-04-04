#ifndef _H_LUT_PREPROCESS_OPT
#define _H_LUT_PREPROCESS_OPT

#include "common_opt.h"

inline EE set_lut_preprocess_opt_mali(DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, kernelOpt->kernelDataType));
    std::string source = "lut_yuvnv21_preprocess";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}
#endif
