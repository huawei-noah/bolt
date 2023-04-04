#ifndef _H_LAYER_NORM_OPT
#define _H_LAYER_NORM_OPT
#include "common_opt.h"

inline EE set_layer_norm_opt_mali(
    bool useNchwFormat, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    std::string source = "layer_norm";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, dt));
    CHECK_STATUS(add_macro_format(opt, useNchwFormat));
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}

#endif
