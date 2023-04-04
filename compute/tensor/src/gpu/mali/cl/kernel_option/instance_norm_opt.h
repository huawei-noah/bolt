#ifndef _H_INSTANCE_NORM_OPT
#define _H_INSTANCE_NORM_OPT

#include "common_opt.h"

inline EE set_instance_norm_opt_mali(
    bool useNchwFormat, DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName, KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    std::string source = "instance_norm";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, dt));
    CHECK_STATUS(add_macro_format(opt, useNchwFormat));
    CHECK_STATUS(add_macro_io(opt, inputMemType, outputMemType));
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}

#endif
