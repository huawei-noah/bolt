#ifndef COPY_OPT
#define COPY_OPT

#include "common_opt.h"

inline EE set_copy_opt_mali(bool useBlockIndex, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    std::string source = "copy";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, dt));
    if (useBlockIndex) {
        CHECK_STATUS(add_macro(opt, "USE_BLOCK_INDEX"));
    }
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}
#endif
