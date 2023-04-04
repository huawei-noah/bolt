#ifndef POOLING_OPT
#define POOLING_OPT

#include "common_opt.h"
#include "parameter_spec.h"

inline EE set_pooling_opt_mali(PoolingMode mode,
    DataType dt,
    DataFormat df,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    std::string source = "pooling";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, dt));
    CHECK_STATUS(add_macro_io(opt, inputMemType, outputMemType));
    switch (mode) {
        case POOLING_MAX: {
            CHECK_STATUS(add_macro(opt, "USE_POOLING_MAX"));
            break;
        }
        case POOLING_MEAN: {
            CHECK_STATUS(add_macro(opt, "USE_POOLING_MEAN"));
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    if (df != DF_NCHWC4) {
        CHECK_STATUS(add_macro(opt, "USE_NCHW"));
    }
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}

#endif
