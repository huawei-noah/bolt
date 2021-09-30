#ifndef POOLING_OPT
#define POOLING_OPT
#include "common_opt.h"
#include "parameter_spec.h"
inline EE set_pooling_opt_mali(PoolingMode mode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    std::string modeName = "";
    char *opt = kernelOpt->option;
    if (mode == POOLING_MAX) {
        modeName = "max";
        CHECK_STATUS(set_chars_define_opt("USE_POOLING_MAX", opt));
    } else if (mode == POOLING_MEAN) {
        modeName = "mean";
        CHECK_STATUS(set_chars_define_opt("USE_POOLING_MEAN", opt));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    sprintf(kernelName, "pooling_%s%s", ioMemName, modeName.c_str());
    sprintf(kernelOpt->sourceName, "pooling");
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

#endif
