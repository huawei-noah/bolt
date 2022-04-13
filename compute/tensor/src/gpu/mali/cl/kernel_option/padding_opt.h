#ifndef PADDING_OPT
#define PADDING_OPT
#include "common_opt.h"
inline EE set_padding_opt_mali(bool useNchwFormat,
    PadMode mode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    std::string name = "padding_";
    if (useNchwFormat) {
        name += "nchw_";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string modeName;
    switch (mode) {
        case PAD_CONSTANT:
            modeName = "constant";
            CHECK_STATUS(set_chars_define_opt("USE_CONSTANT", opt));
            break;
        case PAD_EDGE:
            modeName = "edge";
            CHECK_STATUS(set_chars_define_opt("USE_EDGE", opt));
            break;
        case PAD_REFLECT:
            modeName = "reflect";
            CHECK_STATUS(set_chars_define_opt("USE_REFLECT", opt));
            break;
        case PAD_SYMMETRIC:
            modeName = "symmetric";
            CHECK_STATUS(set_chars_define_opt("USE_SYMMETRIC", opt));
            break;
        default:
            return NOT_SUPPORTED;
    }
    name += modeName;
    UNI_STRCPY(kernelName, name.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "padding");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
