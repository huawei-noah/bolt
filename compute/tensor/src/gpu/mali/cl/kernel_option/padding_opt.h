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
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string modeName = "";
    switch (mode) {
        case Pad_Constant:
            modeName = "constant";
            CHECK_STATUS(set_chars_define_opt("USE_CONSTANT", opt));
            break;
        case Pad_Edge:
            modeName = "edge";
            CHECK_STATUS(set_chars_define_opt("USE_EDGE", opt));
            break;
        case Pad_Reflect:
            modeName = "reflect";
            CHECK_STATUS(set_chars_define_opt("USE_REFLECT", opt));
            break;
        case Pad_Symmetric:
            modeName = "symmetric";
            CHECK_STATUS(set_chars_define_opt("USE_SYMMETRIC", opt));
            break;
        default:
            return NOT_SUPPORTED;
    }

    sprintf(kernelName, "padding_%s%s", formatName.c_str(), modeName.c_str());
    sprintf(kernelOpt->sourceName, "padding");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
