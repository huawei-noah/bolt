#ifndef CAST_OPT
#define CAST_OPT
#include "common_opt.h"

inline EE set_cast_opt_mali(bool useNchwFormat,
    DataType idt,
    DataType odt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    kernelOpt->kernelDataType = DT_F16;
    std::string idtName = "f16";
    std::string odtName = "f16";
    if (idt == DT_F16) {
        CHECK_STATUS(set_chars_define_opt("INPUT_F16", opt));
    } else if (idt == DT_I32) {
        CHECK_STATUS(set_chars_define_opt("INPUT_I32", opt));
        idtName = "i32";
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (odt == DT_F16) {
        CHECK_STATUS(set_chars_define_opt("OUTPUT_F16", opt));
    } else if (odt == DT_I32) {
        CHECK_STATUS(set_chars_define_opt("OUTPUT_I32", opt));
        odtName = "i32";
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    sprintf(kernelName, "cast_%s%s_to_%s", formatName.c_str(), idtName.c_str(), odtName.c_str());
    sprintf(kernelOpt->sourceName, "cast");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
