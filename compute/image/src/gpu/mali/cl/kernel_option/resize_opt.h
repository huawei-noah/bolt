#ifndef RESIZE_OPT
#define RESIZE_OPT
#include "common_opt.h"
inline EE set_resize_nearest_opt_mali(ResizeParamSpec p,
    bool useNchwFormat,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string modeName = "";
    switch (p.trans_mode) {
        case HALF_PIXEL: {
            modeName = "_half_pixel";
            CHECK_STATUS(set_chars_define_opt("USE_HALF_PIXEL", opt));
            break;
        }
        case PYTORCH_HALF_PIXEL: {
            modeName = "_pytorch_half_pixel";
            CHECK_STATUS(set_chars_define_opt("USE_PYTORCH_HALF_PIXEL", opt));
            break;
        }
        case ALIGN_CORNERS: {
            modeName = "_align_corners";
            CHECK_STATUS(set_chars_define_opt("USE_ALIGN_CORNERS", opt));
            break;
        }
        case ASYMMETRIC: {
            modeName = "_asymmetric";
            CHECK_STATUS(set_chars_define_opt("USE_ASYMMETRIC", opt));
            break;
        }
        default:
            CHECK_STATUS(NOT_SUPPORTED);
    }
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw";
    }
    sprintf(kernelName, "resize_nearest_%s%s%s", ioMemName, formatName.c_str(), modeName.c_str());
    sprintf(kernelOpt->sourceName, "resize_nearest");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline EE set_resize_bilinear_opt_mali(bool useNchwFormat,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw";
    }
    sprintf(kernelName, "resize_bilinear_%s%s", ioMemName, formatName.c_str());
    sprintf(kernelOpt->sourceName, "resize_bilinear");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
