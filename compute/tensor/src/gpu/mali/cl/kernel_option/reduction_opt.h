#ifndef REDUCTION_OPT
#define REDUCTION_OPT
#include "common_opt.h"
inline EE set_reduction_opt_mali(bool useNchwFormat,
    bool useOutputC4,
    int axis,
    ReductionMode mode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (useNchwFormat && useOutputC4) {
        CHECK_STATUS(NOT_MATCH)
    }
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    std::string outputC4Name = "";
    if (useOutputC4) {
        formatName = "oc4_";
        CHECK_STATUS(set_chars_define_opt("USE_OUT_C4", opt));
    }
    std::string modeName = "";
    switch (mode) {
        case REDUCTION_SUM:
            modeName = "sum";
            CHECK_STATUS(set_chars_define_opt("USE_SUM", opt));
            CHECK_STATUS(set_chars_define_opt(" TP=sum", opt));
            break;
        case REDUCTION_MEAN:
            modeName = "mean";
            CHECK_STATUS(set_chars_define_opt("USE_MEAN", opt));
            CHECK_STATUS(set_chars_define_opt(" TP=mean", opt));
            break;
        case REDUCTION_STD_DEVIATION:
            modeName = "std_deviation";
            CHECK_STATUS(set_chars_define_opt("USE_STD_DEVIATION", opt));
            CHECK_STATUS(set_chars_define_opt(" TP=std_deviation", opt));
            break;
        case REDUCTION_SCALAR_PRODUCT:
            modeName = "scalar_product";
            CHECK_STATUS(set_chars_define_opt("USE_SCALAR_PRODUCT", opt));
            CHECK_STATUS(set_chars_define_opt(" TP=scalar_product", opt));
            break;
        default:
            return NOT_SUPPORTED;
    }

    std::string kernel =
        std::string("reduction_") + formatName + outputC4Name + modeName + std::to_string(axis);
    UNI_STRCPY(kernelName, kernel.c_str());
    if (useNchwFormat) {
        UNI_STRCPY(kernelOpt->sourceName, "reduction_nchw");
    } else {
        UNI_STRCPY(kernelOpt->sourceName, "reduction");
    }
    CHECK_STATUS(set_value_define_opt(axis, "AXIS", opt));
    return SUCCESS;
}
#endif
