#ifndef PRELU_OPT
#define PRELU_OPT
#include "common_opt.h"
typedef enum {
    RELU_ON_W = 0,
    RELU_ON_H = 1,
    RELU_ON_C = 2,
} ReluAxis;

inline EE set_prelu_opt_mali(bool propagate_down,
    bool useNchwFormat,
    ReluAxis reluAxis,
    DataType dt,
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
    std::string reluAxisName = "";
    std::string progName = "";
    if (propagate_down) {
        progName = "propagate_down";
    } else {
        if (reluAxis == RELU_ON_W) {
            reluAxisName = "w_";
            CHECK_STATUS(set_chars_define_opt("RELU_ON_AXIS_W", opt));
        } else if (reluAxis == RELU_ON_H) {
            reluAxisName = "h_";
            CHECK_STATUS(set_chars_define_opt("RELU_ON_AXIS_H", opt));
        }
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string kernel = std::string("prelu_") + ioMemName + formatName + reluAxisName + progName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "prelu");
    kernelOpt->kernelDataType = dt;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    if (propagate_down) {
        CHECK_STATUS(set_chars_define_opt("USE_PROPAGATE_DOWN", opt));
    }

    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
