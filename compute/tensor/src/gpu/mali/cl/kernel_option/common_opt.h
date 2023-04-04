#ifndef COMMON_OPT
#define COMMON_OPT

#include "gcl_kernel_type.h"
#include "error.h"
#include "ocl_context.h"
#include "gcl_func.h"

inline std::string get_kernel_name(std::string prefix, char *opt) {
    std::string ret = prefix;
    if (opt != NULL) {
        std::hash<std::string> h;
        size_t r = h(opt);
        ret += std::string("_") + std::to_string(r);
    }
    return ret;
}

inline EE add_macro(char *&opt, const std::string &name, std::string value = "")
{
    if (opt == NULL) {
        return NULL_POINTER;
    }
    if (name == "" || name == " ") {
        return SUCCESS;
    }
    std::string macro = " ";
    if (value == "") {
        macro += "-D" + name;
    } else {
        macro += "-D" + name + "=" + value;
    }
    macro += " ";
    UNI_STRCPY(opt, macro.c_str());
    opt += macro.length();
    return SUCCESS;
}

inline EE add_macro(char *&opt, const std::string &name, const I32 &value)
{
    return add_macro(opt, name, std::to_string(value));
}

inline EE add_macro(char *&opt, const std::string &name, const U32 &value)
{
    return add_macro(opt, name, std::to_string(value));
}

inline EE add_macro(char *&opt, const std::string &name, const F32 &value)
{
    return add_macro(opt, name, std::to_string(value));
}

inline EE add_macro_type(char *&opt, DataType dt)
{
    std::string type = gcl_get_type(dt);
    const char *p = type.c_str();
    char buffer[256] = {0};
    UNI_SNPRINTF(buffer, 256,
        "-cl-std=CL2.0 -DT=%s -DT2=%s2 -DT3=%s3 -DT4=%s4 -DT8=%s8 -DT16=%s16", p, p, p,
        p, p, p);
    std::string macro = std::string(buffer);
    if (dt == DT_F16) {
        macro += std::string(" -DUSE_HALF");
    }
    macro += " ";
    UNI_STRCPY(opt, macro.c_str());
    opt += macro.length();
    return SUCCESS;
}

inline EE add_macro_format(char *&opt, bool nchw)
{
    EE ret = SUCCESS;
    if (nchw) {
        ret = add_macro(opt, "USE_NCHW");
    }
    return ret;
}

inline EE add_macro_io(
    char *&opt, GCLMemType inputType, GCLMemType outputType)
{
    bool useInputImg = (inputType == GCL_MEM_BUF) ? false : true;
    bool useOutputImg = (outputType == GCL_MEM_BUF) ? false : true;
    if (useInputImg) {
        CHECK_STATUS(add_macro(opt, "USE_INPUT_IMG"));
    }
    if (useOutputImg) {
        CHECK_STATUS(add_macro(opt, "USE_OUTPUT_IMG"));
    }
    return SUCCESS;
}

inline EE set_chars_define_opt(const char *optName, char *&opt)
{
    return add_macro(opt, optName);
}

inline EE set_value_define_opt(U32 val, const char *valName, char *&opt)
{
    return add_macro(opt, valName, val);
}

inline EE set_activation_define_opt(const ActivationParamSpec &p, char *&opt)
{
    std::hash<std::string> h;
    std::string sopt = " ";
    switch (p.mode) {
        case ACTIVATION_NULL:
            sopt = "-D AM= ";
            break;
        case ACTIVATION_RELU: {
            if (p.value[0] == 0) {
                sopt = "-DUSE_RELU -D AM=relu_ ";
            } else {
                sopt = "-DUSE_LEAKY_RELU -D AM=leakyrelu" + std::to_string(h(std::to_string(p.value[0])))+ "_ ";
                CHECK_STATUS(add_macro(opt, "alpha", p.value[0]))
            }
            break;
        }
        case ACTIVATION_RELU6:
            sopt = "-DUSE_RELU6 -D AM=relu6_ ";
            break;
        case ACTIVATION_H_SIGMOID:
            sopt = "-DUSE_HSIGMOID -D AM=hsigmoid_ ";
            break;
        case ACTIVATION_H_SWISH:
            sopt = "-DUSE_HSWISH -D AM=hswish_ ";
            break;
        case ACTIVATION_GELU:
            sopt = "-DUSE_GELU -D AM=gelu_ ";
            break;
        case ACTIVATION_TANH:
            sopt = "-DUSE_TANH -D AM=tanh_ ";
            break;
        case ACTIVATION_SIGMOID:
            sopt = "-DUSE_SIGMOID -D AM=sigmoid_ ";
            break;
        case ACTIVATION_ABS:
            sopt = "-DUSE_ABS -D AM=abs_ ";
            break;
        case ACTIVATION_LOG:
            sopt = "-DUSE_LOG -D AM=log_ ";
            break;
        case ACTIVATION_NEG:
            sopt = "-DUSE_NEG -D AM=neg_ ";
            break;
        case ACTIVATION_EXP:
            sopt = "-DUSE_EXP -D AM=exp_ ";
            break;
        case ACTIVATION_SWISH:
            sopt = "-DUSE_SWISH -D AM=swish_ ";
            break;
        case ACTIVATION_FLOOR:
            sopt = "-DUSE_FLOOR -D AM=floor_ ";
            break;
        case ACTIVATION_CEIL:
            sopt = "-DUSE_CEIL -D AM=ceil_ ";
            break;
        case ACTIVATION_ROUND:
            sopt = "-DUSE_ROUND -D AM=round_ ";
            break;
        default:
            UNI_ERROR_LOG("please add new activation function support in %s.\n", __FUNCTION__);
            break;
    }
    UNI_STRCPY(opt, sopt.c_str());
    opt += sopt.length();
    return SUCCESS;
}

inline EE set_activation_mode_name(const ActivationParamSpec &p, char *name)
{
    std::hash<std::string> h;
    std::string sname = "";
    switch (p.mode) {
        case ACTIVATION_NULL:
            break;
        case ACTIVATION_RELU: {
            if (p.value[0] == 0) {
                sname = "relu_";
            } else {
                sname = "leakyrelu" + std::to_string(h(std::to_string(p.value[0])))+ "_";
            }
            break;
        }
        case ACTIVATION_RELU6:
            sname = "relu6_";
            break;
        case ACTIVATION_H_SIGMOID:
            sname = "hsigmoid_";
            break;
        case ACTIVATION_H_SWISH:
            sname = "hswish_";
            break;
        case ACTIVATION_GELU:
            sname = "gelu_";
            break;
        case ACTIVATION_TANH:
            sname = "tanh_";
            break;
        case ACTIVATION_SIGMOID:
            sname = "sigmoid_";
            break;
        case ACTIVATION_ABS:
            sname = "abs_";
            break;
        case ACTIVATION_LOG:
            sname = "log_";
            break;
        case ACTIVATION_NEG:
            sname = "neg_";
            break;
        case ACTIVATION_EXP:
            sname = "exp_";
            break;
        case ACTIVATION_SWISH:
            sname = "swish_";
            break;
        case ACTIVATION_FLOOR:
            sname = "floor_";
            break;
        case ACTIVATION_CEIL:
            sname = "ceil_";
            break;
        case ACTIVATION_ROUND:
            sname = "round_";
            break;
        default:
            UNI_ERROR_LOG("please add new activation function support in %s.\n", __FUNCTION__);
            break;
    }
    UNI_STRCPY(name, sname.c_str());
    return SUCCESS;
}

inline EE set_eltwise_define_opt(EltwiseMode eltwiseMode, char *&opt)
{
    std::string sopt = " ";
    switch (eltwiseMode) {
        case ELTWISE_MAX:
            sopt = "-DUSE_MAX -D EM=max_ ";
            break;
        case ELTWISE_MIN:
            sopt = "-DUSE_MIN -D EM=min_ ";
            break;
        case ELTWISE_SUM:
            sopt = "-DUSE_SUM -D EM=sum_ ";
            break;
        case ELTWISE_SUB:
            sopt = "-DUSE_SUB -D EM=sub_ ";
            break;
        case ELTWISE_PROD:
            sopt = "-DUSE_PROD -D EM=prod_ ";
            break;
        case ELTWISE_DIV:
            sopt = "-DUSE_DIV -D EM=div_ ";
            break;
        default:
            UNI_ERROR_LOG("please add new eltwise function support in %s.\n", __FUNCTION__);
            break;
    }
    UNI_STRCPY(opt, sopt.c_str());
    opt += sopt.length();
    return SUCCESS;
}

inline EE set_eltwise_mode_name(EltwiseMode eltwiseMode, char *name)
{
    std::string sname = "";
    switch (eltwiseMode) {
        case ELTWISE_MAX:
            sname = "max_";
            break;
        case ELTWISE_MIN:
            sname = "min_";
            break;
        case ELTWISE_SUM:
            sname = "sum_";
            break;
        case ELTWISE_SUB:
            sname = "sub_";
            break;
        case ELTWISE_PROD:
            sname = "prod_";
            break;
        case ELTWISE_DIV:
            sname = "div_";
            break;
        default:
            UNI_ERROR_LOG("please add new eltwise function support in %s.\n", __FUNCTION__);
            break;
    }
    UNI_STRCPY(name, sname.c_str());
    return SUCCESS;
}

inline EE set_io_mem_define_opt(GCLMemType inputType, GCLMemType outputType, char *&opt)
{
    bool useInputImg = (inputType == GCL_MEM_BUF) ? false : true;
    bool useOutputImg = (outputType == GCL_MEM_BUF) ? false : true;
    std::string def = "";
    if (useInputImg) {
        def += "-DUSE_INPUT_IMG ";
    }
    if (useOutputImg) {
        def += "-DUSE_OUTPUT_IMG ";
    }
    if (useInputImg && !useOutputImg) {
        def += "-D IOM=im_ ";
    } else if (!useInputImg && useOutputImg) {
        def += "-D IOM=om_ ";
    } else if (useInputImg && useOutputImg) {
        def += "-D IOM=iom_ ";
    } else {
        def += "-D IOM= ";
    }
    UNI_STRCPY(opt, def.c_str());
    opt += def.length();
    return SUCCESS;
}

inline EE set_io_mem_name(GCLMemType inputType, GCLMemType outputType, char *name)
{
    bool useInputImg = (inputType == GCL_MEM_BUF) ? false : true;
    bool useOutputImg = (outputType == GCL_MEM_BUF) ? false : true;
    std::string sname = "";
    if (useInputImg && !useOutputImg) {
        sname = "im_";
    } else if (!useInputImg && useOutputImg) {
        sname = "om_";
    } else if (useInputImg && useOutputImg) {
        sname = "iom_";
    }
    UNI_STRCPY(name, sname.c_str());
    return SUCCESS;
}

inline EE set_io_mems_name_and_define_opts(GCLMemType *inputMemType,
    GCLMemType *outputMemType,
    U32 inputNum,
    U32 outputNum,
    char *name,
    char *&opt)
{
    std::string im = "";
    std::string om = "";
    std::string iomDef = "";
    bool hasInputImg = false;
    bool hasOutputImg = false;
    for (U32 i = 0; i < inputNum; i++) {
        iomDef = "USE_INPUT_IMG";
        if (inputMemType[i] == GCL_MEM_IMG_3D) {
            hasInputImg = true;
            if (inputNum > 1) {
                im += std::to_string(i);
                if (i > 0) {
                    iomDef += std::to_string(i);
                }
            }
            CHECK_STATUS(set_chars_define_opt(iomDef.c_str(), opt));
        }
    }
    if (hasInputImg) {
        im = "i" + im;
    }

    for (U32 i = 0; i < outputNum; i++) {
        iomDef = "USE_OUTPUT_IMG";
        if (outputMemType[i] == GCL_MEM_IMG_3D) {
            hasOutputImg = true;
            if (outputNum > 1) {
                om += std::to_string(i);
                if (i > 0) {
                    iomDef += std::to_string(i);
                }
            }
            CHECK_STATUS(set_chars_define_opt(iomDef.c_str(), opt));
        }
    }
    if (hasOutputImg) {
        om = "o" + om;
    }

    std::string iom = im + om;
    iomDef = " IOM=";
    if (iom != "") {
        iom += "m_";
        iomDef += iom;
    }
    CHECK_STATUS(set_chars_define_opt(iomDef.c_str(), opt));

    UNI_STRCPY(name, iom.c_str());
    return SUCCESS;
}

inline EE set_common_opt(DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    const char *sourceName,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string kernel = sourceName + std::string("_") + ioMemName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, sourceName);
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline bool check_qualcomm_device(char *devName = nullptr)
{
    bool useQualcommDev = OCLContext::getInstance().handle->useQualcommDev;
    if (devName) {
        std::string dev = "";
        if (useQualcommDev) {
            dev = "_qc";
        }
        UNI_STRCPY(devName, dev.c_str());
    }
    return useQualcommDev;
}

inline EE add_qcom_acc_16_bit_opt(char *&opt)
{
    std::string qcom_acc = "-qcom-accelerate-16-bit ";
    UNI_STRCPY(opt, qcom_acc.c_str());
    opt += qcom_acc.length();
    return SUCCESS;
}
#endif
