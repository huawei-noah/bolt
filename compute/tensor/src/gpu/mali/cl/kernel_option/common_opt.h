#ifndef COMMON_OPT
#define COMMON_OPT
#include "gcl_kernel_type.h"
#include "error.h"

inline EE set_chars_define_opt(const char *optName, char *opt, U32 *optLen)
{
    std::string sopt = "-D";
    std::string sname = optName;
    sopt += optName;
    sopt += " ";
    strcpy(opt, sopt.c_str());
    *optLen = sopt.length();
    return SUCCESS;
}

inline EE set_value_define_opt(U32 val, const char *valName, char *opt, U32 *optLen)
{
    std::string sopt = "-D ";
    std::string sname = valName;
    std::string sval = std::to_string(val);
    sopt += sname;
    sopt += "=";
    sopt += sval;
    sopt += " ";
    strcpy(opt, sopt.c_str());
    *optLen = sopt.length();
    return SUCCESS;
}

inline EE set_activation_define_opt(ActivationMode activeMode, char *opt, U32 *optLen)
{
    std::string sopt = " ";
    std::string name = "";
    switch (activeMode) {
        case ACTIVATION_NULL:
            sopt = "-D AM= ";
            break;
        case ACTIVATION_RELU:
            sopt = "-DUSE_RELU -D AM=relu_ ";
            break;
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
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    strcpy(opt, sopt.c_str());
    *optLen = sopt.length();
    return SUCCESS;
}

inline EE set_activation_modeName(ActivationMode activeMode, char *name, U32 *nameLen)
{
    std::string sname = "";
    switch (activeMode) {
        case ACTIVATION_NULL:
            break;
        case ACTIVATION_RELU:
            sname = "relu_";
            break;
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
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    strcpy(name, sname.c_str());
    *nameLen = sname.length();
    return SUCCESS;
}

inline EE set_eltwise_define_opt(EltwiseMode eltwiseMode, char *opt, U32 *optLen)
{
    std::string sopt = " ";
    std::string name = "";
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
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    strcpy(opt, sopt.c_str());
    *optLen = sopt.length();
    return SUCCESS;
}

inline EE set_eltwise_modeName(EltwiseMode eltwiseMode, char *name, U32 *nameLen)
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
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    strcpy(name, sname.c_str());
    *nameLen = sname.length();
    return SUCCESS;
}
#endif
