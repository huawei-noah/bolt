// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



#ifndef GCL_KERNELMAP
#define GCL_KERNELMAP

#include"gcl_common.h"
#include<unordered_map>
#include<mutex>
typedef GCLKernelBin kernelBin;

class gcl_kernel_binmap{
    public:
    gcl_kernel_binmap(){}
    std::unordered_map<std::string, kernelBin>& binMap() {return binMap_;}
    EE put(std::string kernelname, kernelBin kernelbin) {
            std::lock_guard<std::mutex> lock(mtx_);
            auto it = binMap_.find(kernelname);
            if(it == binMap_.end()) binMap_.insert({kernelname, kernelbin});
            return SUCCESS;
    }
    EE get(std::string kernelname, kernelBin** kernelbin_ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = binMap_.find(kernelname);
        if(it == binMap_.end()){
                printf("the kernel %s doesn't exist in binMap\n", kernelname.c_str());
                return NULL_POINTER;
        }
        *kernelbin_ptr = &it->second;
        return SUCCESS;
    }
    private:
    std::unordered_map<std::string, kernelBin> binMap_;
    std::mutex mtx_;
};


class gcl_kernel_binmap_container{
    public:
    static gcl_kernel_binmap_container* instance(){
        static gcl_kernel_binmap_container sInst;
        return &sInst;
    }
    EE put(std::string kernel_binmap_name, std::unique_ptr<gcl_kernel_binmap> kernel_binmap) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = kernel_binmap_container_.find(kernel_binmap_name);
        if(it == kernel_binmap_container_.end()) kernel_binmap_container_.insert(std::make_pair(kernel_binmap_name, std::move(kernel_binmap)));
        return SUCCESS;
    }
    EE get(std::string kernel_binmap_name, gcl_kernel_binmap** kernel_binmap_ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = kernel_binmap_container_.find(kernel_binmap_name);
        if(it == kernel_binmap_container_.end()){
                printf("the kernel_binmap %s doesn't exist in kernel_binmap container\n", kernel_binmap_name.c_str());
                return NULL_POINTER;
        }
        *kernel_binmap_ptr = it->second.get();
        return SUCCESS;
    }
    private:
    gcl_kernel_binmap_container(){}
    std::unordered_map<std::string, std::unique_ptr<gcl_kernel_binmap>> kernel_binmap_container_;
    std::mutex mtx_;
};

class gcl_kernel_binmap_factory{
    public:
    static gcl_kernel_binmap_factory* instance() {
        static gcl_kernel_binmap_factory sInst;
        return &sInst;
    }
    typedef gcl_kernel_binmap* (*PFN_GCLKERNELMAP_CREATOR)();
    EE register_gcl_kernel_binmap(const std::string& kernel_binmap_name, PFN_GCLKERNELMAP_CREATOR pfnCreator) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = creators_.find(kernel_binmap_name);
        if(it == creators_.end()) creators_.insert({kernel_binmap_name, pfnCreator});
        return SUCCESS;
    }
    EE create_gcl_kernel_binmap(const std::string& kernel_binmap_name) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = creators_.find(kernel_binmap_name);
        if(it == creators_.end()){
                printf("the kernel_binmap creator %s doesn't exist in kernel_binmap factory\n", kernel_binmap_name.c_str());
                return NULL_POINTER;
        }
        PFN_GCLKERNELMAP_CREATOR pfn = it->second;
        gcl_kernel_binmap_container::instance()->put(kernel_binmap_name, std::unique_ptr<gcl_kernel_binmap>(pfn()));
        return SUCCESS;
    }
    private:
    gcl_kernel_binmap_factory(){}
    std::unordered_map<std::string, PFN_GCLKERNELMAP_CREATOR> creators_;
    std::mutex mtx_;
};

#define REGISTER_GCLKERNELMAP_CREATOR_IMPL(kernel_binmap_name)\
    namespace{\
        static gcl_kernel_binmap* kernel_binmap_name ## _gcl_kernel_binmap_pfn() {return new kernel_binmap_name();}\
        class kernel_binmap_name ## _gcl_kernel_binmap_loader{\
            public:\
                kernel_binmap_name ## _gcl_kernel_binmap_loader() {\
                    gcl_kernel_binmap_factory::instance()->register_gcl_kernel_binmap(#kernel_binmap_name, kernel_binmap_name ## _gcl_kernel_binmap_pfn);\
                }\
        };\
        static kernel_binmap_name ## _gcl_kernel_binmap_loader kernel_binmap_name ## _sLoader;\
    } 

#define REGISTER_GCLKERNELMAP(kernel_binmap_name) REGISTER_GCLKERNELMAP_CREATOR_IMPL(kernel_binmap_name)
#endif
