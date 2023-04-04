// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GCL_KERNELBIN_MAP
#define GCL_KERNELBIN_MAP

#include <unordered_map>
#include <mutex>
#include <memory>

#ifdef _WIN32
#define _USE_SEPERATE_KERNELBIN
#endif

#ifdef _USE_SEPERATE_KERNELBIN
#include "file.h"
#endif

struct GCLKernelBin {
    const unsigned char *data;
    unsigned int len;
};

typedef GCLKernelBin KernelBin;

class gcl_kernel_binmap {
public:
    gcl_kernel_binmap()
    {
#ifdef _USE_SEPERATE_KERNELBIN
        this->binary_content = NULL;
        this->binary_len = 0;
        this->hardware_name = "";
#endif
    }

    ~gcl_kernel_binmap()
    {
#ifdef _USE_SEPERATE_KERNELBIN
        if (this->binary_content != NULL) {
            free(this->binary_content);
            this->binary_content = NULL;
        }
        this->binary_len = 0;
#endif
    }

    std::unordered_map<std::string, KernelBin> &binMap()
    {
        return binMap_;
    }

    void put(std::string kernelname, KernelBin kernelbin)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = binMap_.find(kernelname);
        if (it == binMap_.end()) {
            binMap_.insert({kernelname, kernelbin});
        }
    }

    bool get(std::string kernelname, KernelBin **kernelbin_ptr)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = binMap_.find(kernelname);
        if (it == binMap_.end()) {
#ifdef _DEBUG
            printf("kernel bin map size:%d.\n", (int)binMap_.size());
            for (auto iter : binMap_) {
                printf("    %s\n", iter.first.c_str());
            }
#endif
            return false;
        }
        *kernelbin_ptr = &it->second;
        return true;
    }

#ifdef _USE_SEPERATE_KERNELBIN
    void insert(std::string kernelname, KernelBin kernelbin)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        binMap_.insert({kernelname, kernelbin});
    }

    std::string getHardwareName()
    {
        return hardware_name;
    }

    void setHardwareName(std::string name)
    {
        hardware_name = name;
    }

    virtual void loadKernelBin()
    {
        UNI_DEBUG_LOG("load kernel.bin...\n");
        std::lock_guard<std::mutex> lock(mtx_);
        EE ret =
            load_binary(this->binary_file, (void **)&(this->binary_content), &(this->binary_len));
        if (ret != SUCCESS) {
            UNI_WARNING_LOG("can not load kernelbin from %s.\n", this->binary_file);
            return;
        }
        auto p = this->binary_content;
        uint32_t len = 0, num = 0;
        GCLKernelBin item;
        UNI_MEMCPY(&len, p, sizeof(uint32_t));
        p += sizeof(uint32_t);
        this->hardware_name = std::string(p, p + len);
        p += len;

        UNI_MEMCPY(&num, p, sizeof(uint32_t));
        p += sizeof(uint32_t);
        for (uint32_t i = 0; i < num; i++) {
            UNI_MEMCPY(&len, p, sizeof(uint32_t));
            p += sizeof(uint32_t);
            std::string kernelname(p, p + len);
            p += len;

            UNI_MEMCPY(&len, p, sizeof(uint32_t));
            item.len = len;
            p += sizeof(uint32_t);
            item.data = p;
            p += len;
            this->binMap_[kernelname] = item;
        }
        UNI_DEBUG_LOG("load kernel.bin end.\n");
    }

    virtual void saveKernelBin()
    {
        UNI_DEBUG_LOG("save kernel.bin...\n");
        std::lock_guard<std::mutex> lock(mtx_);
        size_t length = sizeof(uint32_t) + this->hardware_name.length() + sizeof(uint32_t);
        for (auto iter : this->binMap_) {
            length += sizeof(uint32_t) + iter.first.length() + sizeof(uint32_t) + iter.second.len;
        }
        unsigned char *content = (unsigned char *)malloc(length);
        auto p = content;
        uint32_t len = this->hardware_name.length();
        UNI_MEMCPY(p, &len, sizeof(uint32_t));
        p += sizeof(uint32_t);
        UNI_MEMCPY(p, this->hardware_name.c_str(), len);
        p += len;

        len = this->binMap_.size();
        UNI_MEMCPY(p, &len, sizeof(uint32_t));
        p += sizeof(uint32_t);
        for (auto iter : this->binMap_) {
            len = iter.first.length();
            UNI_MEMCPY(p, &len, sizeof(uint32_t));
            p += sizeof(uint32_t);
            UNI_MEMCPY(p, iter.first.c_str(), len);
            p += len;

            len = iter.second.len;
            UNI_MEMCPY(p, &len, sizeof(uint32_t));
            p += sizeof(uint32_t);
            UNI_MEMCPY(p, iter.second.data, len);
            p += len;
        }
        EE ret = save_binary(this->binary_file, content, length);
        if (ret != SUCCESS) {
            UNI_ERROR_LOG("can not write kernelbin to %s.\n", this->binary_file);
        }
        if (content != NULL) {
            free(content);
        }
        UNI_DEBUG_LOG("save kernel.bin end.\n");
    }
#endif

private:
    std::unordered_map<std::string, KernelBin> binMap_;
    std::mutex mtx_;
#ifdef _USE_SEPERATE_KERNELBIN
    const char *binary_file = "./kernel.bin";
    unsigned char *binary_content = NULL;
    size_t binary_len = 0;
    std::string hardware_name = "";
#endif
};

class gcl_kernel_binmap_container {
public:
    static gcl_kernel_binmap_container *instance()
    {
        static gcl_kernel_binmap_container sInst;
        return &sInst;
    }
    void put(std::string kernel_binmap_name, std::unique_ptr<gcl_kernel_binmap> kernel_binmap)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = kernel_binmap_container_.find(kernel_binmap_name);
        if (it == kernel_binmap_container_.end()) {
            kernel_binmap_container_.insert(
                std::make_pair(kernel_binmap_name, std::move(kernel_binmap)));
        }
    }
    bool get(std::string kernel_binmap_name, gcl_kernel_binmap **kernel_binmap_ptr)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = kernel_binmap_container_.find(kernel_binmap_name);
        if (it == kernel_binmap_container_.end()) {
            return false;
        }
        *kernel_binmap_ptr = it->second.get();
        return true;
    }

private:
    gcl_kernel_binmap_container()
    {}
    ~gcl_kernel_binmap_container()
    {}
    std::unordered_map<std::string, std::unique_ptr<gcl_kernel_binmap>> kernel_binmap_container_;
    std::mutex mtx_;
};

class gcl_kernel_binmap_factory {
public:
    static gcl_kernel_binmap_factory *instance()
    {
        static gcl_kernel_binmap_factory sInst;
        return &sInst;
    }
    ~gcl_kernel_binmap_factory()
    {}
    typedef gcl_kernel_binmap *(*PFN_GCLKERNELMAP_CREATOR)();
    void register_gcl_kernel_binmap(
        const std::string &kernel_binmap_name, PFN_GCLKERNELMAP_CREATOR pfnCreator)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = creators_.find(kernel_binmap_name);
        if (it == creators_.end()) {
            creators_.insert({kernel_binmap_name, pfnCreator});
        }
    }
    bool create_gcl_kernel_binmap(const std::string &kernel_binmap_name)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = creators_.find(kernel_binmap_name);
        if (it == creators_.end()) {
            printf("the kernel_binmap creator %s doesn't exist in kernel_binmap factory\n",
                kernel_binmap_name.c_str());
            return false;
        }
        PFN_GCLKERNELMAP_CREATOR pfn = it->second;
        gcl_kernel_binmap_container::instance()->put(
            kernel_binmap_name, std::unique_ptr<gcl_kernel_binmap>(pfn()));
        return true;
    }

private:
    gcl_kernel_binmap_factory()
    {}
    std::unordered_map<std::string, PFN_GCLKERNELMAP_CREATOR> creators_;
    std::mutex mtx_;
};

#define REGISTER_GCLKERNELMAP_CREATOR_IMPL(kernel_binmap_name)                         \
    namespace {                                                                        \
    static gcl_kernel_binmap *kernel_binmap_name##_gcl_kernel_binmap_pfn()             \
    {                                                                                  \
        return new kernel_binmap_name();                                               \
    }                                                                                  \
    class kernel_binmap_name##_gcl_kernel_binmap_loader {                              \
    public:                                                                            \
        kernel_binmap_name##_gcl_kernel_binmap_loader()                                \
        {                                                                              \
            gcl_kernel_binmap_factory::instance()->register_gcl_kernel_binmap(         \
                #kernel_binmap_name, kernel_binmap_name##_gcl_kernel_binmap_pfn);      \
        }                                                                              \
    };                                                                                 \
    static kernel_binmap_name##_gcl_kernel_binmap_loader kernel_binmap_name##_sLoader; \
    }

#define REGISTER_GCLKERNELMAP(kernel_binmap_name) \
    REGISTER_GCLKERNELMAP_CREATOR_IMPL(kernel_binmap_name)
#endif
