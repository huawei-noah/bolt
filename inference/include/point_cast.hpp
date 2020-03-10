// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _POINT_CAST_H
#define _POINT_CAST_H
#ifdef _USE_MALI
#include "gcl.h"
#endif

class PtrCaster{
public:
    PtrCaster(void* p)
        :ptr(p){}
    inline operator U8*()    {return (U8*)ptr;}
    inline operator void*()  {return ptr;}
#ifdef _USE_FP16
    inline operator F16*()   {return (F16*)ptr;}
#endif
    inline operator F32*()   {return (F32*)ptr;}
    inline operator U32*()   {return (U32*)ptr;}
    inline operator I32*()   {return (I32*)ptr;}
#ifdef _USE_MALI
    inline operator GCLMem_t(){return (GCLMem_t)ptr;}
#endif
private:
    void* ptr;
};

class PtrCasterShared{
    public:
    PtrCasterShared(std::shared_ptr<void> p){ptr = p;}
    inline operator std::shared_ptr<U8>()   {return std::static_pointer_cast<U8>(ptr);}
    inline operator std::shared_ptr<void>() {return std::static_pointer_cast<void>(ptr);}
#ifdef _USE_FP16
    inline operator std::shared_ptr<F16>()  {return std::static_pointer_cast<F16>(ptr);}
#endif
    inline operator std::shared_ptr<F32>()  {return std::static_pointer_cast<F32>(ptr);}
    inline operator std::shared_ptr<U32>()  {return std::static_pointer_cast<U32>(ptr);}
    inline operator std::shared_ptr<I32>()  {return std::static_pointer_cast<I32>(ptr);}
#ifdef _USE_MALI
    inline operator std::shared_ptr<GCLMem>() {return std::static_pointer_cast<GCLMem>(ptr);}
#endif

    private:
    std::shared_ptr<void> ptr;
};
#endif
