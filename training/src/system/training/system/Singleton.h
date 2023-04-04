// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SINGLETON_H
#define SINGLETON_H

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <list>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#ifdef _MSC_VER
#define LOKI_C_CALLING_CONVENTION_QUALIFIER __cdecl
#else
#define LOKI_C_CALLING_CONVENTION_QUALIFIER
#endif

namespace raul
{
typedef void(LOKI_C_CALLING_CONVENTION_QUALIFIER* atexit_pfn_t)();

/** Auxiliary structure for standard memory allocation in heap
 */
template<class T>
struct CreateUsingNew
{
    static T* Create() { return new T; }

    static void Destroy(T* p) { delete p; }
};

/** Auxiliary structure for memory allocation in static region
 */
template<class T>
struct CreateStatic
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4121)
// alignment of a member was sensitive to packing
#endif // _MSC_VER
    /** Auxiliary union for correct static memory alignment
     */
    union MaxAlign
    {
        char t_[sizeof(T)];
        short int shortInt_;
        int int_;
        long int longInt_;
        float float_;
        double double_;
        long double longDouble_;
        struct Test;
        int Test::*pMember_;
        int (Test::*pMemberFn_)(int);
    };
#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

    static T* Create()
    {
        static MaxAlign staticMemory_;
        return new (&staticMemory_) T;
    }

    static void Destroy(T* p)
    {
        (void)p;
        p->~T();
    }
};

/** Auxiliary structure for default lifetime policy
 */
template<class T>
struct DefaultLifetime
{
    static void ScheduleDestruction(T*, atexit_pfn_t pFun) { std::atexit(pFun); }

    static void OnDeadReference() { throw std::logic_error("Dead Reference Detected"); }
};

/** Auxiliary class for phoenix lifetime policy
 */
template<class T>
class PhoenixSingleton
{
  public:
    static void ScheduleDestruction(T*, atexit_pfn_t pFun)
    {
#ifndef ATEXIT_FIXED
        if (!destroyedOnce_)
        {
#endif
            std::atexit(pFun);
#ifndef ATEXIT_FIXED
        }
#endif
    }

    static void OnDeadReference()
    {
#ifndef ATEXIT_FIXED
        destroyedOnce_ = true;
#endif
    }

  private:
#ifndef ATEXIT_FIXED
    static bool destroyedOnce_;
#endif
};

#ifndef ATEXIT_FIXED
template<class T>
bool PhoenixSingleton<T>::destroyedOnce_ = false;
#endif

/** Auxiliary structure for lifetime policy without destruction
 */
template<class T>
struct NoDestroy
{
    static void ScheduleDestruction(T*, atexit_pfn_t) {}

    static void OnDeadReference() {}
};

/** Singleton class
 */
template<typename T, template<class> class CreationPolicy = CreateUsingNew, template<class> class LifetimePolicy = DefaultLifetime>
class SingletonHolder
{
  public:
    ///  Type of the singleton object
    typedef T ObjectType;

    ///  Returns a reference to singleton object
    static T& Instance();

  private:
    // Helpers
    static void MakeInstance();
    static void LOKI_C_CALLING_CONVENTION_QUALIFIER DestroySingleton();

    // Protection
    SingletonHolder();
    ~SingletonHolder(){}
    // Data
    static T* pInstance_;
    static bool destroyed_;

};

template<class T, template<class> class C, template<class> class L>
T* SingletonHolder<T, C, L>::pInstance_ = nullptr;

template<class T, template<class> class C, template<class> class L>
bool SingletonHolder<T, C, L>::destroyed_ = false;

template<class T, template<class> class CreationPolicy, template<class> class LifetimePolicy>
inline T& SingletonHolder<T, CreationPolicy, LifetimePolicy>::Instance()
{
    if (!pInstance_)
    {
        MakeInstance();
    }
    return *pInstance_;
}

template<class T, template<class> class CreationPolicy, template<class> class LifetimePolicy>
void SingletonHolder<T, CreationPolicy, LifetimePolicy>::MakeInstance()
{

    if (!pInstance_)
    {
        if (destroyed_)
        {
            destroyed_ = false;
            LifetimePolicy<T>::OnDeadReference();
        }
        pInstance_ = CreationPolicy<T>::Create();
        LifetimePolicy<T>::ScheduleDestruction(pInstance_, &DestroySingleton);
    }
}

template<class T, template<class> class CreationPolicy, template<class> class L>
void LOKI_C_CALLING_CONVENTION_QUALIFIER SingletonHolder<T, CreationPolicy, L>::DestroySingleton()
{
    
    if(destroyed_) {exit(0);};

    CreationPolicy<T>::Destroy(pInstance_);
    pInstance_ = 0;
    destroyed_ = true;
}
} // namespace raul
#endif // SINGLETON_H
