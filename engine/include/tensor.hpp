// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _TENSOR_H
#define _TENSOR_H

#include <type.h>
#include <tensor_desc.h>
#include <math.h>
#include <memory>
#include <vector>
#include <iostream>


#define Vec std::vector

class Tensor {
    public:
        Tensor() {

        }

        /**
         * @param desc
         */
        explicit Tensor(TensorDesc desc) {
            this->desc = desc;
        }

        /**
         * @param desc
         * @param v
         */
        Tensor(TensorDesc d, std::shared_ptr<U8> v) {
            this->desc = d;
            this->val = v;
        }

        /**
         * @param desc
         * @param v
         * @param s
         */
        Tensor(TensorDesc d, std::shared_ptr<U8> v, std::shared_ptr<F16> s) {
            this->desc = d;
            this->val = v;
            this->scalePtr = s;
        }

        /**
         * @param newDesc
         */
        void resize(TensorDesc newDesc) {
            this->desc = newDesc;
        }

        void alloc() {
            U32 size = tensorNumElements(this->desc);
            this->val = std::shared_ptr<U8>((U8*)operator new(size*bytesOf(desc.dt)));
            this->scalePtr = std::shared_ptr<F16>((F16*)operator new(bytesOf(DT_F16)));
        }

        void set_desc(TensorDesc d) {
            this->desc = d;
        }

        TensorDesc get_desc(){
            return this->desc;
        };

        void set_val(std::shared_ptr<U8> v) {
            this->val = v;
        }

        std::shared_ptr<U8> get_val() 
        {
            return this->val;
        }

        void set_scale(F16 s)
        {
            *(this->scalePtr.get()) = s;
        }

        F16 get_scale()
        {
            return *(this->scalePtr.get());
        }


        template<typename T>
        bool isInvalid() {
            T* data = (T*) this->val.get();
            U32 num = tensorNumElements(this->desc);
            for(U32 i = 0; i < num; i++) {
                if(isnan(data[i]) || isinf(data[i]))
                    return true;
            }
            return false;
        }

        template<typename T>
        void print() {
            T* data = (T*) this->val.get();
            U32 num = tensorNumElements(this->desc);
            for(U32 i = 0; i < num; i++) {
                std::cout << data[i] << " ";
            }
            std::cout << std::endl;
        }


    public:
        TensorDesc desc;
        std::shared_ptr<U8> val;
        std::shared_ptr<F16> scalePtr;
};

#endif //_TENSOR_H
