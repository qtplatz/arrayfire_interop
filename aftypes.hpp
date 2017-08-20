/**************************************************************************
** Copyright (C) 2010-2017 Toshinobu Hondo, Ph.D.
** Copyright (C) 2013-2017 MS-Cheminformatics LLC, Toin, Mie Japan
*
** Contact: toshi.hondo@qtplatz.com
**
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**************************************************************************/

#pragma once

#include <af/defines.h>
#include <complex>
#include <tuple>
#include <utility>

namespace arrayfire {
    
    typedef std::tuple< float                      // f32
                        , std::complex<float>      // c32
                        , double                   // f64
                        , std::complex<double>     // c64
                        , bool                     // b8
                        , int32_t                  // s32
                        , uint32_t                 // u32
                        , uint8_t                  // u8
                        , int64_t                  // s64
                        , uint64_t                 // u64
#if AF_API_VERSION >= 32                        
                        , int16_t                  // s16
                        , uint16_t                 // u16
#endif
                        > af_array_types;

    template < class... Args > struct type_list {
        template < std::size_t N >
        using type = typename std::tuple_element<N, std::tuple<Args...> >::type;
    };

    template <class T, class Tuple>
    struct type_index;
    
    template <class T, class... Types>
    struct type_index<T, std::tuple<T, Types...> > {
        static const af_dtype /* std::size_t */ value = af_dtype(0);
    };
    
    template <class T, class U, class... Types>
    struct type_index<T, std::tuple<U, Types...> > {
        static const af_dtype /* std::size_t */ value = af_dtype( 1 + type_index<T, std::tuple<Types...> >::value );
    };

    // find index from tuple

    // usage: af_array_type< 0 >::type := 'float'
    template< af_dtype N > using af_type = std::tuple_element< N, af_array_types >;
    
    // usage: std::cout << af_type_value< float >::value << std::endl;
    template< class T > using af_type_value = type_index< T, af_array_types >;
}

