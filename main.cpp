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

#if ARRAYFIRE
# include <arrayfire.h>
# include <af/cuda.h>
#endif
#if OPENCV
# include <opencv2/opencv.hpp>
#endif
#include <boost/format.hpp>
#include "colormap.hpp"
#include <iomanip>

template< typename T > using af_unique_ptr = std::unique_ptr< T, std::function<void(T*)> >;    

int
main(int argc, char * argv[])
{
    (void)argc;
    (void)argv;

    float data[]  = { 0.10, 0.11, 0.12, 0.13, 0.14, 0.15
                      , 0.20, 0.21, 0.22, 0.23, 0.24, 0.25
                      , 0.30, 0.31, 0.32, 0.33, 0.34, 0.35
                      , 0.40, 0.41, 0.42, 0.43, 0.44, 0.45
                      , 0.50, 0.51, 0.52, 0.53, 0.54, 0.55
    };
    constexpr size_t ncols = 6;
    constexpr size_t nrows = 5;

#if 1
    // OpenCV cv::Mat <--> af::arry conversion
    // no transposed version
    {
        // mat is the row,col array
        cv::Mat mat = cv::Mat( 5, ncols, CV_32FC(1), data );
        std::cout << "Source cv::Mat : " << mat << std::endl;
        
        // array is the col,row array
        std::cout << "--------------------------------------------------" << std::endl;
        af::array gray = af::array( mat.cols, mat.rows, 1, mat.ptr< float >( 0 ) );
        af::print( "Converted to af::array gray: ", gray );

        af::array rgb = af::gray2rgb( gray );
        af::print( "rgb: ", rgb );

        // https://groups.google.com/forum/#!topic/arrayfire-users/34_AFiXRnKg
        af::array rgb_t = af::reorder( rgb, 2, 0, 1 ) * 255; // Converts 3rd dimention to be the first dimension
        af::print( "rgb_t: ", rgb_t );

        auto mat2 = cv::Mat( mat.rows, mat.cols, CV_8UC(3) );
        rgb_t.as( u8 ).host( /* reinterpret_cast< void * > */ ( mat2.ptr< uchar >( 0 ) ) );

        std::cout << "Back to cv::Mat :\n" << mat2 << std::endl;
    }
#endif
#if 1
    // transposed version
    {
        cv::Mat mat = cv::Mat( 4, ncols, CV_32FC(1), data );        
        std::cout << "--------------------------------------------------" << std::endl;
        af::array gray = af::array( mat.cols, mat.rows, 1, mat.ptr< float >( 0 ) ).T();
        af::print( "Converted to af::array (transposed) gray: ", gray );

        // rgb as af native format
        af::array rgb = af::gray2rgb( gray );
        af::print( "rgb: ", rgb );

        const int channels = rgb.dims( 2 );
        af::array rgb_t = rgb.T();
        af::print( "rgb_t", rgb_t );

        // https://groups.google.com/forum/#!topic/arrayfire-users/34_AFiXRnKg
        // Converts 3rd dimention to be the first dimension
        auto cv_format_rgb = af::reorder( rgb_t, 2, 0, 1 ); // * 255; 
        af::print( "cv_format_rgb", cv_format_rgb );
        
        auto mat2 = cv::Mat( mat.rows, mat.cols, CV_8UC(3) );
        cv_format_rgb.as( u8 ).host( /* reinterpret_cast< void * > */ ( mat2.ptr< uchar >( 0 ) ) );

        std::cout << "Back to cv::Mat :\n" << mat2 << std::endl;
    }
#endif
#if 1
    // BGR color
    {
        // it seems that af::print does not properly working for u8 data array though f32 works well.
        typedef float bgr_dtype;
        //typedef uint8_t bgr_dtype;
        cv::Mat bgr;
        cv::Mat gray = cv::Mat( 4, ncols, CV_32FC(1), data ) * 100;
        //gray.convertTo( gray, CV_8U );
        cv::cvtColor( gray, bgr, cv::COLOR_GRAY2BGR );
        
        const uint32_t channels = ( ( uint32_t( bgr.type() ) >> 3 ) & 0xf ) + 1;
        std::cout << "bgr(" << bgr.rows << ", " << bgr.cols << ") channels=" << channels << ":\n" << bgr << std::oct << std::endl;

        af::array a = af::array( channels, bgr.cols, bgr.rows, bgr.ptr< bgr_dtype >() );
        af::array abgr = af::reorder( a, 1, 2, 0 );
        //af::print( "a", a );
        af::print( "af::bgr", abgr );
    }
#endif
#if 1
    // CUDA access test using colormap (a.k.a. heat map) color coding kernel
    {
        cv::Mat mat = cv::Mat( 4, ncols, CV_32FC(1), data );
        
        const float __levels [] = { 0.0, 0.2, 0.4, 0.6, 0.8, 0.97, 1.0 };
        //                          black, navy, cyan, green,yellow,red, white
        const float __colors [] = {   0.0,  0.0,  0.0,  0.0,  1.0,  1.0, 1.0      // R
                                    , 0.0,  0.0,  1.0,  1.0,  1.0,  0.0, 1.0      // G
                                    , 0.0,  0.5,  1.0,  0.0,  0.0,  0.0, 1.0 };   // B

        af::array levels = af::array( sizeof(__levels)/sizeof(__levels[0]), 1, __levels );
        af::array colors = af::array( 3, sizeof(__colors)/sizeof(__colors[0]) / 3, __colors );
        // af::print( "colors", colors );
        
        af::array gray = af::array( mat.cols, mat.rows, 1, mat.ptr< float >( 0 ) );

        auto rgb = colorMap( gray, levels, colors );

        af::print( "gray -> rgb", rgb );
    }
#endif
#if 1
    {
        cv::Mat mat = cv::Mat( 4, ncols, CV_32FC(1), data );
        af::array gray = af::array( mat.cols, mat.rows, 1, mat.ptr< float >( 0 ) );
        af::array rgb = af::array( mat.cols, mat.rows, 3 );

        for ( size_t i = 0; i < gray.dims( 0 ); ++i ) {
            for ( size_t j = 0; j < gray.dims( 1 ); ++j ) {
                std::cout << boost::format( "gray(%1%, %2%)= %3%" ) % i % j % gray(i,j).scalar<float>() << std::endl;
                rgb( i, j, 0 ) = gray( i, j ).scalar<float>();
                rgb( i, j, 1 ) = gray( i, j ).scalar<float>();
                rgb( i, j, 2 ) = gray( i, j ).scalar<float>();
            }
        }
        af::print( "rgb", rgb );
    }
#endif
	return 0;
}

