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

int
main(int argc, char * argv[])
{
    (void)argc;
    (void)argv;

    float data[]  = { 0.10, 0.11, 0.12, 0.13, 0.14
                      , 0.20, 0.21, 0.22, 0.23, 0.24
                      , 0.30, 0.31, 0.32, 0.33, 0.34
                      , 0.40, 0.41, 0.42, 0.43, 0.44
                      , 0.50, 0.51, 0.52, 0.53, 0.54
    };
#if 1
    // OpenCV cv::Mat <--> af::arry conversion
    // no transposed version
    {
        // mat is the row,col array
        cv::Mat mat = cv::Mat( 4, 5, CV_32FC(1), data );
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
        cv::Mat mat = cv::Mat( 4, 5, CV_32FC(1), data );        
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
#if 0
    // BGR color
    {
        cv::Mat bgr = cv::Mat( 3, 4, CV_8UC(3) );
        int x(1);
        for ( int i = 0; i < bgr.rows; ++i ) {
            for ( int j = 0; j < bgr.cols; ++j ) {
                bgr.at< cv::Vec3b >( i, j )[0] = x;
                bgr.at< cv::Vec3b >( i, j )[1] = x;
                bgr.at< cv::Vec3b >( i, j )[2] = x;
                ++x;
            }
        }
        
        std::cout << "bgr(" << bgr.rows << ", " << bgr.cols << "):\n" << bgr << std::oct << std::endl;
        const uint32_t channels = ( bgr.type() >> 3 ) + 1;

        af::array a = af::array( channels, bgr.rows, bgr.cols, bgr.ptr< unsigned char >( 0 ) );
        af::print( "bgr->a", a );
        af::array b = af::array( bgr.cols, bgr.rows, dim_t( channels ) );

        for ( int i = 0; i < a.dims(0); ++i ) {
            af::print( (boost::format("a[%1%]")%i).str().c_str(), a(i,af::span,af::span) );
            // b(i,af::span,af::span) = a(i.af::span,af::span);
        }

        auto t = af::reorder( a, 2, 1, 0 );
        af::print( "a->t", t );
    }
#endif
#if 1
    // CUDA access test using colormap (a.k.a. heat map) color coding kernel
    {
        cv::Mat mat = cv::Mat( 4, 5, CV_32FC(1), data );
        
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
	return 0;
}

