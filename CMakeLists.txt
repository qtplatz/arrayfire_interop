cmake_minimum_required (VERSION 3.9)
cmake_policy( SET CMP0020 NEW )
cmake_policy( SET CMP0042 NEW )

project ( af_test )

include ( boost.cmake )

set( CMAKE_CXX_STANDARD 14 )

find_package( Boost REQUIRED system )
find_package( OpenCV 3 REQUIRED core imgproc video highgui features2d )
find_package( ArrayFire REQUIRED )
find_package( CUDA REQUIRED )

if ( OpenCV_FOUND )
  add_definitions ( -DOPENCV=1 )
  include_directories( ${OpenCV_INCLUDE_DIRS} )
endif()

if ( CUDA_FOUND )
  list( APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-DVERBOSE")
  set ( CUDA_PROPAGATE_HOST_FLAGS OFF )
  include_directories (
    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
    ${CUDA_INCLUDE_DIRS}
  )
endif()

if ( ArrayFire_FOUND )
  add_definitions ( -DARRAYFIRE=1 )
  include_directories ( ${ArrayFire_INCLUDE_DIRS} )
endif()

if ( Boost_FOUND )
  include_directories( ${Boost_INCLUDE_DIRS} )
endif()

cuda_add_executable( ${PROJECT_NAME}
  main.cpp
  colormap.cu
  )

target_link_libraries( ${PROJECT_NAME} ${CUDA_LINK_LIBRARIES_KEYWORD}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ArrayFire_LIBRARIES}
  ${ArrayFire_CUDA_LIBRARIES}  
  ${CUDA_LIBRARIES}
  )
