# qtplatz.cmake

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules" )

#####################
# boost setup
#
set( Boost_NO_SYSTEM_PATHS ON )
set( Boost_ADDITIONAL_VERSIONS
  "1.63.0" "1.63"
  "1.62.0" "1.62" )

if( WIN32 )

  find_path( _boost NAMES boost HINTS
    "C:/Boost/include/boost-1_63"   # V14 
    "C:/Boost/include/boost-1_62"   # V14 
    "C:/Boost/include/boost-1_59"   # V13 
    "C:/Boost/include/boost-1_58"   # V12
    "C:/Boost/include/boost-1_57" )

  set( BOOST_ROOT ${_boost} )
  set( BOOST_INCLUDEDIR ${_boost} )
  if ( RTC_ARCH_X64 )
    set( BOOST_LIBRARYDIR "C:/Boost/x86_64/lib" )
  else()
    set( BOOST_LIBRARYDIR "C:/Boost/lib" )    
  endif()
  set( Boost_USE_STATIC_LIBS ON )

else()

  find_path( _boost NAMES include/boost HINTS
    "/usr/local/boost-1_63"        # V14 'libs/serialization/src/basic_archive.cpp library_version_type(14)
    "/usr/local/boost-1_62"        # V14
    "/usr/local/boost-1_59"        # V13
    "/usr/local/boost-1_58"        # V12
    "/usr/local/boost-1_57"
    "/usr/local"
    )

  if ( _boost )
    set( BOOST_ROOT ${_boost} )
  endif()

endif()

