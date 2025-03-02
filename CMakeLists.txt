cmake_minimum_required(VERSION 3.26)

project(DiViEn)

# https://stackoverflow.com/questions/18968979/how-to-make-colorized-message-with-cmake
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset  "${Esc}[m")
  set(ColorBold   "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()


set(CMAKE_CXX_STANDARD 23)

# set release mode if no mode was specified
if (CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  message("${BoldGreen}Building in Release mode [LTO enabled]${ColorReset}")

  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(CMAKE_COLOR_MAKEFILE TRUE)
set(CMAKE_COLOR_DIAGNOSTICS ON)

set(ENABLE_IO_URING OFF CACHE BOOL "Use io-uring for tcp on linux (default off)")
set(CONFIG_MSYS_UCRT64 OFF CACHE BOOL "Enable config for building on msys64 with ucrt64")

# TODO: ensure these options are mutually exclusive
option(ENABLE_IO_URING "Use io-uring for tcp on linux (default off)" OFF)
option(CONFIG_MSYS_UCRT64 "Enable config for building on msys64 with ucrt64" OFF)

if(ENABLE_IO_URING)
  add_definitions(-DASIO_HAS_IO_URING=1)
  add_definitions(-DASIO_DISABLE_EPOLL=1)
endif()

if (CONFIG_MSYS_UCRT64)
  # Target Windows 10.
  add_definitions(-DWINVER=0x0A00)
  add_definitions(-D_WIN32_WINNT=0x0A00)
  set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
endif()


# enable sanitizers and libassert in debug mode
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("${BoldRed}Building in Debug mode  (WARNING: ENABLES SANITIZERS AND LIBASSERT -- INCURS EXTRA RUNTIME OVERHEAD)${ColorReset}")

  add_compile_options(-fsanitize=undefined -fsanitize=address -fsanitize=bounds -fsanitize=signed-integer-overflow,null,alignment -fno-sanitize-recover=null -fsanitize-trap=alignment -fno-omit-frame-pointer)
  add_link_options   (-fsanitize=undefined -fsanitize=address -fsanitize=bounds -fsanitize=signed-integer-overflow,null,alignment -fno-sanitize-recover=null -fsanitize-trap=alignment -fno-omit-frame-pointer)

  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-fsanitize=unsigned-integer-overflow -fsanitize=local-bounds)
    add_link_options   (-fsanitize=unsigned-integer-overflow -fsanitize=local-bounds)
  endif()

  include(FetchContent)
  FetchContent_Declare(
    libassert
    GIT_REPOSITORY https://github.com/jeremy-rifkin/libassert.git
    GIT_TAG        v2.0.1 # <HASH or TAG>
  )
  FetchContent_MakeAvailable(libassert)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  find_program(MOLD_LINKER "mold")
  find_program(LLD_LINKER "lld")
  if (MOLD_LINKER AND NOT CONFIG_MSYS_UCRT64)
    add_link_options   (-fuse-ld=mold)
  elseif(LLD_LINKER)
    add_link_options   (-fuse-ld=lld)
  endif()
endif()

if(NOT CONFIG_MSYS_UCRT64)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
      libavdevice
      libavfilter
      libavformat
      libavcodec
      libswresample
      libswscale
      libavutil
  )
endif()

# TODO can we isolate the code so that we turn on exceptions
# for only part of the code? like the part that deals with ASIO.
# I think it's possible.

add_executable(DiViEn main.cpp decode.cpp segment.cpp encode.cpp network.cpp)

target_compile_options(DiViEn PRIVATE -Wall -Wextra -Wformat )
if(ENABLE_IO_URING)
  target_link_libraries( DiViEn uring )
endif()

if(CONFIG_MSYS_UCRT64)
  # target_link_libraries( DiViEn libavdevice libavfilter libavformat libavcodec libswresample libswscale libavutil )
  target_link_libraries( DiViEn avdevice avfilter avformat avcodec swresample swscale avutil )
  target_link_libraries( DiViEn wsock32 ws2_32)
  if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND CONFIG_MSYS_UCRT64)
    # target_link_libraries( DiViEn -lc++ -static -lc++abi -pthread )
    target_link_libraries(DiViEn -Wl,-Bstatic -lc++ -Wl,-Bdynamic)
  else()
    target_link_libraries( DiViEn -lstdc++)
  endif()
else()
  target_link_libraries( DiViEn PkgConfig::LIBAV )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_libraries(DiViEn libassert::assert)
endif()
