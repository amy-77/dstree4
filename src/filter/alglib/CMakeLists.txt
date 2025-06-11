   cmake_minimum_required(VERSION 3.0)
   project(alglib)
   
   # 收集所有ALGLIB源文件
   file(GLOB ALGLIB_SOURCES "*.cpp")
   
   # 创建静态库
   add_library(alglib STATIC ${ALGLIB_SOURCES})
   
   # 设置包含目录
   target_include_directories(alglib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
