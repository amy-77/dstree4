cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET filter)
add_library(${TARGET} SHARED filter.cc conformal.cc)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
#target_compile_options(${TARGET} PUBLIC -march=native -msse3 -msse4.2 -mavx -mavx2)

# 添加Eigen头文件路径
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/third_party/eigen)

# 移除dlib相关配置，因为代码中没有使用dlib
# include_directories(${CMAKE_SOURCE_DIR}/third_party)

# 移除dlib相关宏定义
# add_definitions(-DUSE_DLIB_GAM)
# add_definitions(-DDLIB_NO_GUI_SUPPORT)
# add_definitions(-DDLIB_USE_BLAS)
# add_definitions(-DDLIB_USE_LAPACK)
# add_definitions(-DDLIB_NO_GUI_SUPPORT)
# add_definitions(-DDLIB_USE_CUDA=0)

target_link_libraries(${TARGET}
        PRIVATE model util common GSL::gsl GSL::gslcblas ${TORCH_LIBRARIES}
        )

# 添加ALGLIB子目录
add_subdirectory(alglib)

# 将alglib库链接到filter目标（假设您的主要目标名称是"filter"）
target_link_libraries(${TARGET} PRIVATE alglib)
