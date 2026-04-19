cmake_minimum_required(VERSION 3.14)

project(libtea VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------
# Library target
# -------------------------
add_library(libtea
    src/linear_regression.cpp
    src/metrics.cpp
)

# include directory for users of the library
target_include_directories(libtea
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# -------------------------
# Example executable
# -------------------------
add_executable(example
    examples/basic_usage.cpp
)

target_link_libraries(example PRIVATE libtea)

# -------------------------
# INSTALL RULES (IMPORTANT)
# -------------------------

include(GNUInstallDirs)

# install library
install(TARGETS libtea
    EXPORT libteaTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# export config for find_package
install(EXPORT libteaTargets
    FILE libteaTargets.cmake
    NAMESPACE libtea::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libtea
)

# -------------------------
# Package config
# -------------------------
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libteaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libteaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libteaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libtea
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libteaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libteaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libtea
)
