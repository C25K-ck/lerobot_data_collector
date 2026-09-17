cmake_minimum_required(VERSION 3.8)
project(zs_dog_genisom)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

if(DEFINED ENV{GENISOM_SDK} AND NOT "$ENV{GENISOM_SDK}" STREQUAL "")
  set(GENISOM_SDK_ROOT "$ENV{GENISOM_SDK}")
else()
  set(GENISOM_SDK_ROOT "/home/ck/Desktop/genisom_robot_sdk-main")
endif()

set(GENISOM_LIB_DIR "${GENISOM_SDK_ROOT}/lib/${CMAKE_SYSTEM_PROCESSOR}")
if(NOT EXISTS "${GENISOM_SDK_ROOT}/include/robot_sdk/sdk_client.hpp")
  message(FATAL_ERROR "找不到 Genisom Robot SDK: ${GENISOM_SDK_ROOT}")
endif()
if(NOT EXISTS "${GENISOM_LIB_DIR}/librobot_sdk.so")
  message(FATAL_ERROR "找不到 librobot_sdk.so: ${GENISOM_LIB_DIR}")
endif()

add_executable(zs_dog_bridge_node src/zs_dog_bridge_node.cpp)
target_include_directories(zs_dog_bridge_node PRIVATE
  ${GENISOM_SDK_ROOT}/include
)
target_link_directories(zs_dog_bridge_node PRIVATE ${GENISOM_LIB_DIR})
target_link_libraries(zs_dog_bridge_node robot_sdk)
ament_target_dependencies(zs_dog_bridge_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
)
set_target_properties(zs_dog_bridge_node PROPERTIES
  BUILD_RPATH "${GENISOM_LIB_DIR}"
  INSTALL_RPATH "${GENISOM_LIB_DIR}"
)

install(TARGETS zs_dog_bridge_node DESTINATION lib/${PROJECT_NAME})

ament_package()
