cmake_minimum_required(VERSION 3.8)
project(cr_description)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

install(DIRECTORY urdf config docs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # 守护 bumper 碰撞名与桥接 topic 的接线（需要 xacro + gz；缺环境时自跳过）。
  # 继承自上游 oomwoo-one，见 UPSTREAM.md。
  ament_add_pytest_test(test_bumper_wiring test/test_bumper_wiring.py TIMEOUT 180)
endif()

ament_package()
