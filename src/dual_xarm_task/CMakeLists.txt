cmake_minimum_required(VERSION 3.8)
project(dual_xarm_task)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
install(DIRECTORY
  launch
  config
  urdf
  srdf
  worlds
  models
  rviz
  ${CMAKE_CURRENT_SOURCE_DIR}/../test_file
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/camera_calibration_check.py
  scripts/validate_charuco_report.py
  scripts/cylinder_pose_estimator.py
  scripts/validate_cylinder_pose_report.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
