cmake_minimum_required(VERSION 3.5)
project(xarm_planner)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic
    -Wno-sign-compare
    -Wno-unused-parameter 
    -Wno-unused-variable
  )
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(xarm_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)

set(dependencies "rclcpp" "std_msgs" "geometry_msgs" "moveit_msgs" "xarm_msgs" "moveit")

include_directories(
  include
  ${ament_INCLUDE_DIRS}
)

add_library(xarm_planner SHARED
  src/xarm_planner.cpp
)
ament_target_dependencies(xarm_planner 
  moveit_ros_planning_interface
  ${dependencies}
)

add_executable(xarm_planner_node src/xarm_planner_node.cpp)
ament_target_dependencies(xarm_planner_node
  ${dependencies}
)
target_link_libraries(xarm_planner_node 
  xarm_planner
  ${ament_LIBRARIES}
)

add_executable(xarm_gripper_planner_node src/xarm_gripper_planner_node.cpp)
ament_target_dependencies(xarm_gripper_planner_node
  ${dependencies}
)
target_link_libraries(xarm_gripper_planner_node 
  xarm_planner
  ${ament_LIBRARIES}
)

add_executable(test_xarm_planner_api_joint test/test_xarm_planner_api_joint.cpp)
ament_target_dependencies(test_xarm_planner_api_joint
  ${dependencies}
)
target_link_libraries(test_xarm_planner_api_joint 
  xarm_planner
  ${ament_LIBRARIES}
)

add_executable(test_dual_xarm_planner_api_joint test/test_dual_xarm_planner_api_joint.cpp)
ament_target_dependencies(test_dual_xarm_planner_api_joint
  ${dependencies}
)
target_link_libraries(test_dual_xarm_planner_api_joint 
  xarm_planner
  ${ament_LIBRARIES}
)

add_executable(test_xarm_planner_api_pose test/test_xarm_planner_api_pose.cpp)
ament_target_dependencies(test_xarm_planner_api_pose
  ${dependencies}
)
target_link_libraries(test_xarm_planner_api_pose 
  xarm_planner
  ${ament_LIBRARIES}
)

add_executable(test_xarm_gripper_planner_api_joint test/test_xarm_gripper_planner_api_joint.cpp)
ament_target_dependencies(test_xarm_gripper_planner_api_joint
  ${dependencies}
)
target_link_libraries(test_xarm_gripper_planner_api_joint 
  xarm_planner
  ${ament_LIBRARIES}
)

add_executable(test_xarm_planner_client_joint test/test_xarm_planner_client_joint.cpp)
ament_target_dependencies(test_xarm_planner_client_joint
  ${dependencies}
)

add_executable(test_xarm_planner_client_pose test/test_xarm_planner_client_pose.cpp)
ament_target_dependencies(test_xarm_planner_client_pose
  ${dependencies}
)

add_executable(test_xarm_gripper_planner_client_joint test/test_xarm_gripper_planner_client_joint.cpp)
ament_target_dependencies(test_xarm_gripper_planner_client_joint
  ${dependencies}
)

add_executable(test_lite_gripper_realmove test/test_lite_gripper_realmove.cpp)
ament_target_dependencies(test_lite_gripper_realmove
  ${dependencies}
)

# Hand-eye calibration data recorder (Phase 2)
add_executable(xarm_handeye_sample_recorder commander_cpp/xarm_handeye_sample_recorder.cpp)
ament_target_dependencies(xarm_handeye_sample_recorder
  rclcpp
  std_srvs
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
)

# Visual pick-and-place task (Phase 6)
add_executable(xarm_pickup_node commander_cpp/xarm_pickup_node.cpp)
ament_target_dependencies(xarm_pickup_node
  rclcpp
  moveit_ros_planning_interface
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS 
  xarm_planner
  xarm_planner_node
  xarm_gripper_planner_node
  test_xarm_planner_api_joint
  test_dual_xarm_planner_api_joint
  test_xarm_planner_api_pose
  test_xarm_gripper_planner_api_joint
  test_xarm_planner_client_joint
  test_xarm_planner_client_pose
  test_xarm_gripper_planner_client_joint
  test_lite_gripper_realmove
  xarm_handeye_sample_recorder  # Phase 2
  xarm_pickup_node  # Phase 6
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  models
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  commander_cpp
  DESTINATION share/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY
  worlds
  DESTINATION share/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.sdf"
)

install(PROGRAMS
  scripts/compute_handeye.py
  commander_py/opencv_detector.py
  commander_py/evaluate_handeye.py
  commander_py/publish_handeye_tf.py
  commander_py/generate_aruco_marker.py
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
