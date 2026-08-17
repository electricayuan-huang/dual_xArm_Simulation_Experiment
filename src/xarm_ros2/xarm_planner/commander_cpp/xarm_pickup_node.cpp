#include <signal.h>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

using namespace std::chrono_literals;

static volatile bool g_running = true;
void sig_handler(int) { g_running = false; }

struct Target {
    std::string name;
    double x, y, z;
    double grasp_offset;
    double place_x, place_y;
    bool detected;
    bool has_original = false;
    double original_x = 0.0, original_y = 0.0, original_z = 0.0;
    int64_t last_stamp_ns = 0;
};

class PickupNode
{
public:
    explicit PickupNode(const rclcpp::Node::SharedPtr& node) : node_(node)
    {
        signal(SIGINT, sig_handler);
        signal(SIGTERM, sig_handler);
    }

    bool init()
    {
        auto log = node_->get_logger();

        arm_group_    = node_->get_parameter("arm_group").as_string();
        grip_group_   = node_->get_parameter("grip_group").as_string();
        base_frame_   = node_->get_parameter("base_frame").as_string();
        eef_frame_    = node_->get_parameter("eef_frame").as_string();
        node_->get_parameter_or("planning_eef_link", planning_eef_link_, std::string("link_tcp"));
        pre_z_        = node_->get_parameter("pre_z_offset").as_double();
        lift_z_       = node_->get_parameter("lift_z_offset").as_double();
        grasp_z_      = node_->get_parameter("grasp_z_offset").as_double();
        place_z_      = node_->get_parameter("place_z").as_double();
        grip_close_   = node_->get_parameter("gripper_close").as_double();
        grip_open_    = node_->get_parameter("gripper_open").as_double();
        vel_          = node_->get_parameter("velocity_scale").as_double();
        home_         = node_->get_parameter("home_joints").as_double_array();
        gripper_names_ = node_->get_parameter("gripper_joints").as_string_array();

        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_, true);

        RCLCPP_INFO(log, "Waiting for TF (%s -> %s)...", base_frame_.c_str(), eef_frame_.c_str());
        rclcpp::Rate r(1);
        for (int i = 0; i < 30 && rclcpp::ok(); ++i) {
            try {
                auto t = tf_buffer_->lookupTransform(
                    base_frame_, eef_frame_, tf2::TimePointZero, 1s);
                (void)t; break;
            } catch (...) {}
            r.sleep();
        }

        arm_mg_  = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, arm_group_);
        grip_mg_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, grip_group_);
        arm_mg_->setMaxVelocityScalingFactor(vel_);
        arm_mg_->setMaxAccelerationScalingFactor(vel_);
        arm_mg_->setPlanningTime(30.0);
        arm_mg_->setNumPlanningAttempts(5);
        arm_mg_->setPoseReferenceFrame(base_frame_);
        arm_mg_->setEndEffectorLink(planning_eef_link_);

        RCLCPP_INFO(log, "Pickup node ready.");
        RCLCPP_INFO(log, "  arm: %s  gripper: %s  base: %s  eef: %s",
                    arm_group_.c_str(), grip_group_.c_str(),
                    base_frame_.c_str(), eef_frame_.c_str());
        RCLCPP_INFO(log, "  planning eef: %s", planning_eef_link_.c_str());
        return true;
    }

    void run()
    {
        auto log = node_->get_logger();
        auto targets = getTargets();

        RCLCPP_INFO(log, "\n========== PICK-AND-PLACE START ==========\n");

        RCLCPP_INFO(log, "\n--- Move to observation pose before detection ---");
        if (!moveToObservePose()) {
            RCLCPP_ERROR(log, "Cannot reach observation pose; aborting before detection");
            return;
        }
        wait(1.0);

        RCLCPP_INFO(log, "\n--- Detect objects ---");
        rclcpp::Rate r(2);
        int all_detected = 0;
        for (int retry = 0; retry < 30 && rclcpp::ok() && g_running; ++retry) {
            all_detected = 0;
            for (auto& t : targets) {
                if (!t.detected) {
                    if (detect(t)) all_detected++;
                } else {
                    all_detected++;
                }
            }
            if (all_detected == static_cast<int>(targets.size())) break;
            RCLCPP_INFO(log, "  detected %d/%zu, retrying...", all_detected, targets.size());
            r.sleep();
        }
        if (all_detected < static_cast<int>(targets.size())) {
            RCLCPP_ERROR(log, "Not all objects detected. Check camera view.");
            return;
        }

        for (size_t i = 0; i < targets.size() && g_running; ++i) {
            auto& t = targets[i];
            if (!t.detected) {
                RCLCPP_WARN(log, "  %s not detected, skipping", t.name.c_str());
                continue;
            }
            RCLCPP_INFO(log, "\n--- %zu/%zu: %s ---", i+1, targets.size(), t.name.c_str());
            RCLCPP_INFO(log, "  %s in %s: (%.3f, %.3f, %.3f)",
                        t.name.c_str(), base_frame_.c_str(), t.x, t.y, t.z);

            if (!transfer(t, t.x, t.y, t.z,
                          t.place_x, t.place_y, place_z_)) {
                RCLCPP_ERROR(log, "  %s pick-and-place FAILED", t.name.c_str());
                moveHome();
                break;
            }
            RCLCPP_INFO(log, "  %s done.", t.name.c_str());
        }

        RCLCPP_INFO(log, "\n--- Return placed objects to original positions ---");
        for (auto& t : targets) {
            if (!g_running || !t.has_original) break;
            captureCurrentTarget(t);
            RCLCPP_INFO(log, "  Return %s -> (%.3f, %.3f, %.3f)",
                        t.name.c_str(), t.original_x, t.original_y, t.original_z);
            if (!transfer(t, t.place_x, t.place_y, place_z_,
                          t.original_x, t.original_y, t.original_z)) {
                RCLCPP_ERROR(log, "  Return failed for %s", t.name.c_str());
                moveHome();
                break;
            }
        }

        RCLCPP_INFO(log, "\n--- Return home ---");
        moveHome();
        RCLCPP_INFO(log, "\n========== PICK-AND-PLACE DONE ==========\n");
    }

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_mg_, grip_mg_;

    std::string arm_group_, grip_group_, base_frame_, eef_frame_, planning_eef_link_;
    double pre_z_, lift_z_, grasp_z_, place_z_, grip_close_, grip_open_, vel_;
    std::vector<double> home_;
    std::vector<std::string> gripper_names_;

    std::vector<Target> getTargets()
    {
        return {
            {"obj_green", 0, 0, 0, -0.02, 0.34, 0.30, false},
            {"obj_red",   0, 0, 0, -0.02, 0.22, 0.38, false},
            {"obj_blue",  0, 0, 0, -0.02, 0.22, 0.22, false},
        };
    }

    bool detect(Target& t)
    {
        try {
            auto msg = tf_buffer_->lookupTransform(
                base_frame_, t.name, tf2::TimePointZero, 3s);
            t.x = msg.transform.translation.x;
            t.y = msg.transform.translation.y;
            t.z = msg.transform.translation.z;
            t.last_stamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
            if (!t.has_original) {
                t.original_x = t.x;
                t.original_y = t.y;
                t.original_z = t.z;
                t.has_original = true;
            }
            t.detected = true;
            return true;
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN(node_->get_logger(), "  detect %s failed: %s",
                        t.name.c_str(), e.what());
            t.detected = false;
            return false;
        }
    }

    bool captureCurrentTarget(Target& t)
    {
        try {
            auto msg = tf_buffer_->lookupTransform(
                base_frame_, t.name, tf2::TimePointZero, 2s);
            t.x = msg.transform.translation.x;
            t.y = msg.transform.translation.y;
            t.z = msg.transform.translation.z;
            t.last_stamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
            return true;
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN(node_->get_logger(), "  Cannot refresh %s: %s",
                        t.name.c_str(), e.what());
            return false;
        }
    }

    bool verifyLifted(Target& t, double before_x, double before_y, double before_z,
                      double before_stamp_ns)
    {
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (std::chrono::steady_clock::now() < deadline && rclcpp::ok()) {
            try {
                auto msg = tf_buffer_->lookupTransform(
                    base_frame_, t.name, tf2::TimePointZero, 300ms);
                const auto stamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
                const double dx = msg.transform.translation.x - before_x;
                const double dy = msg.transform.translation.y - before_y;
                const double dz = msg.transform.translation.z - before_z;
                const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (stamp_ns > before_stamp_ns && (distance > 0.03 || dz > 0.03)) {
                    RCLCPP_INFO(node_->get_logger(),
                                "  GRASP VERIFIED: %s moved (%.3f, %.3f, %.3f)",
                                t.name.c_str(), dx, dy, dz);
                    return true;
                }
            } catch (const tf2::TransformException&) {
                // The detector may temporarily lose the object while it is lifted.
            }
            wait(0.2);
        }
        RCLCPP_ERROR(node_->get_logger(),
                     "  GRASP NOT VERIFIED: %s did not move after lift",
                     t.name.c_str());
        return false;
    }

    bool moveJoints(const std::vector<double>& joints)
    {
        arm_mg_->setJointValueTarget(joints);
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (arm_mg_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(node_->get_logger(), "  joint plan FAILED");
            return false;
        }
        return arm_mg_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
    }

    geometry_msgs::msg::Pose makePose(double x, double y, double z,
                                      double roll, double pitch, double yaw)
    {
        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        geometry_msgs::msg::Pose pose;
        pose.position.x = x;
        pose.position.y = y;
        pose.position.z = z;
        pose.orientation = tf2::toMsg(q);
        return pose;
    }

    bool moveToPose(double x, double y, double z, double roll, double pitch, double yaw)
    {
        RCLCPP_INFO(node_->get_logger(), "  MoveTo (%.3f, %.3f, %.3f)", x, y, z);
        arm_mg_->setPoseReferenceFrame(base_frame_);
        arm_mg_->setStartStateToCurrentState();
        arm_mg_->clearPoseTargets();
        auto target = makePose(x, y, z, roll, pitch, yaw);
        arm_mg_->setPoseTarget(target, planning_eef_link_);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (arm_mg_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            return arm_mg_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
        }
        RCLCPP_ERROR(node_->get_logger(), "  MoveTo plan FAILED at (%.3f,%.3f,%.3f)", x, y, z);
        return false;
    }

    bool cartMove(double x, double y, double z)
    {
        RCLCPP_INFO(node_->get_logger(), "  Cart -> (%.3f, %.3f, %.3f)", x, y, z);
        geometry_msgs::msg::Pose cur_pose;
        try {
            auto current_tf = tf_buffer_->lookupTransform(
                base_frame_, planning_eef_link_, tf2::TimePointZero, 1s);
            cur_pose.position.x = current_tf.transform.translation.x;
            cur_pose.position.y = current_tf.transform.translation.y;
            cur_pose.position.z = current_tf.transform.translation.z;
            cur_pose.orientation = current_tf.transform.rotation;
        } catch (const tf2::TransformException& e) {
            RCLCPP_ERROR(node_->get_logger(), "  Cannot read current EEF TF: %s", e.what());
            return false;
        }
        std::vector<geometry_msgs::msg::Pose> waypoints;
        geometry_msgs::msg::Pose wp = cur_pose;
        wp = makePose(x, y, z, M_PI, 0.0, 0.0);
        waypoints.push_back(wp);

        moveit_msgs::msg::RobotTrajectory traj;
        double fraction = arm_mg_->computeCartesianPath(waypoints, 0.005, 1.5, traj);
        RCLCPP_INFO(node_->get_logger(), "  Cart fraction=%.2f", fraction);
        if (fraction >= 0.85) {
            return arm_mg_->execute(traj) == moveit::core::MoveItErrorCode::SUCCESS;
        }

        RCLCPP_WARN(node_->get_logger(), "  cart path partial, fallback to joint-space");
        tf2::Quaternion qcur(cur_pose.orientation.x, cur_pose.orientation.y,
                             cur_pose.orientation.z, cur_pose.orientation.w);
        double cr, cp, cy;
        tf2::Matrix3x3(qcur).getRPY(cr, cp, cy);
        return moveToPose(x, y, z, cr, cp, cy);
    }

    bool moveGripper(double pos)
    {
        std::vector<double> vals(gripper_names_.size(), pos);
        grip_mg_->setJointValueTarget(gripper_names_, vals);
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (grip_mg_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(node_->get_logger(), "  gripper plan FAILED");
            return false;
        }
        return grip_mg_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
    }

    void moveHome()
    {
        RCLCPP_INFO(node_->get_logger(), "  Moving to HOME");
        moveJoints(home_);
    }

    bool moveToObservePose()
    {
        RCLCPP_INFO(node_->get_logger(), "  Moving to configured observe pose");
        std::vector<double> observe_joints;
        if (!node_->get_parameter("observe_joints", observe_joints) || observe_joints.size() != 6) {
            RCLCPP_ERROR(node_->get_logger(), "observe_joints must contain 6 joint values");
            return false;
        }
        if (!moveJoints(observe_joints)) {
            RCLCPP_ERROR(node_->get_logger(), "Observe pose planning/execution failed");
            return false;
        }
        RCLCPP_INFO(node_->get_logger(), "  Observe pose reached");
        return true;
    }

    void wait(double sec)
    {
        rclcpp::Rate r(10);
        for (int i = 0; i < static_cast<int>(sec * 10); ++i) r.sleep();
    }

    bool transfer(Target& t, double source_x, double source_y, double source_z,
                  double destination_x, double destination_y, double destination_z)
    {
        auto log = node_->get_logger();
        double px = source_x, py = source_y, pz = source_z;
        double pre_z  = pz + pre_z_;
        double gz = pz + t.grasp_offset;
        double lift_z = pz + lift_z_;
        const double before_x = px;
        const double before_y = py;
        const double before_z = pz;
        const int64_t before_stamp_ns = t.last_stamp_ns;

        RCLCPP_INFO(log, "  [1/9] Approach pre-grasp (%.3f, %.3f, %.3f)", px, py, pre_z);
        if (!moveToPose(px, py, pre_z, M_PI, 0.0, 0.0)) { return false; }

        RCLCPP_INFO(log, "  [2/9] Open gripper");
        if (!moveGripper(grip_open_)) { return false; }
        wait(0.5);

        RCLCPP_INFO(log, "  [3/9] Descend to grasp (z=%.3f)", gz);
        if (!cartMove(px, py, gz)) {
            moveGripper(grip_open_);
            return false;
        }

        RCLCPP_INFO(log, "  [4/9] Close gripper");
        if (!moveGripper(grip_close_)) { return false; }
        wait(0.5);

        RCLCPP_INFO(log, "  [5/9] Ascend");
        if (!cartMove(px, py, lift_z)) {
            moveGripper(grip_open_);
            return false;
        }
        if (!verifyLifted(t, before_x, before_y, before_z, before_stamp_ns)) {
            moveGripper(grip_open_);
            return false;
        }

        RCLCPP_INFO(log, "  [6/9] Transport to place (%.3f, %.3f)", t.place_x, t.place_y);
        if (!moveToPose(destination_x, destination_y, lift_z, M_PI, 0.0, 0.0)) {
            moveGripper(grip_open_);
            return false;
        }

        RCLCPP_INFO(log, "  [7/9] Descend to place (z=%.3f)", destination_z);
        if (!cartMove(destination_x, destination_y, destination_z)) {
            RCLCPP_WARN(log, "  place descend failed, releasing anyway");
        }

        RCLCPP_INFO(log, "  [8/9] Release");
        if (!moveGripper(grip_open_)) { return false; }
        wait(0.5);

        RCLCPP_INFO(log, "  [9/9] Retreat");
        cartMove(destination_x, destination_y, lift_z);

        return true;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("xarm_pickup_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
    PickupNode pn(node);
    if (!pn.init()) { rclcpp::shutdown(); return 1; }
    pn.run();
    rclcpp::shutdown();
    return 0;
}
