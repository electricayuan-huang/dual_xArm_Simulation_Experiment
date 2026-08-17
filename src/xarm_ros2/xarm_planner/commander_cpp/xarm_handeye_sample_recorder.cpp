#include <signal.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <std_srvs/srv/trigger.hpp>

using namespace std::chrono_literals;

static volatile bool g_running = true;
void sig_handler(int) { g_running = false; }

class HandEyeSampleRecorder
{
public:
    explicit HandEyeSampleRecorder(const rclcpp::Node::SharedPtr& node)
        : node_(node)
    {
        signal(SIGINT, sig_handler);
        signal(SIGTERM, sig_handler);
    }

    bool initialize()
    {
        auto log = node_->get_logger();

        base_frame_    = node_->get_parameter("base_frame").as_string();
        eef_frame_     = node_->get_parameter("eef_frame").as_string();
        camera_frame_  = node_->get_parameter("camera_frame").as_string();
        marker_frame_  = node_->get_parameter("marker_frame").as_string();
        filepath_      = node_->get_parameter("filepath").as_string();
        node_->get_parameter_or("max_detection_age", max_detection_age_, 0.5);

        tf_buffer_     = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
        tf_listener_   = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        bool use_sim = false;
        node_->get_parameter_or("use_sim", use_sim, false);
        if (use_sim)
        {
            RCLCPP_ERROR(log,
                         "use_sim=true is disabled for hand-eye recording. "
                         "The marker pose must come from image detection TF.");
            return false;
        }

        RCLCPP_INFO(log, "\n==============================================");
        RCLCPP_INFO(log, "  HAND-EYE SAMPLE RECORDER");
        RCLCPP_INFO(log, "==============================================");
        RCLCPP_INFO(log, "  robot tf   : %s -> %s", base_frame_.c_str(), eef_frame_.c_str());
        RCLCPP_INFO(log, "  camera tf  : %s -> %s", camera_frame_.c_str(), marker_frame_.c_str());
        RCLCPP_INFO(log, "  mode       : IMAGE_DETECTION");
        RCLCPP_INFO(log, "  max TF age : %.2f s", max_detection_age_);
        RCLCPP_INFO(log, "  output     : %s", filepath_.c_str());
        RCLCPP_INFO(log, "\n  Commands:");
        RCLCPP_INFO(log, "    record  : ros2 service call /record_sample std_srvs/srv/Trigger");
        RCLCPP_INFO(log, "    save    : ros2 service call /save_samples std_srvs/srv/Trigger");
        RCLCPP_INFO(log, "    delete  : ros2 service call /delete_last_sample std_srvs/srv/Trigger");
        RCLCPP_INFO(log, "    count   : ros2 service call /get_sample_count std_srvs/srv/Trigger");
        RCLCPP_INFO(log, "==============================================\n");

        record_srv_ = node_->create_service<std_srvs::srv::Trigger>(
            "/record_sample",
            [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr res) {
                auto result = recordSample();
                res->success = result.first;
                res->message = result.second;
            });

        save_srv_ = node_->create_service<std_srvs::srv::Trigger>(
            "/save_samples",
            [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr res) {
                auto result = saveSamples();
                res->success = result.first;
                res->message = result.second;
            });

        delete_srv_ = node_->create_service<std_srvs::srv::Trigger>(
            "/delete_last_sample",
            [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr res) {
                if (samples_.empty()) {
                    res->success = false;
                    res->message = "No samples to delete";
                } else {
                    samples_.pop_back();
                    res->success = true;
                    res->message = "Deleted, remaining: " + std::to_string(samples_.size());
                }
            });

        count_srv_ = node_->create_service<std_srvs::srv::Trigger>(
            "/get_sample_count",
            [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr res) {
                res->success = true;
                res->message = std::to_string(samples_.size()) + " samples recorded";
            });

        RCLCPP_INFO(log, "Waiting for robot TF...");
        rclcpp::Rate rate(2);
        bool tf_ok = false;
        for (int retry = 0; retry < 30 && rclcpp::ok() && !tf_ok; ++retry)
        {
            try {
                auto t1 = tf_buffer_->lookupTransform(
                    base_frame_, eef_frame_, tf2::TimePointZero, 1s);
                (void)t1;
                tf_ok = true;
            } catch (const tf2::TransformException& e) {
                RCLCPP_WARN(log, "  TF wait (retry %d/30): %s", retry+1, e.what());
            }
            rate.sleep();
        }
        if (!tf_ok) {
            RCLCPP_ERROR(log, "TF not available. Check that:");
            RCLCPP_ERROR(log, "  1. %s -> %s is published", base_frame_.c_str(), eef_frame_.c_str());
            return false;
        }
        RCLCPP_INFO(log,
                    "Robot TF ready. Camera marker TF is checked at each sample; "
                    "invisible markers are rejected.\n");
        return true;
    }

    void run()
    {
        while (rclcpp::ok() && g_running) {
            rclcpp::spin_some(node_);
            std::this_thread::sleep_for(100ms);
        }
    }

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr record_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr delete_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr count_srv_;

    std::string base_frame_, eef_frame_, camera_frame_, marker_frame_;
    std::string filepath_;
    double max_detection_age_ = 0.5;

    struct Sample {
        tf2::Transform base_to_eef;
        tf2::Transform camera_to_marker;
    };
    std::vector<Sample> samples_;

    std::pair<bool, std::string> recordSample()
    {
        auto log = node_->get_logger();
        Sample s;

        try {
            auto eef_msg = tf_buffer_->lookupTransform(
                base_frame_, eef_frame_, tf2::TimePointZero, 2s);
            tf2::fromMsg(eef_msg.transform, s.base_to_eef);
        } catch (const tf2::TransformException& e) {
            RCLCPP_ERROR(log, "Failed %s->%s: %s",
                         base_frame_.c_str(), eef_frame_.c_str(), e.what());
            return {false, "Failed to read robot TF: " + std::string(e.what())};
        }

        try {
            auto cam_msg = tf_buffer_->lookupTransform(
                camera_frame_, marker_frame_, tf2::TimePointZero, 2s);
            const double detection_age =
                (node_->now() - rclcpp::Time(cam_msg.header.stamp)).seconds();
            if (detection_age < -0.1 || detection_age > max_detection_age_)
            {
                RCLCPP_ERROR(log,
                             "Marker detection TF is stale (age %.3f s, limit %.3f s)",
                             detection_age, max_detection_age_);
                return {false, "Marker detection TF is stale; wait for a fresh image detection"};
            }
            tf2::fromMsg(cam_msg.transform, s.camera_to_marker);
        } catch (const tf2::TransformException& e) {
            RCLCPP_ERROR(log, "Marker detection TF unavailable (%s -> %s): %s",
                         camera_frame_.c_str(), marker_frame_.c_str(), e.what());
            return {false, "Marker is not visible or detector TF is unavailable: " +
                            std::string(e.what())};
        }

        samples_.push_back(s);
        double er, ep, eyaw_val;
        tf2::Matrix3x3(s.base_to_eef.getRotation()).getRPY(er, ep, eyaw_val);
        RCLCPP_INFO(log, "\n=== SAMPLE #%zu RECORDED ===", samples_.size());
        RCLCPP_INFO(log, "  EEF in %s:  pos(%.4f, %.4f, %.4f)  rpy(%.2f, %.2f, %.2f) deg",
                    base_frame_.c_str(),
                    s.base_to_eef.getOrigin().x(), s.base_to_eef.getOrigin().y(),
                    s.base_to_eef.getOrigin().z(),
                    er * 180 / M_PI, ep * 180 / M_PI, eyaw_val * 180 / M_PI);
        RCLCPP_INFO(log, "  Marker in %s:  pos(%.4f, %.4f, %.4f)\n",
                    camera_frame_.c_str(),
                    s.camera_to_marker.getOrigin().x(),
                    s.camera_to_marker.getOrigin().y(),
                    s.camera_to_marker.getOrigin().z());
        return {true, "Sample " + std::to_string(samples_.size()) + " recorded"};
    }

    std::pair<bool, std::string> saveSamples()
    {
        auto log = node_->get_logger();
        if (samples_.empty()) {
            return {false, "No samples to save"};
        }

        std::string path = filepath_;
        if (!path.empty() && path[0] == '~') {
            const char* home = std::getenv("HOME");
            if (home) path = std::string(home) + path.substr(1);
        }

        size_t slash = path.find_last_of('/');
        if (slash != std::string::npos) {
            std::string dir = path.substr(0, slash);
            std::string cmd = "mkdir -p \"" + dir + "\"";
            system(cmd.c_str());
        }

        std::ofstream ofs(path);
        if (!ofs) {
            return {false, "Cannot write to " + path};
        }

        ofs << std::fixed << std::setprecision(8);
        ofs << "# Hand-eye calibration samples\n";
        ofs << "# eef_x eef_y eef_z eef_qx eef_qy eef_qz eef_qw  "
               "marker_x marker_y marker_z marker_qx marker_qy marker_qz marker_qw\n";

        for (auto& s : samples_)
        {
            auto& t = s.base_to_eef;
            const auto& q = t.getRotation();
            ofs << t.getOrigin().x() << " " << t.getOrigin().y() << " " << t.getOrigin().z() << " ";
            ofs << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "  ";

            auto& tc = s.camera_to_marker;
            const auto& qc = tc.getRotation();
            ofs << tc.getOrigin().x() << " " << tc.getOrigin().y() << " " << tc.getOrigin().z() << " ";
            ofs << qc.x() << " " << qc.y() << " " << qc.z() << " " << qc.w() << "\n";
        }
        ofs.close();

        RCLCPP_INFO(log, "\n==============================================");
        RCLCPP_INFO(log, "  %zu samples saved to: %s", samples_.size(), path.c_str());
        RCLCPP_INFO(log, "==============================================\n");
        return {true, std::to_string(samples_.size()) + " samples saved to " + path};
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("xarm_handeye_sample_recorder",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
    HandEyeSampleRecorder recorder(node);
    if (!recorder.initialize()) {
        rclcpp::shutdown();
        return 1;
    }
    recorder.run();
    rclcpp::shutdown();
    return 0;
}
