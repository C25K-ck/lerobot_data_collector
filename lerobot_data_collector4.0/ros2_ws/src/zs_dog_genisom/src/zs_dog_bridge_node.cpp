#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "robot_sdk/sdk_callback.hpp"
#include "robot_sdk/sdk_client.hpp"

using robot_sdk::DeviceType;
using robot_sdk::JointStateData;
using robot_sdk::SDKClient;

namespace {

const std::array<const char*, 16> kDefaultJointNames = {
    "fl1", "fl2", "fl3", "fl4", "fr1", "fr2", "fr3", "fr4",
    "bl1", "bl2", "bl3", "bl4", "br1", "br2", "br3", "br4",
};

bool IsM1Family(DeviceType type) {
  switch (type) {
    case DeviceType::M1:
    case DeviceType::M1F:
    case DeviceType::M1_PRO:
    case DeviceType::M1F_PRO:
    case DeviceType::M1_ULTRA:
    case DeviceType::M1F_ULTRA:
    case DeviceType::M1_AIR:
    case DeviceType::M1F_AIR:
      return true;
    default:
      return false;
  }
}

bool IsL2Family(DeviceType type) {
  switch (type) {
    case DeviceType::L2:
    case DeviceType::L2_ULTRA:
    case DeviceType::L2F:
    case DeviceType::L2F_ULTRA:
      return true;
    default:
      return false;
  }
}

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

}  // namespace

class BridgeDataCallback : public robot_sdk::IDataCallback {
 public:
  void OnJointStateData(const JointStateData& data) override {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = data;
    has_data_ = true;
  }

  bool CopyLatest(JointStateData* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_data_) {
      return false;
    }
    *out = latest_;
    return true;
  }

 private:
  std::mutex mutex_;
  JointStateData latest_;
  bool has_data_ = false;
};

class ZsDogBridgeNode : public rclcpp::Node {
 public:
  ZsDogBridgeNode() : Node("zs_dog_bridge_node") {
    series_ = ToLower(declare_parameter<std::string>("series", "m1"));
    robot_ip_ = declare_parameter<std::string>("robot_ip", "192.168.234.1");
    robot_port_ = declare_parameter<std::string>("robot_port", "8082");
    joint_dim_ = declare_parameter<int>("joint_dim", 16);
    enable_control_ = declare_parameter<bool>("enable_control", false);
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 30.0);

    if (series_ != "m1" && series_ != "l2") {
      throw std::runtime_error("series 只能是 m1 或 l2");
    }
    if (joint_dim_ <= 0) {
      throw std::runtime_error("joint_dim 必须大于 0");
    }

    const std::string ns = (series_ == "m1") ? "/zsl_m1" : "/zsl_l2";
    joint_topic_ = ns + "/joint_states";
    cmd_topic_ = ns + "/joint_states_cmd";
    cmd_vel_topic_ = ns + "/cmd_vel";

    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>(joint_topic_, 10);
    cmd_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(cmd_topic_, 10);
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        cmd_vel_topic_, 10,
        std::bind(&ZsDogBridgeNode::OnCmdVel, this, std::placeholders::_1));

    data_cb_ = std::make_shared<BridgeDataCallback>();
    robot_sdk::ConnectionConfig cfg;
    cfg.auto_reconnect = true;
    client_ = std::make_unique<SDKClient>(
        [](const std::error_code&) {}, cfg, robot_sdk::TransportProtocol::Udp);
    client_->SetDataCallback(data_cb_);

    RCLCPP_INFO(get_logger(), "连接智身机器 %s:%s (series=%s)", robot_ip_.c_str(),
                robot_port_.c_str(), series_.c_str());
    if (auto ec = client_->Connect(robot_ip_, robot_port_, true)) {
      throw std::runtime_error("连接失败: " + ec.message());
    }

    const auto info = client_->GetDeviceInfo();
    const char* type_name = robot_sdk::DeviceTypeName(info.device_type);
    RCLCPP_INFO(get_logger(), "握手成功 device_type=%s sn=%s sdk=%s proto=%s sys=%s",
                type_name, info.sn.c_str(), client_->Version().c_str(),
                client_->ProtocolVersion().c_str(), client_->SystemVersion().c_str());

    if (series_ == "m1" && !IsM1Family(info.device_type) &&
        info.device_type != DeviceType::UNKNOWN) {
      RCLCPP_WARN(get_logger(), "当前设备 %s 不像 M1 系列，请确认是否选错了 ZSL-M1",
                  type_name);
    }
    if (series_ == "l2" && !IsL2Family(info.device_type) &&
        info.device_type != DeviceType::UNKNOWN) {
      RCLCPP_WARN(get_logger(), "当前设备 %s 不像 L2 系列，请确认是否选错了 ZSL-L2",
                  type_name);
    }

    if (auto ec = client_->SetJointStateConfig(true)) {
      RCLCPP_WARN(get_logger(), "开启关节上报失败: %s", ec.message().c_str());
    }

    last_cmd_.assign(static_cast<size_t>(joint_dim_), 0.0f);

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_rate_hz_));
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                               std::bind(&ZsDogBridgeNode::OnTimer, this));

    RCLCPP_INFO(get_logger(), "发布 %s 与 %s ；订阅 %s (enable_control=%s)",
                joint_topic_.c_str(), cmd_topic_.c_str(), cmd_vel_topic_.c_str(),
                enable_control_ ? "true" : "false");
  }

  ~ZsDogBridgeNode() override {
    if (client_) {
      client_->SetJointStateConfig(false);
      client_->Disconnect(true);
    }
  }

 private:
  void OnCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      last_cmd_.assign(static_cast<size_t>(joint_dim_), 0.0f);
      if (joint_dim_ >= 1) last_cmd_[0] = static_cast<float>(msg->linear.x);
      if (joint_dim_ >= 2) last_cmd_[1] = static_cast<float>(msg->linear.y);
      if (joint_dim_ >= 3) last_cmd_[2] = static_cast<float>(msg->angular.z);
    }
    if (!enable_control_ || !client_ || !client_->IsConnected()) {
      return;
    }
    // SDK: Move(left_right, forward_back, yaw)
    client_->Move(static_cast<float>(msg->linear.y), static_cast<float>(msg->linear.x),
                  static_cast<float>(msg->angular.z));
  }

  void OnTimer() {
    JointStateData data;
    if (!data_cb_->CopyLatest(&data)) {
      return;
    }

    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    js.header.frame_id = series_ == "m1" ? "zsl_m1" : "zsl_l2";

    const size_t n = std::max(data.names.size(), data.positions.size());
    js.name.resize(n);
    js.position.resize(n, 0.0);
    js.velocity.resize(n, 0.0);
    js.effort.resize(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      if (i < data.names.size() && !data.names[i].empty()) {
        js.name[i] = data.names[i];
      } else if (i < kDefaultJointNames.size()) {
        js.name[i] = kDefaultJointNames[i];
      } else {
        js.name[i] = "joint_" + std::to_string(i);
      }
      if (i < data.positions.size()) js.position[i] = data.positions[i];
      if (i < data.velocities.size()) js.velocity[i] = data.velocities[i];
      if (i < data.efforts.size()) js.effort[i] = data.efforts[i];
    }
    joint_pub_->publish(js);

    std_msgs::msg::Float32MultiArray cmd;
    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      cmd.data = last_cmd_;
    }
    cmd_pub_->publish(cmd);
  }

  std::string series_;
  std::string robot_ip_;
  std::string robot_port_;
  int joint_dim_ = 16;
  bool enable_control_ = false;
  double publish_rate_hz_ = 30.0;
  std::string joint_topic_;
  std::string cmd_topic_;
  std::string cmd_vel_topic_;

  std::unique_ptr<SDKClient> client_;
  std::shared_ptr<BridgeDataCallback> data_cb_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr cmd_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex cmd_mutex_;
  std::vector<float> last_cmd_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<ZsDogBridgeNode>());
  } catch (const std::exception& ex) {
    fprintf(stderr, "[zs_dog_bridge_node] %s\n", ex.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
