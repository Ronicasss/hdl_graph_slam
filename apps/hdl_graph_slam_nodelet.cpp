// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>

#include <ros/ros.h>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <pcl_ros/point_cloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>

#include <std_msgs/Time.h>
#include <nav_msgs/Odometry.h>
#include <nmea_msgs/Sentence.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <geographic_msgs/GeoPointStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <hdl_graph_slam/FloorCoeffs.h>

#include <hdl_graph_slam/SaveMap.h>
#include <hdl_graph_slam/DumpGraph.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/ros_time_hash.hpp>

#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/keyframe_updater.hpp>
#include <hdl_graph_slam/loop_detector.hpp>
#include <hdl_graph_slam/information_matrix_calculator.hpp>
#include <hdl_graph_slam/map_cloud_generator.hpp>
#include <hdl_graph_slam/nmea_sentence_parser.hpp>

#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>

#include "hdl_graph_slam/building_tools.hpp"
#include "hdl_graph_slam/building_node.hpp"
#include "hdl_graph_slam/matrix.h"
#include <pclomp/gicp_omp.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>

namespace hdl_graph_slam {

class HdlGraphSlamNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZI PointT;
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> ApproxSyncPolicy;

  HdlGraphSlamNodelet() {}
  virtual ~HdlGraphSlamNodelet() {}

  virtual void onInit() {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    // init parameters
    map_frame_id = private_nh.param<std::string>("map_frame_id", "map");
    odom_frame_id = private_nh.param<std::string>("odom_frame_id", "odom");
    map_cloud_resolution = private_nh.param<double>("map_cloud_resolution", 0.05);
    trans_odom2map.setIdentity();

    max_keyframes_per_update = private_nh.param<int>("max_keyframes_per_update", 10);

    // buildings parameters
    buildings_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/hdl_graph_slam/buildings_cloud", 1);
    odom_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/hdl_graph_slam/odom_cloud", 32);
    transformed_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/hdl_graph_slam/transformed_cloud", 32);
    original_odom_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/hdl_graph_slam/original_odom_cloud", 32);

    lidar_range = private_nh.param<float>("lidar_range", 300);
    ground_floor_max_thresh = private_nh.param<double>("ground_floor_max_thresh", 0.5);
    radius_search = private_nh.param<double>("radius_search", 1);
    min_neighbors_in_radius = private_nh.param<double>("min_neighbors_in_radius", 100);
    enter = 1;
    ii = -1;
    //read ground truth
    gt = loadPoses(private_nh.param<std::string>("gt_path", ""));
    std::cout << "gt poses loaded: " << gt.size() << std::endl;
    gt_markers_pub = mt_nh.advertise<visualization_msgs::Marker>("/hdl_graph_slam/gt_markers", 16);
    first_guess = true;
    prev_guess = Eigen::Matrix4f::Identity();
    prev_pose = Eigen::Matrix4f::Identity();
    b_tf_pub = mt_nh.advertise<geometry_msgs::TransformStamped>("/hdl_graph_slam/b_tf", 16);

    //
    anchor_node = nullptr;
    anchor_edge = nullptr;
    floor_plane_node = nullptr;
    graph_slam.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
    keyframe_updater.reset(new KeyframeUpdater(private_nh));
    loop_detector.reset(new LoopDetector(private_nh));
    map_cloud_generator.reset(new MapCloudGenerator());
    inf_calclator.reset(new InformationMatrixCalculator(private_nh));
    nmea_parser.reset(new NmeaSentenceParser());

    gps_time_offset = private_nh.param<double>("gps_time_offset", 0.0);
    gps_edge_stddev_xy = private_nh.param<double>("gps_edge_stddev_xy", 10000.0);
    gps_edge_stddev_z = private_nh.param<double>("gps_edge_stddev_z", 10.0);
    floor_edge_stddev = private_nh.param<double>("floor_edge_stddev", 10.0);

    imu_time_offset = private_nh.param<double>("imu_time_offset", 0.0);
    enable_imu_orientation = private_nh.param<bool>("enable_imu_orientation", false);
    enable_imu_acceleration = private_nh.param<bool>("enable_imu_acceleration", false);
    imu_orientation_edge_stddev = private_nh.param<double>("imu_orientation_edge_stddev", 0.1);
    imu_acceleration_edge_stddev = private_nh.param<double>("imu_acceleration_edge_stddev", 3.0);

    points_topic = private_nh.param<std::string>("points_topic", "/velodyne_points");

    // subscribers
    odom_sub.reset(new message_filters::Subscriber<nav_msgs::Odometry>(mt_nh, "/odom", 256));
    cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/filtered_points", 32));
    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *odom_sub, *cloud_sub));
    sync->registerCallback(boost::bind(&HdlGraphSlamNodelet::cloud_callback, this, _1, _2));
    imu_sub = nh.subscribe("/gpsimu_driver/imu_data", 1024, &HdlGraphSlamNodelet::imu_callback, this);
    floor_sub = nh.subscribe("/floor_detection/floor_coeffs", 1024, &HdlGraphSlamNodelet::floor_coeffs_callback, this);

    if(private_nh.param<bool>("enable_gps", true)) {
      gps_sub = mt_nh.subscribe("/gps/geopoint", 1024, &HdlGraphSlamNodelet::gps_callback, this);
      nmea_sub = mt_nh.subscribe("/gpsimu_driver/nmea_sentence", 1024, &HdlGraphSlamNodelet::nmea_callback, this);
      navsat_sub = mt_nh.subscribe("/gps/navsat", 1024, &HdlGraphSlamNodelet::navsat_callback, this);
    }

    // publishers
    markers_pub = mt_nh.advertise<visualization_msgs::MarkerArray>("/hdl_graph_slam/markers", 16);
    odom2map_pub = mt_nh.advertise<geometry_msgs::TransformStamped>("/hdl_graph_slam/odom2pub", 16);
    map_points_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/hdl_graph_slam/map_points", 1, true);
    read_until_pub = mt_nh.advertise<std_msgs::Header>("/hdl_graph_slam/read_until", 32);

    dump_service_server = mt_nh.advertiseService("/hdl_graph_slam/dump", &HdlGraphSlamNodelet::dump_service, this);
    save_map_service_server = mt_nh.advertiseService("/hdl_graph_slam/save_map", &HdlGraphSlamNodelet::save_map_service, this);

    graph_updated = false;
    double graph_update_interval = private_nh.param<double>("graph_update_interval", 3.0);
    double map_cloud_update_interval = private_nh.param<double>("map_cloud_update_interval", 10.0);
    optimization_timer = mt_nh.createWallTimer(ros::WallDuration(graph_update_interval), &HdlGraphSlamNodelet::optimization_timer_callback, this);
    map_publish_timer = mt_nh.createWallTimer(ros::WallDuration(map_cloud_update_interval), &HdlGraphSlamNodelet::map_points_publish_timer_callback, this);
  }

private:
  /**
   * @brief received point clouds are pushed to #keyframe_queue
   * @param odom_msg
   * @param cloud_msg
   */
  void cloud_callback(const nav_msgs::OdometryConstPtr& odom_msg, const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    ii++;
    const ros::Time& stamp = cloud_msg->header.stamp;
    Eigen::Isometry3d odom = odom2isometry(odom_msg);

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(base_frame_id.empty()) {
      base_frame_id = cloud_msg->header.frame_id;
    }

    if(!keyframe_updater->update(odom)) {
      std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
      if(keyframe_queue.empty()) {
        std_msgs::Header read_until;
        read_until.stamp = stamp + ros::Duration(10, 0);
        read_until.frame_id = points_topic;
        read_until_pub.publish(read_until);
        read_until.frame_id = "/filtered_points";
        read_until_pub.publish(read_until);
      }

      return;
    }

    double accum_d = keyframe_updater->get_accum_distance();
    KeyFrame::Ptr keyframe(new KeyFrame(stamp, odom, accum_d, cloud, ii));

    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
    keyframe_queue.push_back(keyframe);
  }

  /**
   * @brief this method adds all the keyframes in #keyframe_queue to the pose graph (odometry edges)
   * @return if true, at least one keyframe was added to the pose graph
   */
  bool flush_keyframe_queue() {
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);

    if(keyframe_queue.empty()) {
      return false;
    }

    trans_odom2map_mutex.lock();
    Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();

    int num_processed = 0;
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); i++) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      // new_keyframes will be tested later for loop closure
      new_keyframes.push_back(keyframe);

      // add pose node
      Eigen::Isometry3d odom = odom2map * keyframe->odom;
      keyframe->node = graph_slam->add_se3_node(odom);
      keyframe_hash[keyframe->stamp] = keyframe;

      // fix the first node
      if(keyframes.empty() && new_keyframes.size() == 1) {
        if(private_nh.param<bool>("fix_first_node", false)) {
          Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(6, 6);
          std::stringstream sst(private_nh.param<std::string>("fix_first_node_stddev", "1 1 1 1 1 1"));
          for(int i = 0; i < 6; i++) {
            double stddev = 1.0;
            sst >> stddev;
            inf(i, i) = 1.0 / stddev;
          }

          anchor_node = graph_slam->add_se3_node(Eigen::Isometry3d::Identity());
          anchor_node->setFixed(true);
          anchor_edge = graph_slam->add_se3_edge(anchor_node, keyframe->node, Eigen::Isometry3d::Identity(), inf);
        }
      }

      if(i == 0 && keyframes.empty()) {
        continue;
      }

      // add edge between consecutive keyframes
      const auto& prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];

      Eigen::Isometry3d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;
      Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, relative_pose);
      auto edge = graph_slam->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("odometry_edge_robust_kernel", "NONE"), private_nh.param<double>("odometry_edge_robust_kernel_size", 1.0));
    }

    std_msgs::Header read_until;
    read_until.stamp = keyframe_queue[num_processed]->stamp + ros::Duration(10, 0);
    read_until.frame_id = points_topic;
    read_until_pub.publish(read_until);
    read_until.frame_id = "/filtered_points";
    read_until_pub.publish(read_until);

    keyframe_queue.erase(keyframe_queue.begin(), keyframe_queue.begin() + num_processed + 1);
    return true;
  }

  void nmea_callback(const nmea_msgs::SentenceConstPtr& nmea_msg) {
    GPRMC grmc = nmea_parser->parse(nmea_msg->sentence);

    if(grmc.status != 'A') {
      return;
    }

    geographic_msgs::GeoPointStampedPtr gps_msg(new geographic_msgs::GeoPointStamped());
    gps_msg->header = nmea_msg->header;
    gps_msg->position.latitude = grmc.latitude;
    gps_msg->position.longitude = grmc.longitude;
    gps_msg->position.altitude = NAN;

    gps_callback(gps_msg);
  }

  void navsat_callback(const sensor_msgs::NavSatFixConstPtr& navsat_msg) {
    geographic_msgs::GeoPointStampedPtr gps_msg(new geographic_msgs::GeoPointStamped());
    gps_msg->header = navsat_msg->header;
    gps_msg->position.latitude = navsat_msg->latitude;
    gps_msg->position.longitude = navsat_msg->longitude;
    gps_msg->position.altitude = navsat_msg->altitude;
    gps_callback(gps_msg);
  }

  /**
   * @brief received gps data is added to #gps_queue
   * @param gps_msg
   */
  void gps_callback(const geographic_msgs::GeoPointStampedPtr& gps_msg) {
    std::lock_guard<std::mutex> lock(gps_queue_mutex);
    gps_msg->header.stamp += ros::Duration(gps_time_offset);
    gps_queue.push_back(gps_msg);
  }

  /**
   * @brief
   * @return
   */
  bool flush_gps_queue() {
    std::lock_guard<std::mutex> lock(gps_queue_mutex);

    if(keyframes.empty() || gps_queue.empty()) {
      return false;
    }

    bool updated = false;
    auto gps_cursor = gps_queue.begin();

    for(auto& keyframe : keyframes) {
      if(keyframe->stamp > gps_queue.back()->header.stamp) {
        break;
      }

      if(keyframe->stamp < (*gps_cursor)->header.stamp || keyframe->utm_coord) {
        continue;
      }

      // find the gps data which is closest to the keyframe
      auto closest_gps = gps_cursor;
      for(auto gps = gps_cursor; gps != gps_queue.end(); gps++) {
        auto dt = ((*closest_gps)->header.stamp - keyframe->stamp).toSec();
        auto dt2 = ((*gps)->header.stamp - keyframe->stamp).toSec();
        if(std::abs(dt) < std::abs(dt2)) {
          break;
        }

        closest_gps = gps;
      }

      // if the time residual between the gps and keyframe is too large, skip it
      gps_cursor = closest_gps;
      if(0.2 < std::abs(((*closest_gps)->header.stamp - keyframe->stamp).toSec())) {
        continue;
      }

      // convert (latitude, longitude, altitude) -> (easting, northing, altitude) in UTM coordinate
      geodesy::UTMPoint utm;
      geodesy::fromMsg((*closest_gps)->position, utm);
      Eigen::Vector3d xyz(utm.easting, utm.northing, utm.altitude);

      // the first gps data position will be the origin of the map
      if(!zero_utm) {
        zero_utm = xyz;
        zero_utm_zone = utm.zone;
        zero_utm_band = utm.band;
      }
      xyz -= (*zero_utm);

      keyframe->utm_coord = xyz;
      keyframe->utm_zone = utm.zone;
      keyframe->utm_band = utm.band;

      /*
      g2o::OptimizableGraph::Edge* edge;
      if(std::isnan(xyz.z())) {
        Eigen::Matrix2d information_matrix = Eigen::Matrix2d::Identity() / gps_edge_stddev_xy;
        edge = graph_slam->add_se3_prior_xy_edge(keyframe->node, xyz.head<2>(), information_matrix);
      } else {
        Eigen::Matrix3d information_matrix = Eigen::Matrix3d::Identity();
        information_matrix.block<2, 2>(0, 0) /= gps_edge_stddev_xy;
        information_matrix(2, 2) /= gps_edge_stddev_z;
        edge = graph_slam->add_se3_prior_xyz_edge(keyframe->node, xyz, information_matrix);
      }
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("gps_edge_robust_kernel", "NONE"), private_nh.param<double>("gps_edge_robust_kernel_size", 1.0));
      */
      updated = true;
    }

    auto remove_loc = std::upper_bound(gps_queue.begin(), gps_queue.end(), keyframes.back()->stamp, [=](const ros::Time& stamp, const geographic_msgs::GeoPointStampedConstPtr& geopoint) { return stamp < geopoint->header.stamp; });
    gps_queue.erase(gps_queue.begin(), remove_loc);
    return updated;
  }

  void imu_callback(const sensor_msgs::ImuPtr& imu_msg) {
    if(!enable_imu_orientation && !enable_imu_acceleration) {
      return;
    }

    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    imu_msg->header.stamp += ros::Duration(imu_time_offset);
    imu_queue.push_back(imu_msg);
  }

  bool flush_imu_queue() {
    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    if(keyframes.empty() || imu_queue.empty() || base_frame_id.empty()) {
      return false;
    }

    bool updated = false;
    auto imu_cursor = imu_queue.begin();

    for(auto& keyframe : keyframes) {
      if(keyframe->stamp > imu_queue.back()->header.stamp) {
        break;
      }

      if(keyframe->stamp < (*imu_cursor)->header.stamp || keyframe->acceleration) {
        continue;
      }

      // find imu data which is closest to the keyframe
      auto closest_imu = imu_cursor;
      for(auto imu = imu_cursor; imu != imu_queue.end(); imu++) {
        auto dt = ((*closest_imu)->header.stamp - keyframe->stamp).toSec();
        auto dt2 = ((*imu)->header.stamp - keyframe->stamp).toSec();
        if(std::abs(dt) < std::abs(dt2)) {
          break;
        }

        closest_imu = imu;
      }

      imu_cursor = closest_imu;
      if(0.2 < std::abs(((*closest_imu)->header.stamp - keyframe->stamp).toSec())) {
        continue;
      }

      const auto& imu_ori = (*closest_imu)->orientation;
      const auto& imu_acc = (*closest_imu)->linear_acceleration;

      geometry_msgs::Vector3Stamped acc_imu;
      geometry_msgs::Vector3Stamped acc_base;
      geometry_msgs::QuaternionStamped quat_imu;
      geometry_msgs::QuaternionStamped quat_base;

      quat_imu.header.frame_id = acc_imu.header.frame_id = (*closest_imu)->header.frame_id;
      quat_imu.header.stamp = acc_imu.header.stamp = ros::Time(0);
      acc_imu.vector = (*closest_imu)->linear_acceleration;
      quat_imu.quaternion = (*closest_imu)->orientation;
      
      try {
        tf_listener.transformVector(base_frame_id, acc_imu, acc_base);
        tf_listener.transformQuaternion(base_frame_id, quat_imu, quat_base);
      } catch(std::exception& e) {
        std::cerr << "failed to find transform!!" << std::endl;
        return false;
      }
      
      keyframe->acceleration = Eigen::Vector3d(acc_base.vector.x, acc_base.vector.y, acc_base.vector.z);
      keyframe->orientation = Eigen::Quaterniond(quat_base.quaternion.w, quat_base.quaternion.x, quat_base.quaternion.y, quat_base.quaternion.z);
      keyframe->orientation = keyframe->orientation;
      if(keyframe->orientation->w() < 0.0) {
        keyframe->orientation->coeffs() = -keyframe->orientation->coeffs();
      }
      /*
      if(enable_imu_orientation) {
        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(3, 3) / imu_orientation_edge_stddev;
        auto edge = graph_slam->add_se3_prior_quat_edge(keyframe->node, *keyframe->orientation, info);
        graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("imu_orientation_edge_robust_kernel", "NONE"), private_nh.param<double>("imu_orientation_edge_robust_kernel_size", 1.0));
      }

      if(enable_imu_acceleration) {
        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(3, 3) / imu_acceleration_edge_stddev;
        g2o::OptimizableGraph::Edge* edge = graph_slam->add_se3_prior_vec_edge(keyframe->node, -Eigen::Vector3d::UnitZ(), *keyframe->acceleration, info);
        graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("imu_acceleration_edge_robust_kernel", "NONE"), private_nh.param<double>("imu_acceleration_edge_robust_kernel_size", 1.0));
      }
      */
      updated = true;
    }

    auto remove_loc = std::upper_bound(imu_queue.begin(), imu_queue.end(), keyframes.back()->stamp, [=](const ros::Time& stamp, const sensor_msgs::ImuConstPtr& imu) { return stamp < imu->header.stamp; });
    imu_queue.erase(imu_queue.begin(), remove_loc);
    return updated;
  }

  /**
   * @brief received floor coefficients are added to #floor_coeffs_queue
   * @param floor_coeffs_msg
   */
  void floor_coeffs_callback(const hdl_graph_slam::FloorCoeffsConstPtr& floor_coeffs_msg) {
    if(floor_coeffs_msg->coeffs.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(floor_coeffs_queue_mutex);
    floor_coeffs_queue.push_back(floor_coeffs_msg);
  }

  /**
   * @brief this methods associates floor coefficients messages with registered keyframes, and then adds the associated coeffs to the pose graph
   * @return if true, at least one floor plane edge is added to the pose graph
   */
  bool flush_floor_queue() {
    std::lock_guard<std::mutex> lock(floor_coeffs_queue_mutex);

    if(keyframes.empty()) {
      return false;
    }

    const auto& latest_keyframe_stamp = keyframes.back()->stamp;

    bool updated = false;
    for(const auto& floor_coeffs : floor_coeffs_queue) {
      if(floor_coeffs->header.stamp > latest_keyframe_stamp) {
        break;
      }

      auto found = keyframe_hash.find(floor_coeffs->header.stamp);
      if(found == keyframe_hash.end()) {
        continue;
      }

      if(!floor_plane_node) {
        floor_plane_node = graph_slam->add_plane_node(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0));
        floor_plane_node->setFixed(true);
      }

      const auto& keyframe = found->second;

      Eigen::Vector4d coeffs(floor_coeffs->coeffs[0], floor_coeffs->coeffs[1], floor_coeffs->coeffs[2], floor_coeffs->coeffs[3]);
      Eigen::Matrix3d information = Eigen::Matrix3d::Identity() * (1.0 / floor_edge_stddev);
      auto edge = graph_slam->add_se3_plane_edge(keyframe->node, floor_plane_node, coeffs, information);
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("floor_edge_robust_kernel", "NONE"), private_nh.param<double>("floor_edge_robust_kernel_size", 1.0));

      keyframe->floor_coeffs = coeffs;

      updated = true;
    }

    auto remove_loc = std::upper_bound(floor_coeffs_queue.begin(), floor_coeffs_queue.end(), latest_keyframe_stamp, [=](const ros::Time& stamp, const hdl_graph_slam::FloorCoeffsConstPtr& coeffs) { return stamp < coeffs->header.stamp; });
    floor_coeffs_queue.erase(floor_coeffs_queue.begin(), remove_loc);

    //std::cout << "floor_updated: " << updated << std::endl;
    return updated;
  }

  bool update_buildings_nodes() {
    bool b_updated = false;
    for(auto& keyframe : keyframes) {
      // if the keyframe is never been aligned with map and there is a gps, then enter
      if((keyframe->buildings_nodes).empty() && keyframe->utm_coord && enter) { 
        
        std::cout << "update_buildings_nodes - get new buildings " << ii << std::endl;
        
          //enter = 0;
        

        geodesy::UTMPoint utm;
        Eigen::Vector3d e_utm_coord = keyframe->utm_coord.get();
        Eigen::Vector3d e_zero_utm = zero_utm.get();
        // e_utm_coord are the coords of current keyframe wrt zero_utm, so to get real coords we add zero_utm
        utm.easting = e_utm_coord(0) + e_zero_utm(0);
        utm.northing = e_utm_coord(1) + e_zero_utm(1);
        utm.altitude = e_utm_coord(2) + e_zero_utm(2);
        utm.zone = keyframe->utm_zone.get();
        utm.band = keyframe->utm_band.get();
        geographic_msgs::GeoPoint lla = geodesy::toMsg(utm); // convert from utm to lla

        // download and parse buildings
        std::vector<Building> new_buildings = BuildingTools::getBuildings(lla.latitude, lla.longitude, lidar_range, e_zero_utm);
        if(new_buildings.size() > 0) {
          std::cout << "We found buildings!" << std::endl;
          b_updated = true;
          //e_zero_utm(2) = 0;
          //e_utm_coord(2) = 0;
 
          std::vector<BuildingNode::Ptr> bnodes; // vector containing all buildings nodes (new and not new)
          // buildingsCloud is the cloud containing all buildings
          pcl::PointCloud<pcl::PointXYZ>::Ptr buildingsCloud(new pcl::PointCloud<pcl::PointXYZ>);

          bool first_b = first_guess;

          // construct building nodes
          for(auto it2 = new_buildings.begin(); it2 != new_buildings.end(); it2++)
          {
            Building btemp = *it2;
            *buildingsCloud += *(btemp.geometry);
            BuildingNode::Ptr bntemp(new BuildingNode());
            bntemp = get_building_node(btemp);
            if(bntemp == nullptr) { // enter if the building is new
              BuildingNode::Ptr bt(new BuildingNode());
              bt->building = btemp;
              bt->setReferenceSystem();
              // retrieve informations to build the se3 node
              // translation
              Eigen::Vector3d T = bt->local_origin; // local origin is already referring to zero_utm
              // rotation
              Eigen::Matrix3d R = Eigen::Matrix3d::Identity(); // gps coords don't give orientation
              // rototranslation
              Eigen::Isometry3d A;
              A.linear() = R;
              A.translation() = T;
              // set global_origin
              bt->global_origin = A;
              // set the node
              bt->node = graph_slam->add_se3_node(A); 
              if(first_b) {
                bt->node->setFixed(true);
                first_b = false;
                std::cout << "here" << std::endl;
              }
            

              buildings.push_back(bt);
              bnodes.push_back(bt);
            } else {
              bnodes.push_back(bntemp);
            }
          }
          
          // pre-processing on odom cloud
          pcl::PointCloud<pcl::PointXYZ>::Ptr odomCloud(new pcl::PointCloud<pcl::PointXYZ>); // cloud containing lidar data
          pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_2(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_3(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_4(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::copyPointCloud(*keyframe->cloud,*temp_cloud); // convert from pointxyzi to pointxyz

          //std::cout << "size 1: " << temp_cloud->size() << std::endl; 
          // height filtering
          pcl::PassThrough<pcl::PointXYZ> pass;
          pass.setInputCloud (temp_cloud);
          pass.setFilterFieldName ("z");
          pass.setFilterLimits (ground_floor_max_thresh, 100.0);
          pass.filter(*temp_cloud_2);
          temp_cloud_2->header = (*keyframe->cloud).header;
          //std::cout << "size 2: " << temp_cloud_2->size() << std::endl;
          // downsampling + outlier removal
          pcl::Filter<pcl::PointXYZ>::Ptr downsample_filter;
          double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
          boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZ>> voxelgrid(new pcl::VoxelGrid<pcl::PointXYZ>());
          voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
          downsample_filter = voxelgrid;
          downsample_filter->setInputCloud(temp_cloud_2);
          downsample_filter->filter(*temp_cloud_3);
          temp_cloud_3->header = temp_cloud_2->header;

          //std::cout << "size 3: " << temp_cloud_3->size() << std::endl;

          pcl::RadiusOutlierRemoval<pcl::PointXYZ>::Ptr rad(new pcl::RadiusOutlierRemoval<pcl::PointXYZ>());
          rad->setRadiusSearch(radius_search);
          rad->setMinNeighborsInRadius(min_neighbors_in_radius);
          //std::cout << "rad: " << rad->getRadiusSearch() << " neighbors: " << rad->getMinNeighborsInRadius() << std::endl; 
          pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
          rad->setInputCloud(temp_cloud_3);
          rad->filter(*temp_cloud_4);
          temp_cloud_4->header = temp_cloud_3->header;

          pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_5(new pcl::PointCloud<pcl::PointXYZ>);
          // project the cloud on plane z=0
          pcl::ProjectInliers<pcl::PointXYZ> proj;
          pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
          coefficients->values.resize(4);
          coefficients->values[0]=0;
          coefficients->values[1]=0;
          coefficients->values[2]=1;
          coefficients->values[3]=0; 
          proj.setModelType(pcl::SACMODEL_PLANE); 
          proj.setInputCloud (temp_cloud_4);
          proj.setModelCoefficients (coefficients);
          proj.filter (*temp_cloud_5);
          temp_cloud_5->header = temp_cloud_4->header;

          trans_odom2map_mutex.lock();
          Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
          trans_odom2map_mutex.unlock();

          /*tf::StampedTransform transform_t;
          tf_listener.lookupTransform("map", "base_link", ros::Time(0), transform_t);
          Eigen::Quaterniond q;
          tf::quaternionTFToEigen(transform_t.getRotation(), q);
          Eigen::Vector3d v;
          tf::vectorTFToEigen (transform_t.getOrigin(), v);
          Eigen::Isometry3d tfs = Eigen::Isometry3d::Identity();
          tfs.linear() = q.normalized().toRotationMatrix();
          tfs.translation() = v;*/
          
          pcl::transformPointCloud(*temp_cloud_5, *odomCloud, (keyframe->odom).matrix());
          odomCloud->header = temp_cloud_5->header;
          odomCloud->header.frame_id = "odom";
          buildingsCloud->header.frame_id = "map";
          
           // publish original odom cloud
          sensor_msgs::PointCloud2Ptr oc_cloud_msg(new sensor_msgs::PointCloud2());
          pcl::toROSMsg(*keyframe->cloud, *oc_cloud_msg);
          oc_cloud_msg->header.frame_id = "base_link";
          oc_cloud_msg->header.stamp = keyframe->stamp;
          original_odom_pub.publish(oc_cloud_msg);
          // publish odom cloud
          sensor_msgs::PointCloud2Ptr o_cloud_msg(new sensor_msgs::PointCloud2());
          pcl::toROSMsg(*odomCloud, *o_cloud_msg);
          o_cloud_msg->header.frame_id = "odom";
          o_cloud_msg->header.stamp = keyframe->stamp;
          odom_pub.publish(o_cloud_msg);
          // publish buildings cloud
          sensor_msgs::PointCloud2Ptr b_cloud_msg(new sensor_msgs::PointCloud2());
          pcl::toROSMsg(*buildingsCloud, *b_cloud_msg);
          b_cloud_msg->header.frame_id = "map";
          b_cloud_msg->header.stamp = keyframe->stamp;
          buildings_pub.publish(b_cloud_msg);

          Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
          if(first_guess) {
            std::cout << "first guess" << std::endl;
            Eigen::Quaterniond orientation = *(keyframe->orientation);
            orientation.normalize();
            Eigen::Matrix3f R = (orientation.cast<float>()).toRotationMatrix();
            guess.block<3,1>(0,3) = e_utm_coord.cast<float>();
            guess.block<3,3>(0,0) = R;
            first_guess = false;
          } else {
            std::cout << "prev" << std::endl;
            guess = prev_guess;
          }
          std::cout << "guess: " << guess << std::endl;

          // gicp_omp registration
          pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr registration;

          //pcl::registration::WarpPointRigid3D<pcl::PointXYZ, pcl::PointXYZ>::Ptr warp_fcn(new pcl::registration::WarpPointRigid3D<pcl::PointXYZ,pcl::PointXYZ>);
          //pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ>::Ptr te(new pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ>);
          //te->setWarpFunction(warp_fcn);

          std::cout << "registration: FAST_VGICP" << std::endl;
          boost::shared_ptr<fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>> gicp(new fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>());
          gicp->setNumThreads(private_nh.param<int>("reg_num_threads", 0));
          gicp->setResolution(private_nh.param<double>("reg_resolution", 1.0));
          
          //std::cout << "registration: FAST_GICP" << std::endl;
          //boost::shared_ptr<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>> gicp(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>());
          //gicp->setNumThreads(private_nh.param<int>("reg_num_threads", 0));

          //std::cout << "registration: GICP_OMP" << std::endl;
          //boost::shared_ptr<pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>> gicp(new pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>());
          
          if(private_nh.param<bool>("enable_transformation_epsilon", true))
            gicp->setTransformationEpsilon(private_nh.param<double>("transformation_epsilon", 0.01));
          if(private_nh.param<bool>("enable_maximum_iterations", true))
            gicp->setMaximumIterations(private_nh.param<int>("maximum_iterations", 64));
          if(private_nh.param<bool>("enable_use_reciprocal_correspondences", true))
            //gicp->setUseReciprocalCorrespondences(private_nh.param<bool>("use_reciprocal_correspondences", false));
          if(private_nh.param<bool>("enable_gicp_correspondence_randomness", true))
            gicp->setCorrespondenceRandomness(private_nh.param<int>("gicp_correspondence_randomness", 20));
          if(private_nh.param<bool>("enable_gicp_max_optimizer_iterations", true))
            //gicp->setMaximumOptimizerIterations(private_nh.param<int>("gicp_max_optimizer_iterations", 20));

          if(private_nh.param<bool>("enable_gicp_max_correspondance_distance", false))
            gicp->setMaxCorrespondenceDistance(private_nh.param<double>("gicp_max_correspondance_distance", 0.05));
          if(private_nh.param<bool>("enable_gicp_euclidean_fitness_epsilon", false))
            gicp->setEuclideanFitnessEpsilon(private_nh.param<double>("gicp_euclidean_fitness_epsilon", 1));
          if(private_nh.param<bool>("enable_gicp_ransac_outlier_threshold", false)) 
            gicp->setRANSACOutlierRejectionThreshold(private_nh.param<double>("gicp_ransac_outlier_threshold", 1.5));

          std::cout << "max dist: " << gicp->getMaxCorrespondenceDistance() << std::endl;
          //std::cout << "ransac: " << gicp->getRANSACOutlierRejectionThreshold() << std::endl;
          //std::cout << "fitness: " << gicp->getEuclideanFitnessEpsilon() << std::endl;

          registration = gicp;
          //registration->setTransformationEstimation(te);
          registration->setInputTarget(buildingsCloud);
          registration->setInputSource(odomCloud);
          pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
          registration->align(*aligned, guess);
          std::cout << "has converged:" << registration->hasConverged() << " score: " << registration->getFitnessScore() << std::endl;
          Eigen::Matrix4f transformation = registration->getFinalTransformation();
          std::cout<< "Transformation: " << transformation << std::endl;
          prev_guess = transformation;

          // publish icp resulting transform
          sensor_msgs::PointCloud2Ptr t_cloud_msg(new sensor_msgs::PointCloud2());
          pcl::toROSMsg(*aligned, *t_cloud_msg);
          aligned->header.frame_id = "map";
          transformed_pub.publish(aligned);

          Eigen::Matrix4d t_s_bs = transformation.cast<double>();
          
          Eigen::Isometry3d t_s_bs_iso = Eigen::Isometry3d::Identity();
          t_s_bs_iso.translation() = t_s_bs.block<3, 1>(0, 3);
          t_s_bs_iso.linear() = t_s_bs.block<3, 3>(0, 0);

          /*pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_6(new pcl::PointCloud<pcl::PointXYZ>());
          Eigen::Matrix4d provas = ((t_s_bs_iso*(keyframe->odom)).matrix());
          pcl::transformPointCloud(*odomCloud, *temp_cloud_6, provas);
          pcl::toROSMsg(*temp_cloud_6, *t_cloud_msg);
          aligned->header.frame_id = "map";
          transformed_pub.publish(aligned);*/
          
          //Eigen::MatrixXd information_matrix = Eigen::MatrixXd::Identity(6, 6);
          //information_matrix.topLeftCorner(2, 2).array() /= private_nh.param<float>("building_edge_stddev_xy", 0.25);
          //information_matrix(2, 2) /= private_nh.param<float>("building_edge_stddev_z", 0.25);
          //information_matrix.bottomRightCorner(3, 3).array() /= private_nh.param<float>("building_edge_stddev_q", 1);
          
          // pc XYZI needed to compute the information matrix 
          pcl::PointCloud<PointT>::Ptr btempcloud(new pcl::PointCloud<PointT>);
          pcl::copyPointCloud(*buildingsCloud,*btempcloud); // convert pcl buildings pxyz to pxyzi
          pcl::PointCloud<PointT>::Ptr otempcloud(new pcl::PointCloud<PointT>);
          pcl::copyPointCloud(*odomCloud,*otempcloud); // convert pcl odom pxyz to pxyzi
          Eigen::MatrixXd information_matrix = inf_calclator->calc_information_matrix_buildings(btempcloud, otempcloud, t_s_bs_iso);
          //std::cout << "buildings inf: " << information_matrix << std::endl;
           /*Eigen::MatrixXd information_matrix = Eigen::MatrixXd::Identity(6, 6);
          information_matrix.topLeftCorner(3, 3).array() /= 0.000001;
          information_matrix.bottomRightCorner(3, 3).array() /= 0.000001;*/

          //Eigen::Isometry3d new_m = t_s_bs_iso*(keyframe->odom);
          //auto edge = graph_slam->add_se3_edge_prior(keyframe->node, new_m, information_matrix);
          //graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));
          
          /*Eigen::Matrix3d rot = new_m.block<3, 3>(0, 0);
          std::cout << "rot: " << rot << std::endl;
          Eigen::Quaterniond quat(rot); 
          Eigen::MatrixXd im = information_matrix.block<3, 3>(3, 3);
          std::cout << "im: " << im << std::endl;
          std::cout << "quat: " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
          auto edge = graph_slam->add_se3_prior_quat_edge(keyframe->node, quat, im);
          graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));

          Eigen::MatrixXd im2 = information_matrix.block<3, 3>(0, 0);
          std::cout << "im2: " << im2 << std::endl;
          Eigen::Vector3d trans = new_m.block<3, 1>(0, 3);
          std::cout << "trans: " << trans << std::endl;
          auto edge2 = graph_slam->add_se3_prior_xyz_edge(keyframe->node, trans, im2);
          graph_slam->add_robust_kernel(edge2, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));
          */ 

          //geometry_msgs::TransformStamped ts = matrix2transform(keyframe->stamp, (t_s_bs*keyframe->odom.matrix()).cast<float>(), "map", "ref_"+std::to_string(ii));
          //b_tf_broadcaster.sendTransform(ts);

          // add edges
          
          for(auto it1 = bnodes.begin(); it1 != bnodes.end(); it1++)
          {
            BuildingNode::Ptr bntemp = *it1;
            Eigen::Matrix4d t_bs_b = Eigen::Matrix4d::Identity();
            t_bs_b.block<3,1>(0, 3) = bntemp->local_origin;
            Eigen::Matrix4d t1 = t_s_bs * keyframe->odom.matrix(); 
            

            
            /*geometry_msgs::TransformStamped ts = matrix2transform(keyframe->stamp, (t_s_bs*keyframe->odom.matrix()).cast<float>(), map_frame_id, "local_"+bntemp->building.id);
            b_tf_pub.publish(ts);
            b_tf_broadcaster.sendTransform(ts);*/

            /*std::cout << "id: " << bntemp->building.id << std::endl;
            std::cout << "t_bs_b: " << t_bs_b << std::endl;
            std::cout << "t_s_bs: " << t_s_bs << std::endl;
            std::cout << "odom: " << keyframe->odom.matrix() << std::endl;
            std::cout << "t_s_bs*odom: " << t_s_bs*keyframe->odom.matrix() << std::endl;
            std::cout << "t_s_bs*odom inverse: " << ((t_s_bs*keyframe->odom.matrix()).inverse()) << std::endl;
            */
            Eigen::Matrix4d temp1 = t_s_bs*keyframe->odom.matrix();
            Eigen::Isometry3d temp1_iso;
            temp1_iso.matrix() = temp1;
            
            Eigen::Vector3d dist = temp1.block<3, 1>(0, 3) - bntemp->local_origin;
            //Eigen::Matrix4d t_s_b = temp1;
            //t_s_b.block<3, 1>(0, 3) = dist;
            Eigen::Matrix4d t_s_b = (t_bs_b.inverse())*temp1;
            Eigen::Isometry3d t_s_b_iso = Eigen::Isometry3d::Identity();
            //t_s_b_iso.translation() = t_s_b.block<3, 1>(0, 3);
            //t_s_b_iso.linear() = t_s_b.block<3, 3>(0, 0);
            t_s_b_iso.matrix() = t_s_b;

            /*std::cout << "t_s_b: " << t_s_b << std::endl;
            std::cout << "inverse: " << t_s_b.inverse() << std::endl;
            std::cout << "dist: " << dist << std::endl;*/

            geometry_msgs::TransformStamped ts2 = matrix2transform(keyframe->stamp,  t_bs_b.cast<float>(), "map", "local_"+bntemp->building.id);
            b_tf_broadcaster.sendTransform(ts2);

            //information_matrix = Eigen::MatrixXd::Zero(6, 6);
            //std::cout << "im: " << information_matrix << std::endl;

            //geometry_msgs::TransformStamped ts3 = matrix2transform(keyframe->stamp,  t_s_b.cast<float>(), "map", "yo_"+bntemp->building.id);
            //b_tf_broadcaster.sendTransform(ts3);
            
            //auto edge1 = graph_slam->add_se3_edge_prior(keyframe->node, temp1_iso, information_matrix);
            //graph_slam->add_robust_kernel(edge1, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));
            
            auto edge = graph_slam->add_se3_edge(bntemp->node, keyframe->node, t_s_b_iso, information_matrix);
            graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));
          }
          keyframe->buildings_nodes = bnodes;
        } else {
          std::cout << "No buildings found!" << std::endl;
          b_updated = false;
        }
      } 
    }
    //std::cout << "b_updated: " << b_updated << std::endl;
    return b_updated;
  }

  BuildingNode::Ptr get_building_node(Building b) {
    for(auto it = buildings.begin(); it != buildings.end(); it++)
    {
      BuildingNode::Ptr bntemp = *it;
      Building btemp = bntemp->building;
      if(btemp.id.compare(b.id) == 0)
        return bntemp;
    }
    return nullptr;
  }

  std::vector<Matrix> loadPoses(std::string file_name) {
    std::vector<Matrix> poses;
    FILE *fp = fopen(file_name.c_str(),"r");
    if (!fp)
      return poses;
    while (!feof(fp)) {
      Matrix P = Matrix::eye(4);
      if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                     &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                     &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                     &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
        poses.push_back(P);
      }
    }
    fclose(fp);
    return poses;
  }

  /**
   * @brief generate map point cloud and publish it
   * @param event
   */
  void map_points_publish_timer_callback(const ros::WallTimerEvent& event) {
    if(!map_points_pub.getNumSubscribers() || !graph_updated) {
      return;
    }

    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, map_cloud_resolution);
    if(!cloud) {
      return;
    }

    cloud->header.frame_id = map_frame_id;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*cloud, *cloud_msg);

    map_points_pub.publish(cloud_msg);
  }

  /**
   * @brief this methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
   * @param event
   */
  void optimization_timer_callback(const ros::WallTimerEvent& event) {
    //std::cout << "entered opt" << std::endl;
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    // add keyframes and floor coeffs in the queues to the pose graph
    bool keyframe_updated = flush_keyframe_queue();

    if(!keyframe_updated) {
      std_msgs::Header read_until;
      read_until.stamp = ros::Time::now() + ros::Duration(30, 0);
      read_until.frame_id = points_topic;
      read_until_pub.publish(read_until);
      read_until.frame_id = "/filtered_points";
      read_until_pub.publish(read_until);
    }

    //std::cout << "keyframe_updated: " << keyframe_updated << std::endl;
    if(!keyframe_updated & !flush_floor_queue() & !flush_gps_queue() & !flush_imu_queue() & !update_buildings_nodes()) {
      //std::cout << "no opt!" << std::endl;
      return;
    }

    // loop detection
    std::vector<Loop::Ptr> loops = loop_detector->detect(keyframes, new_keyframes, *graph_slam);
    for(const auto& loop : loops) {
      Eigen::Isometry3d relpose(loop->relative_pose.cast<double>());
      Eigen::MatrixXd information_matrix = inf_calclator->calc_information_matrix(loop->key1->cloud, loop->key2->cloud, relpose);
      auto edge = graph_slam->add_se3_edge(loop->key1->node, loop->key2->node, relpose, information_matrix);
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("loop_closure_edge_robust_kernel", "NONE"), private_nh.param<double>("loop_closure_edge_robust_kernel_size", 1.0));
    }

    std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
    new_keyframes.clear();

    // move the first node anchor position to the current estimate of the first node pose
    // so the first node moves freely while trying to stay around the origin
    if(anchor_node && private_nh.param<bool>("fix_first_node_adaptive", true)) {
      Eigen::Isometry3d anchor_target = static_cast<g2o::VertexSE3*>(anchor_edge->vertices()[1])->estimate();
      anchor_node->setEstimate(anchor_target);
    }

    // optimize the pose graph
    int num_iterations = private_nh.param<int>("g2o_solver_num_iterations", 1024);
    graph_slam->optimize(num_iterations);

    // publish tf
    const auto& keyframe = keyframes.back();
    Eigen::Isometry3d trans = keyframe->node->estimate() * keyframe->odom.inverse();
    trans_odom2map_mutex.lock();
    trans_odom2map = trans.matrix().cast<float>();
    trans_odom2map_mutex.unlock();

    std::vector<KeyFrameSnapshot::Ptr> snapshot(keyframes.size());
    std::transform(keyframes.begin(), keyframes.end(), snapshot.begin(), [=](const KeyFrame::Ptr& k) { return std::make_shared<KeyFrameSnapshot>(k); });

    keyframes_snapshot_mutex.lock();
    keyframes_snapshot.swap(snapshot);
    keyframes_snapshot_mutex.unlock();
    graph_updated = true;

    if(odom2map_pub.getNumSubscribers()) {
      geometry_msgs::TransformStamped ts = matrix2transform(keyframe->stamp, trans.matrix().cast<float>(), map_frame_id, odom_frame_id);
      odom2map_pub.publish(ts);
    }

    if(markers_pub.getNumSubscribers()) {
      auto markers = create_marker_array(ros::Time::now());
      markers_pub.publish(markers);
    }

    /********************************************************************************************/
    tf::StampedTransform transform_t;
    tf_listener.lookupTransform("base_link", "camera_gray_left", ros::Time(0), transform_t);
    Eigen::Quaterniond q;
    tf::quaternionTFToEigen(transform_t.getRotation(), q);
    Eigen::Vector3d v;
    tf::vectorTFToEigen (transform_t.getOrigin(), v);

    int index = -1, k = 0;
    while(k < keyframes.size()) {
      index = isIndexFromKeyframe(k);
      if(index >=0)
        break;
      k++;
    }
    if(index == -1)
      index = 0;

    Eigen::Isometry3d estimate = Eigen::Isometry3d::Identity();
    //estimate = keyframes[index]->node->estimate();
    
    trans_odom2map_mutex.lock();
    Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();
    estimate = odom2map;
    estimate.translation() = Eigen::Vector3d::Zero();
   

    Eigen::Isometry3d base_camera = Eigen::Isometry3d::Identity();
    base_camera.linear() = q.normalized().toRotationMatrix();
    //std::cout << "estimate: " << estimate.matrix() << std::endl;
    //std::cout << "base camera: " << base_camera.matrix() << std::endl;

    Eigen::Isometry3d tr = estimate*base_camera;

    Matrix tr_m = Matrix::eye(4);
    tr_m.val[0][0] = tr.linear()(0,0); 
    tr_m.val[0][1] = tr.linear()(0,1);
    tr_m.val[0][2] = tr.linear()(0,2);
    tr_m.val[0][3] = tr.translation()(0);
    tr_m.val[1][0] = tr.linear()(1,0);
    tr_m.val[1][1] = tr.linear()(1,1);
    tr_m.val[1][2] = tr.linear()(1,2);
    tr_m.val[1][3] = tr.translation()(1); 
    tr_m.val[2][0] = tr.linear()(2,0); 
    tr_m.val[2][1] = tr.linear()(2,1); 
    tr_m.val[2][2] = tr.linear()(2,2); 
    tr_m.val[2][3] = tr.translation()(2);
    //std::cout << "tr_m: " << tr_m << std::endl;

    /*******************DO NOT CANCEL******************************************************/
    // delta transforms gt to keyframes, used for visualization
    Matrix delta = Matrix::eye(4);
    delta = tr_m * Matrix::inv(gt[0]);
    //std::cout << "delta: " << delta << std::endl;
    
    // delta2 transforms keyframes to gt, used for error calculation
    Matrix delta_2 = Matrix::eye(4);
    delta_2 = gt[0] * Matrix::inv(tr_m);
    //std::cout << "delta_2: " << delta_2 << std::endl;
    
    std::ofstream myfile5;
    myfile5.open("align.txt");
    myfile5 << delta.val[0][0] << " " << delta.val[0][1] << " " << delta.val[0][2] << " " << delta.val[0][3] << " " << delta.val[1][0] << " " << delta.val[1][1] << " " << delta.val[1][2] << " " << delta.val[1][3] << " " << delta.val[2][0] << " " << delta.val[2][1] << " " << delta.val[2][2] << " " << delta.val[2][3] << "\n";
    myfile5 << delta_2.val[0][0] << " " << delta_2.val[0][1] << " " << delta_2.val[0][2] << " " << delta_2.val[0][3] << " " << delta_2.val[1][0] << " " << delta_2.val[1][1] << " " << delta_2.val[1][2] << " " << delta_2.val[1][3] << " " << delta_2.val[2][0] << " " << delta_2.val[2][1] << " " << delta_2.val[2][2] << " " << delta_2.val[2][3] << "\n";
    myfile5.close();

    //std::vector<Matrix> deltas = loadPoses("align.txt");
    //std::cout << "read delta: " << deltas[0] << std::endl; 
    //std::cout << "read delta_2: " << deltas[1] << std::endl; 
    //Matrix delta = deltas[0];
    //Matrix delta_2 = deltas[1];
    /****************************************************************************************/
    
    visualization_msgs::Marker gt_traj_marker;
    gt_traj_marker.header.frame_id = "map";
    gt_traj_marker.header.stamp = ros::Time::now();
    gt_traj_marker.ns = "gt";
    gt_traj_marker.id = 0;
    gt_traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    gt_traj_marker.action = visualization_msgs::Marker::ADD;

    gt_traj_marker.pose.orientation.w = 1.0;
    gt_traj_marker.scale.x = gt_traj_marker.scale.y = gt_traj_marker.scale.z = 0.5;

    for(int i = 0; i < gt.size(); i++) {
      geometry_msgs::Point p;

      p.x = (delta*gt[i]).val[0][3];
      p.y = (delta*gt[i]).val[1][3];
      p.z = (delta*gt[i]).val[2][3];
      gt_traj_marker.points.push_back(p);
      std_msgs::ColorRGBA c;
      c.r = 1.0;
      c.g = 1.0;
      c.b = 1.0;
      c.a = 1.0;
      gt_traj_marker.colors.push_back(c);
    }

    gt_markers_pub.publish(gt_traj_marker);
    /***************************************************************************************/
    std::ofstream myfile;
    myfile.open ("poses.txt");
    std::ofstream myfile2;
    myfile2.open ("gt.txt");
    std::ofstream myfile3;
    myfile3.open ("dist.txt");
    std::ofstream myfile4;
    myfile4.open ("stats.txt");
    std::cout << "dist" << std::endl;
    float sum = 0.0;
    float sum_sq = 0.0;
    int j = 0;
    for(int i = 0; i < gt.size(); i++) {
      int index = isIndexFromKeyframe(i);
      if(index >= 0) {
        j++;
        Eigen::Isometry3d pose = keyframes[index]->node->estimate();
        
        Matrix pose_matrix = Matrix::eye(4);
        pose_matrix.val[0][0] = pose.linear()(0,0); 
        pose_matrix.val[0][1] = pose.linear()(0,1);
        pose_matrix.val[0][2] = pose.linear()(0,2);
        pose_matrix.val[0][3] = pose.translation()(0);
        pose_matrix.val[1][0] = pose.linear()(1,0);
        pose_matrix.val[1][1] = pose.linear()(1,1);
        pose_matrix.val[1][2] = pose.linear()(1,2);
        pose_matrix.val[1][3] = pose.translation()(1); 
        pose_matrix.val[2][0] = pose.linear()(2,0); 
        pose_matrix.val[2][1] = pose.linear()(2,1); 
        pose_matrix.val[2][2] = pose.linear()(2,2); 
        pose_matrix.val[2][3] = pose.translation()(2);
        
        Matrix pose_trans = delta_2*pose_matrix;
        Matrix pose_error = Matrix::inv(gt[i])*pose_trans;
        //std::cout << "pose_error: " << pose_error << std::endl;

        float dx = pose_error.val[0][3];
        float dy = pose_error.val[1][3];
        float dz = pose_error.val[2][3];
        float dist = sqrt(dx*dx+dy*dy+dz*dz);
        sum += dist;
        sum_sq += (dx*dx+dy*dy+dz*dz);
        std::cout << dist << " " << dx << " " << dy << " " << dz << std::endl;
        myfile3 << i << " " << dist << std::endl;
        myfile << pose_trans.val[0][0] << " " << pose_trans.val[0][1] << " " << pose_trans.val[0][2] << " " << pose_trans.val[0][3] << " " << pose_trans.val[1][0] << " " << pose_trans.val[1][1] << " " << pose_trans.val[1][2] << " " << pose_trans.val[1][3] << " " << pose_trans.val[2][0] << " " << pose_trans.val[2][1] << " " << pose_trans.val[2][2] << " " << pose_trans.val[2][3] << "\n";
      
        myfile2 << gt[i].val[0][0] << " " << gt[i].val[0][1] << " " << gt[i].val[0][2] << " " << gt[i].val[0][3] << " " << gt[i].val[1][0] << " " << gt[i].val[1][1] << " " << gt[i].val[1][2] << " " << gt[i].val[1][3] << " " << gt[i].val[2][0] << " " << gt[i].val[2][1] << " " << gt[i].val[2][2] << " " << gt[i].val[2][3] << "\n";
      }  
    }
    float t_mean = sum/(j);
    float t_variance = sqrt((sum_sq/(j))-(t_mean*t_mean));

    myfile4 << t_mean << " " << t_variance << std::endl;

    myfile.close();
    myfile2.close();
    myfile3.close();
    myfile4.close();
    std::cout << "finished" << std::endl;

    /****************************************************************************************/

  }

  int isIndexFromKeyframe(int index) {
    //std::cout << "search " << index << std::endl;
    for(int i = 0; i < keyframes.size(); i++) {
      //std::cout << "i " << i << std::endl;
      //std::cout << "internal index: " << keyframes[i]->index << std::endl;
      if(keyframes[i]->index==index) {
        //std::cout << "found" << std::endl;
        return i;
      }
    }
    //std::cout << "not found" << std::endl;
    return -1;
  }

  /**
   * @brief create visualization marker
   * @param stamp
   * @return
   */
  visualization_msgs::MarkerArray create_marker_array(const ros::Time& stamp) const {
    visualization_msgs::MarkerArray markers;
    markers.markers.resize(4);

    // node markers
    visualization_msgs::Marker& traj_marker = markers.markers[0];
    traj_marker.header.frame_id = "map";
    traj_marker.header.stamp = stamp;
    traj_marker.ns = "nodes";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.5;

    visualization_msgs::Marker& imu_marker = markers.markers[1];
    imu_marker.header = traj_marker.header;
    imu_marker.ns = "imu";
    imu_marker.id = 1;
    imu_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    imu_marker.pose.orientation.w = 1.0;
    imu_marker.scale.x = imu_marker.scale.y = imu_marker.scale.z = 0.75;

    traj_marker.points.resize(keyframes.size() + buildings.size());
    traj_marker.colors.resize(keyframes.size() + buildings.size());
   
    /***********************************************************************/
    /*Eigen::Matrix4d gt_m = Eigen::Matrix4d::Identity(); 
    gt_m(0,0) = gt[0].val[0][0];
    gt_m(0,1) = gt[0].val[0][1];
    gt_m(0,2) = gt[0].val[0][2];
    gt_m(0,3) = gt[0].val[0][3];
    gt_m(1,0) = gt[0].val[1][0];
    gt_m(1,1) = gt[0].val[1][1];
    gt_m(1,2) = gt[0].val[1][2];
    gt_m(1,3) = gt[0].val[1][3];
    gt_m(2,0) = gt[0].val[2][0];
    gt_m(2,1) = gt[0].val[2][1];
    gt_m(2,2) = gt[0].val[2][2];
    gt_m(2,3) = gt[0].val[2][3];

    tf::StampedTransform transform_t;
    tf_listener.lookupTransform("base_link", "camera_gray_left", ros::Time(0), transform_t);
    Eigen::Quaterniond q;
    tf::quaternionTFToEigen(transform_t.getRotation(), q);
    Eigen::Vector3d v;
    tf::vectorTFToEigen (transform_t.getOrigin(), v);
    Eigen::Matrix3d rot = q.normalized().toRotationMatrix();
    //std::cout << "base_link -> camera_gray_left trans: " << v << std::endl;
    //std::cout << "base_link -> camera_gray_left rot: " << rot << std::endl;

    Eigen::Isometry3d estimate = keyframes[0]->node->estimate();
    Eigen::Matrix4d estimate_m = Eigen::Matrix4d::Identity();
    estimate_m.block<3, 1>(0, 3) = estimate.translation();
    estimate_m.block<3, 3>(0, 0) = estimate.linear();
    Eigen::Matrix4d base_camera_m = Eigen::Matrix4d::Identity();
    base_camera_m.block<3, 1>(0, 3) = v;
    base_camera_m.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();*/
    /*******************************************************************************/

    for(int i = 0; i < keyframes.size(); i++) {
      Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
      traj_marker.points[i].x = pos.x();
      traj_marker.points[i].y = pos.y();
      traj_marker.points[i].z = pos.z();

      /***************************************************************************************/
      /*Eigen::Matrix4d temp = Eigen::Matrix4d::Identity();
      temp.block<3, 1>(0, 3) = keyframes[i]->node->estimate().translation();
      temp.block<3, 3>(0, 0) = keyframes[i]->node->estimate().linear();
      
      Eigen::Matrix4d pos = gt_m*(base_camera_m.inverse())*(estimate_m.inverse())*temp;

      traj_marker.points[i].x = pos(0, 3);
      traj_marker.points[i].y = pos(1, 3);
      traj_marker.points[i].z = pos(2, 3);*/
      /**************************************************************************************/

      double p = static_cast<double>(i) / keyframes.size();
      traj_marker.colors[i].r = 1.0 - p;
      traj_marker.colors[i].g = p;
      traj_marker.colors[i].b = 0.0;
      traj_marker.colors[i].a = 1.0;

      if(keyframes[i]->acceleration) {
        Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
        geometry_msgs::Point point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();

        std_msgs::ColorRGBA color;
        color.r = 0.0;
        color.g = 0.0;
        color.b = 1.0;
        color.a = 0.1;

        imu_marker.points.push_back(point);
        imu_marker.colors.push_back(color);
      }
    }

     for(int i = 0; i < buildings.size(); i++) {
       if(buildings[i]->node != nullptr) {
        
        Eigen::Vector3d pos = buildings[i]->node->estimate().translation();
        traj_marker.points[i+keyframes.size()].x = pos.x();
        traj_marker.points[i+keyframes.size()].y = pos.y();
        traj_marker.points[i+keyframes.size()].z = pos.z();

        traj_marker.colors[i+keyframes.size()].r = 143.0/255.0;
        traj_marker.colors[i+keyframes.size()].g = 0.0/255.0;
        traj_marker.colors[i+keyframes.size()].b = 255.0/255.0;
        traj_marker.colors[i+keyframes.size()].a = 1.0;
      }
     
    }
    
    // edge markers
    visualization_msgs::Marker& edge_marker = markers.markers[2];
    edge_marker.header.frame_id = "map";
    edge_marker.header.stamp = stamp;
    edge_marker.ns = "edges";
    edge_marker.id = 2;
    edge_marker.type = visualization_msgs::Marker::LINE_LIST;

    edge_marker.pose.orientation.w = 1.0;
    edge_marker.scale.x = 0.05;

    edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
    edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

    auto edge_itr = graph_slam->graph->edges().begin();
    for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
      g2o::HyperGraph::Edge* edge = *edge_itr;
      g2o::EdgeSE3* edge_se3 = dynamic_cast<g2o::EdgeSE3*>(edge);
      if(edge_se3) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[0]);
        g2o::VertexSE3* v2 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[1]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = v2->estimate().translation();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
        double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();
        edge_marker.colors[i * 2].r = 1.0 - p1;
        edge_marker.colors[i * 2].g = p1;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0 - p2;
        edge_marker.colors[i * 2 + 1].g = p2;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        /*if(std::abs(v1->id() - v2->id()) > 2) {
          edge_marker.points[i * 2].z += 0.5;
          edge_marker.points[i * 2 + 1].z += 0.5;
        }*/

        continue;
      }

      g2o::EdgeSE3Plane* edge_plane = dynamic_cast<g2o::EdgeSE3Plane*>(edge);
      if(edge_plane) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_plane->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2(pt1.x(), pt1.y(), 0.0);

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].b = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].b = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXY* edge_priori_xy = dynamic_cast<g2o::EdgeSE3PriorXY*>(edge);
      if(edge_priori_xy) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xy->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
        pt2.head<2>() = edge_priori_xy->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z() + 0.5;

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXYZ* edge_priori_xyz = dynamic_cast<g2o::EdgeSE3PriorXYZ*>(edge);
      if(edge_priori_xyz) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xyz->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = edge_priori_xyz->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        //edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].r = 192.0/255.0;
        edge_marker.colors[i * 2].g = 192.0/255.0;
        edge_marker.colors[i * 2].b = 192.0/255.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 192.0/255.0;
        edge_marker.colors[i * 2 + 1].g = 192.0/255.0;
        edge_marker.colors[i * 2 + 1].b = 192.0/255.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3Prior* edge_priori = dynamic_cast<g2o::EdgeSE3Prior*>(edge);
      if(edge_priori) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Isometry3d pt2 = edge_priori->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        //edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.translation().x();
        edge_marker.points[i * 2 + 1].y = pt2.translation().y();
        edge_marker.points[i * 2 + 1].z = pt2.translation().z();

        edge_marker.colors[i * 2].r = 255.0/255.0;
        edge_marker.colors[i * 2].g = 20.0/255.0;
        edge_marker.colors[i * 2].b = 147.0/255.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 255.0/255.0;
        edge_marker.colors[i * 2 + 1].g = 20.0/255.0;
        edge_marker.colors[i * 2 + 1].b = 147.0/255.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }
    }

    // sphere
    visualization_msgs::Marker& sphere_marker = markers.markers[3];
    sphere_marker.header.frame_id = "map";
    sphere_marker.header.stamp = stamp;
    sphere_marker.ns = "loop_close_radius";
    sphere_marker.id = 3;
    sphere_marker.type = visualization_msgs::Marker::SPHERE;

    if(!keyframes.empty()) {
      Eigen::Vector3d pos = keyframes.back()->node->estimate().translation();
      sphere_marker.pose.position.x = pos.x();
      sphere_marker.pose.position.y = pos.y();
      sphere_marker.pose.position.z = pos.z();
    }
    sphere_marker.pose.orientation.w = 1.0;
    sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = loop_detector->get_distance_thresh() * 2.0;

    sphere_marker.color.r = 1.0;
    sphere_marker.color.a = 0.3;
    return markers;
  }

  /**
   * @brief dump all data to the current directory
   * @param req
   * @param res
   * @return
   */
  bool dump_service(hdl_graph_slam::DumpGraphRequest& req, hdl_graph_slam::DumpGraphResponse& res) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    std::string directory = req.destination;

    if(directory.empty()) {
      std::array<char, 64> buffer;
      buffer.fill(0);
      time_t rawtime;
      time(&rawtime);
      const auto timeinfo = localtime(&rawtime);
      strftime(buffer.data(), sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
    }

    if(!boost::filesystem::is_directory(directory)) {
      boost::filesystem::create_directory(directory);
    }

    std::cout << "all data dumped to:" << directory << std::endl;

    graph_slam->save(directory + "/graph.g2o");
    for(int i = 0; i < keyframes.size(); i++) {
      std::stringstream sst;
      sst << boost::format("%s/%06d") % directory % i;

      keyframes[i]->save(sst.str());
    }

    if(zero_utm) {
      std::ofstream zero_utm_ofs(directory + "/zero_utm");
      zero_utm_ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    std::ofstream ofs(directory + "/special_nodes.csv");
    ofs << "anchor_node " << (anchor_node == nullptr ? -1 : anchor_node->id()) << std::endl;
    ofs << "anchor_edge " << (anchor_edge == nullptr ? -1 : anchor_edge->id()) << std::endl;
    ofs << "floor_node " << (floor_plane_node == nullptr ? -1 : floor_plane_node->id()) << std::endl;

    res.success = true;
    return true;
  }

  /**
   * @brief save map data as pcd
   * @param req
   * @param res
   * @return
   */
  bool save_map_service(hdl_graph_slam::SaveMapRequest& req, hdl_graph_slam::SaveMapResponse& res) {
    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, req.resolution);
    if(!cloud) {
      res.success = false;
      return true;
    }

    if(zero_utm && req.utm) {
      for(auto& pt : cloud->points) {
        pt.getVector3fMap() += (*zero_utm).cast<float>();
      }
    }

    cloud->header.frame_id = map_frame_id;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    if(zero_utm) {
      std::ofstream ofs(req.destination + ".utm");
      ofs << (*zero_utm).transpose() << std::endl;
    }

    int ret = pcl::io::savePCDFileBinary(req.destination, *cloud);
    res.success = ret == 0;

    return true;
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;
  ros::WallTimer optimization_timer;
  ros::WallTimer map_publish_timer;

  std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

  ros::Subscriber gps_sub;
  ros::Subscriber nmea_sub;
  ros::Subscriber navsat_sub;

  ros::Subscriber imu_sub;
  ros::Subscriber floor_sub;

  ros::Publisher markers_pub;

  std::string map_frame_id;
  std::string odom_frame_id;

  std::mutex trans_odom2map_mutex;
  Eigen::Matrix4f trans_odom2map;
  ros::Publisher odom2map_pub;

  std::string points_topic;
  ros::Publisher read_until_pub;
  ros::Publisher map_points_pub;

  tf::TransformListener tf_listener;

  ros::ServiceServer dump_service_server;
  ros::ServiceServer save_map_service_server;

  // keyframe queue
  std::string base_frame_id;
  std::mutex keyframe_queue_mutex;
  std::deque<KeyFrame::Ptr> keyframe_queue;

  // gps queue
  double gps_time_offset;
  double gps_edge_stddev_xy;
  double gps_edge_stddev_z;
  boost::optional<Eigen::Vector3d> zero_utm;
  std::mutex gps_queue_mutex;
  std::deque<geographic_msgs::GeoPointStampedConstPtr> gps_queue;

  // imu queue
  double imu_time_offset;
  bool enable_imu_orientation;
  double imu_orientation_edge_stddev;
  bool enable_imu_acceleration;
  double imu_acceleration_edge_stddev;
  std::mutex imu_queue_mutex;
  std::deque<sensor_msgs::ImuConstPtr> imu_queue;

  // floor_coeffs queue
  double floor_edge_stddev;
  std::mutex floor_coeffs_queue_mutex;
  std::deque<hdl_graph_slam::FloorCoeffsConstPtr> floor_coeffs_queue;

  // for map cloud generation
  std::atomic_bool graph_updated;
  double map_cloud_resolution;
  std::mutex keyframes_snapshot_mutex;
  std::vector<KeyFrameSnapshot::Ptr> keyframes_snapshot;
  std::unique_ptr<MapCloudGenerator> map_cloud_generator;

  // graph slam
  // all the below members must be accessed after locking main_thread_mutex
  std::mutex main_thread_mutex;

  int max_keyframes_per_update;
  std::deque<KeyFrame::Ptr> new_keyframes;

  g2o::VertexSE3* anchor_node;
  g2o::EdgeSE3* anchor_edge;
  g2o::VertexPlane* floor_plane_node;
  std::vector<KeyFrame::Ptr> keyframes;
  std::unordered_map<ros::Time, KeyFrame::Ptr, RosTimeHash> keyframe_hash;

  std::unique_ptr<GraphSLAM> graph_slam;
  std::unique_ptr<LoopDetector> loop_detector;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  std::unique_ptr<NmeaSentenceParser> nmea_parser;

  std::unique_ptr<InformationMatrixCalculator> inf_calclator;

  // map 
  std::vector<BuildingNode::Ptr> buildings;
  float lidar_range;
  int enter;
  int zero_utm_zone;
  char zero_utm_band;
  ros::Publisher buildings_pub;
  ros::Publisher odom_pub;
  ros::Publisher transformed_pub;
  ros::Publisher original_odom_pub;
  double ground_floor_max_thresh;
  double radius_search;
  int min_neighbors_in_radius;
  int ii;
  std::vector<Matrix> gt;
  ros::Publisher gt_markers_pub;
  bool first_guess;
  Eigen::Matrix4f prev_guess;
  Eigen::Matrix4f prev_pose;
  ros::Publisher b_tf_pub;
  tf::TransformBroadcaster b_tf_broadcaster;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::HdlGraphSlamNodelet, nodelet::Nodelet)
