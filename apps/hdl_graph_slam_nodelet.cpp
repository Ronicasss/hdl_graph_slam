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

#include <g2o/edge_se2_priorxy.hpp>
#include <g2o/edge_se2_priorquat.hpp>

#include "hdl_graph_slam/building_tools.hpp"
#include "hdl_graph_slam/building_node.hpp"
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
#include <g2o/types/slam2d/edge_se2.h>
#include <g2o/types/slam2d/vertex_se2.h>

namespace hdl_graph_slam {

class HdlGraphSlamNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZI PointT;
  typedef pcl::PointXYZ PointT3;
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
    estimated_buildings_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/hdl_graph_slam/estimated_buildings", 32);
    original_odom_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/hdl_graph_slam/original_odom_cloud", 32);

    lidar_range = private_nh.param<float>("lidar_range", 300);
    ground_floor_max_thresh = private_nh.param<double>("ground_floor_max_thresh", 0.5);
    radius_search = private_nh.param<double>("radius_search", 1);
    min_neighbors_in_radius = private_nh.param<double>("min_neighbors_in_radius", 100);
    enter = private_nh.param<bool>("enable_buildings", true);
    ii = -1;
    //read ground truth
    gt = loadPoses(private_nh.param<std::string>("gt_path", ""));
    std::cout << "gt poses loaded: " << gt.size() << std::endl;
    gt_markers_pub = mt_nh.advertise<visualization_msgs::Marker>("/hdl_graph_slam/gt_markers", 16);
    first_guess = true;
    prev_guess = Eigen::Matrix4f::Identity();
    fix_first_building = true;

    //
    anchor_node = nullptr;
    anchor_edge = nullptr;
    graph_slam.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
    keyframe_updater.reset(new KeyframeUpdater(private_nh));
    loop_detector.reset(new LoopDetector(private_nh));
    map_cloud_generator.reset(new MapCloudGenerator());
    inf_calclator.reset(new InformationMatrixCalculator(private_nh));
    nmea_parser.reset(new NmeaSentenceParser());

    gps_time_offset = private_nh.param<double>("gps_time_offset", 0.0);
    gps_edge_stddev_xy = private_nh.param<double>("gps_edge_stddev_xy", 10000.0);
    
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
    Eigen::Isometry2d odom = odom2isometry2d(odom_msg);
    //std::cout << "odom: " << odom.matrix() << std::endl;

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
    Eigen::Isometry2d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();

    int num_processed = 0;
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); i++) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      // new_keyframes will be tested later for loop closure
      new_keyframes.push_back(keyframe);

      // add pose node
      Eigen::Isometry2d odom = odom2map * keyframe->odom;
      keyframe->node = graph_slam->add_se2_node(odom);
      keyframe_hash[keyframe->stamp] = keyframe;

      // fix the first node
      if(keyframes.empty() && new_keyframes.size() == 1) {
        if(private_nh.param<bool>("fix_first_node", false)) {
          Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(3, 3);
          std::stringstream sst(private_nh.param<std::string>("fix_first_node_stddev", "1 1 1"));
          for(int i = 0; i < 3; i++) {
            double stddev = 1.0;
            sst >> stddev;
            inf(i, i) = 1.0 / stddev;
          }
          std::cout << "fixed first keyframe" << std::endl;
          anchor_node = graph_slam->add_se2_node(Eigen::Isometry2d::Identity());
          anchor_node->setFixed(true);
          anchor_edge = graph_slam->add_se2_edge(anchor_node, keyframe->node, Eigen::Isometry2d::Identity(), inf);
        }
      }

      if(i == 0 && keyframes.empty()) {
        continue;
      }

      // add edge between consecutive keyframes
      const auto& prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];

      Eigen::Isometry2d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;
      Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, isometry2dto3d(relative_pose));
      Eigen::Matrix3d inf_3d = Eigen::Matrix3d::Identity();
      inf_3d.block<2,2>(0,0) = information.block<2,2>(0,0);
      inf_3d(2,2) = information(5,5);
      std::cout << "keyframe inf: " << inf_3d << std::endl;
      auto edge = graph_slam->add_se2_edge(keyframe->node, prev_keyframe->node, relative_pose, inf_3d);
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
    //std::cout << "gps queue" << std::endl;

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
      Eigen::Vector2d xyz(utm.easting, utm.northing);

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

      if(private_nh.param<bool>("test_enable_gps_imu", true)) {
        g2o::OptimizableGraph::Edge* edge;
        
        Eigen::Matrix2d information_matrix = Eigen::Matrix2d::Identity() / gps_edge_stddev_xy;
        edge = graph_slam->add_se2_prior_xy_edge(keyframe->node, xyz.head<2>(), information_matrix);
        
        graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("gps_edge_robust_kernel", "NONE"), private_nh.param<double>("gps_edge_robust_kernel_size", 1.0));
        
        updated = true; 
      }
    }

    auto remove_loc = std::upper_bound(gps_queue.begin(), gps_queue.end(), keyframes.back()->stamp, [=](const ros::Time& stamp, const geographic_msgs::GeoPointStampedConstPtr& geopoint) { return stamp < geopoint->header.stamp; });
    gps_queue.erase(gps_queue.begin(), remove_loc);
    //std::cout << "gps updated: " << updated << std::endl;
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
    //std::cout << "imu queue" << std::endl;
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
      
      keyframe->acceleration = Eigen::Vector2d(acc_base.vector.x, acc_base.vector.y);
      keyframe->orientation = Eigen::Rotation2D<double>(quatToAngle(Eigen::Quaterniond(quat_base.quaternion.w, quat_base.quaternion.x, quat_base.quaternion.y, quat_base.quaternion.z)));
      
       
      if(private_nh.param<bool>("test_enable_gps_imu", true)) {
        if(enable_imu_orientation) {
          Eigen::MatrixXd info = Eigen::MatrixXd::Identity(3, 3) / imu_orientation_edge_stddev;
          auto edge = graph_slam->add_se2_prior_quat_edge(keyframe->node, *keyframe->orientation, info);
          graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("imu_orientation_edge_robust_kernel", "NONE"), private_nh.param<double>("imu_orientation_edge_robust_kernel_size", 1.0));
        }
        
        updated = true;
      }
    }

    auto remove_loc = std::upper_bound(imu_queue.begin(), imu_queue.end(), keyframes.back()->stamp, [=](const ros::Time& stamp, const sensor_msgs::ImuConstPtr& imu) { return stamp < imu->header.stamp; });
    imu_queue.erase(imu_queue.begin(), remove_loc);
    //std::cout << "imu updated: " << updated << std::endl;
    return updated;
  }

  bool update_buildings_nodes() {
    bool b_updated = false;
    bool new_kf = false;
    if(enter) {
      for(auto& keyframe : keyframes) {
        // if the keyframe is never been aligned with map
        if(!keyframe->buildings_check && keyframe->index != 0) {
          new_kf = true;
          std::cout << "update_buildings_nodes" << std::endl;
          if(first_guess && !keyframe->utm_coord) {
            continue; 
          }
          keyframe->buildings_check = true;
          //enter = 0;
          /***************************************************************************************/
          // pre-processing on odom cloud
          pcl::PointCloud<PointT3>::Ptr odomCloud(new pcl::PointCloud<PointT3>); // cloud containing lidar data
          pcl::PointCloud<PointT3>::Ptr temp_cloud(new pcl::PointCloud<PointT3>);
          pcl::PointCloud<PointT3>::Ptr temp_cloud_2(new pcl::PointCloud<PointT3>);
          pcl::PointCloud<PointT3>::Ptr temp_cloud_3(new pcl::PointCloud<PointT3>);
          pcl::PointCloud<PointT3>::Ptr temp_cloud_4(new pcl::PointCloud<PointT3>);
          pcl::PointCloud<PointT3>::Ptr temp_cloud_5(new pcl::PointCloud<PointT3>);
          pcl::copyPointCloud(*keyframe->cloud,*temp_cloud); // convert from pointxyzi to pointxyz

          //std::cout << "size 1: " << temp_cloud->size() << std::endl; 
          // height filtering
          pcl::PassThrough<PointT3> pass;
          pass.setInputCloud (temp_cloud);
          pass.setFilterFieldName ("z");
          pass.setFilterLimits (ground_floor_max_thresh, 100.0);
          pass.filter(*temp_cloud_2);
          temp_cloud_2->header = (*keyframe->cloud).header;
          //std::cout << "size 2: " << temp_cloud_2->size() << std::endl;
          // downsampling
          pcl::Filter<PointT3>::Ptr downsample_filter;
          double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
          boost::shared_ptr<pcl::VoxelGrid<PointT3>> voxelgrid(new pcl::VoxelGrid<PointT3>());
          voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
          downsample_filter = voxelgrid;
          downsample_filter->setInputCloud(temp_cloud_2);
          downsample_filter->filter(*temp_cloud_3);
          temp_cloud_3->header = temp_cloud_2->header;
          //std::cout << "size 3: " << temp_cloud_3->size() << std::endl;
          // outlier removal
          pcl::RadiusOutlierRemoval<PointT3>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT3>());
          rad->setRadiusSearch(radius_search);
          rad->setMinNeighborsInRadius(min_neighbors_in_radius);
          //std::cout << "rad: " << rad->getRadiusSearch() << " neighbors: " << rad->getMinNeighborsInRadius() << std::endl; 
          rad->setInputCloud(temp_cloud_3);
          rad->filter(*temp_cloud_4);
          temp_cloud_4->header = temp_cloud_3->header;
  
          // project the cloud on plane z=0
          pcl::ProjectInliers<PointT3> proj;
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
          
          pcl::transformPointCloud(*temp_cloud_5, *odomCloud, isometry2dto3d(keyframe->odom).matrix());
          odomCloud->header = temp_cloud_5->header;
          odomCloud->header.frame_id = "base_link";
          
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
          /***************************************************************************************/
          Eigen::Vector4f c;
          pcl::compute3DCentroid(*odomCloud, c);
          pcl::PointXYZ cpcl;
          cpcl.x = c(0);
          cpcl.y = c(1);
          cpcl.z = c(2);
          double sum = 0.0;
          for(int i = 0; i < odomCloud->size(); i++) {
            pcl::PointXYZ ptemp = odomCloud->at(i);
            double dist = pcl::euclideanDistance(cpcl, ptemp);
            sum += dist;
          }
          Eigen::Matrix<double, 3, 3> covariance_matrix; 
          Eigen::Matrix<double, 4, 1> centroid;
          pcl::computeMeanAndCovarianceMatrix(*odomCloud, covariance_matrix, centroid);
          
          //std::cout << "centroid 1: " << c << std::endl;
          //std::cout << "centroid 2: " << centroid << std::endl;

          double dist = (sum/(odomCloud->size()));
          
          lidar_range = dist + 5;
          
          /*if(keyframe->index > 100)
              lidar_range = 20;
          else
              lidar_range = 5;  */
          std::cout << "lidar range: " << lidar_range << std::endl;  
          /***************************************************************************************/  
          
          Eigen::Vector2d e_utm_coord = Eigen::Vector2d::Zero();
          int zone = 0;
          char band;
          if(!first_guess) {
            std::cout << "using est" << std::endl; 
            e_utm_coord = keyframe->node->estimate().translation(); 
            zone = zero_utm_zone;
            band = zero_utm_band;
          } else {
            std::cout << "first guess" << std::endl; 
            e_utm_coord = (*keyframe->utm_coord);
            zone = *keyframe->utm_zone;
            band = *keyframe->utm_band;
          }
         
          Eigen::Vector2d e_zero_utm = (*zero_utm);
          geodesy::UTMPoint utm;
          // e_utm_coord are the coords of current keyframe wrt zero_utm, so to get real coords we add zero_utm
          utm.easting = e_utm_coord(0) + e_zero_utm(0);
          utm.northing = e_utm_coord(1) + e_zero_utm(1);
          utm.altitude = 0;
          utm.zone = zone;
          utm.band = band;
          geographic_msgs::GeoPoint lla = geodesy::toMsg(utm); // convert from utm to lla

          // download and parse buildings
          std::vector<Building> new_buildings = BuildingTools::getBuildings(lla.latitude, lla.longitude, lidar_range, e_zero_utm, private_nh.param<std::string>("buildings_host", "https://overpass-api.de"));
          if(new_buildings.size() > 0) {
            std::cout << "We found buildings! " << keyframe->index << std::endl;
            b_updated = true;
   
            std::vector<BuildingNode::Ptr> bnodes; // vector containing all buildings nodes for current kf (new and not new)
            // buildingsCloud is the cloud containing all buildings
            pcl::PointCloud<PointT3>::Ptr buildingsCloud(new pcl::PointCloud<PointT3>);
           
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
                Eigen::Vector2d T = bt->local_origin; // local origin is already referring to zero_utm
                // rotation
                Eigen::Matrix2d R = Eigen::Matrix2d::Identity(); // gps coords don't give orientation
                // rototranslation
                Eigen::Isometry2d A;
                A.linear() = R;
                A.translation() = T;
                // set the node
                bt->node = graph_slam->add_se2_node(A); 
                if(fix_first_building) {
                  if(private_nh.param<bool>("fix_first_building", true)) {
                    //std::cout << "fixed building!" << std::endl;
                    //bt->node->setFixed(true);
                    //keyframe->node->setFixed(true);

                    /********************************************************************/
                    /*Eigen::MatrixXd info = Eigen::MatrixXd::Identity(1, 1) / imu_orientation_edge_stddev;
                    auto edge = graph_slam->add_se2_prior_quat_edge(keyframe->node, *keyframe->orientation, info);
                    graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("imu_orientation_edge_robust_kernel", "NONE"), private_nh.param<double>("imu_orientation_edge_robust_kernel_size", 1.0));
                    
                    g2o::OptimizableGraph::Edge* edge2;
        
                    Eigen::Matrix2d information_matrix = Eigen::Matrix2d::Identity() / gps_edge_stddev_xy;
                    edge2 = graph_slam->add_se2_prior_xy_edge(keyframe->node, *keyframe->utm_coord, information_matrix);
                    
                    graph_slam->add_robust_kernel(edge2, private_nh.param<std::string>("gps_edge_robust_kernel", "NONE"), private_nh.param<double>("gps_edge_robust_kernel_size", 1.0));
                    
                    std::cout << "gps info: " << information_matrix << std::endl;
                    std::cout << "quat info: " << info<< std::endl;*/
                    /************************************************************************/
                  }
                  //fix_first_building = false;
                }
              
                //std::cout << "id: " << bt->building.id << " est: " << bt->node->estimate().translation() << std::endl;
                buildings.push_back(bt);
                bnodes.push_back(bt);
              } else {
                //std::cout << "id: " << bntemp->building.id << " est: " << bntemp->node->estimate().translation() << std::endl;
                bnodes.push_back(bntemp);
              }
            }
            
            buildingsCloud->header.frame_id = "map";
            // publish buildings cloud
            sensor_msgs::PointCloud2Ptr b_cloud_msg(new sensor_msgs::PointCloud2());
            pcl::toROSMsg(*buildingsCloud, *b_cloud_msg);
            b_cloud_msg->header.frame_id = "map";
            b_cloud_msg->header.stamp = keyframe->stamp;
            buildings_pub.publish(b_cloud_msg);

            Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
            if(first_guess) {
              std::cout << "first guess" << std::endl;
              guess.block<2,1>(0,3) = e_utm_coord.cast<float>();
              guess.block<2,2>(0,0) = (*(keyframe->orientation)).cast<float>().toRotationMatrix();
              guess = guess * ((isometry2dto3d(keyframe->odom)).cast<float>().matrix()).inverse();
              first_guess = false;
            } else {
              std::cout << "prev guess" << std::endl;
              guess = prev_guess;
            }
            std::cout << "guess: " << guess << std::endl;

            // gicp_omp registration
            pcl::Registration<PointT3, PointT3>::Ptr registration;

            pcl::registration::WarpPointRigid3D<PointT3, PointT3>::Ptr warp_fcn(new pcl::registration::WarpPointRigid3D<PointT3,PointT3>);
            pcl::registration::TransformationEstimationLM<PointT3, PointT3>::Ptr te(new pcl::registration::TransformationEstimationLM<PointT3, PointT3>);
            te->setWarpFunction(warp_fcn);

            /*std::cout << "registration: FAST_VGICP" << std::endl;
            boost::shared_ptr<fast_gicp::FastVGICP<PointT3, PointT3>> gicp(new fast_gicp::FastVGICP<PointT3, PointT3>());
            gicp->setNumThreads(private_nh.param<int>("gicp_reg_num_threads", 0));
            if(keyframe->index >= (private_nh.param<int>("gicp_init_res_kf_thresh", 35)))
              gicp->setResolution(private_nh.param<double>("gicp_reg_resolution", 1.0));
            else
              gicp->setResolution(private_nh.param<double>("gicp_initial_reg_resolution", 2.0));*/
            
            
            //std::cout << "registration: FAST_GICP" << std::endl;
            //boost::shared_ptr<fast_gicp::FastGICP<PointT3, PointT3>> gicp(new fast_gicp::FastGICP<PointT3, PointT3>());
            //gicp->setNumThreads(private_nh.param<int>("reg_num_threads", 0));

            std::cout << "registration: GICP_OMP" << std::endl;
            boost::shared_ptr<pclomp::GeneralizedIterativeClosestPoint<PointT3, PointT3>> gicp(new pclomp::GeneralizedIterativeClosestPoint<PointT3, PointT3>());

            if(private_nh.param<bool>("enable_transformation_epsilon", true))
              gicp->setTransformationEpsilon(private_nh.param<double>("transformation_epsilon", 0.01));
            if(private_nh.param<bool>("enable_maximum_iterations", true))
              gicp->setMaximumIterations(private_nh.param<int>("maximum_iterations", 64));
            if(private_nh.param<bool>("enable_use_reciprocal_correspondences", true))
              gicp->setUseReciprocalCorrespondences(private_nh.param<bool>("use_reciprocal_correspondences", false));
            if(private_nh.param<bool>("enable_gicp_correspondence_randomness", true))
              gicp->setCorrespondenceRandomness(private_nh.param<int>("gicp_correspondence_randomness", 20));
            if(private_nh.param<bool>("enable_gicp_max_optimizer_iterations", true))
              gicp->setMaximumOptimizerIterations(private_nh.param<int>("gicp_max_optimizer_iterations", 20));

            if(private_nh.param<bool>("enable_gicp_max_correspondance_distance", false))
              gicp->setMaxCorrespondenceDistance(private_nh.param<double>("gicp_max_correspondance_distance", 0.05));
            if(private_nh.param<bool>("enable_gicp_euclidean_fitness_epsilon", false))
              gicp->setEuclideanFitnessEpsilon(private_nh.param<double>("gicp_euclidean_fitness_epsilon", 1));
            if(private_nh.param<bool>("enable_gicp_ransac_outlier_threshold", false)) 
              gicp->setRANSACOutlierRejectionThreshold(private_nh.param<double>("gicp_ransac_outlier_threshold", 1.5));

            std::cout << "max dist: " << gicp->getMaxCorrespondenceDistance() << std::endl;
            std::cout << "ransac: " << gicp->getRANSACOutlierRejectionThreshold() << std::endl;
            std::cout << "fitness: " << gicp->getEuclideanFitnessEpsilon() << std::endl;

            registration = gicp;
            registration->setTransformationEstimation(te);
            registration->setInputTarget(buildingsCloud);
            registration->setInputSource(odomCloud);
            pcl::PointCloud<PointT3>::Ptr aligned(new pcl::PointCloud<PointT3>());
            registration->align(*aligned, guess);
            std::cout << "has converged:" << registration->hasConverged() << " score: " << registration->getFitnessScore() << std::endl;
            Eigen::Matrix4f transformation = registration->getFinalTransformation();
            std::cout<< "Transformation: " << transformation << std::endl;
            prev_guess = transformation;

            //if(registration->getFitnessScore(2.0) > private_nh.param<double>("guess_thresh", 0.5)) {
            //  first_guess = true;
            //}

            // publish icp resulting transform
            aligned->header.frame_id = "map";
            transformed_pub.publish(aligned);

            Eigen::Matrix3d t_s_bs = matrix4dto3d(transformation.cast<double>());
            
            Eigen::MatrixXd information_matrix = Eigen::MatrixXd::Identity(3, 3);
            if(private_nh.param<bool>("b_use_const_inf_matrix", false)) {
              information_matrix.topLeftCorner(2, 2).array() /= private_nh.param<float>("building_edge_stddev_xy", 0.25);
              information_matrix(2, 2) /= private_nh.param<float>("building_edge_stddev_q", 1);
            } else {
              // pc XYZI needed to compute the information matrix 
              pcl::PointCloud<PointT>::Ptr btempcloud(new pcl::PointCloud<PointT>);
              pcl::copyPointCloud(*buildingsCloud,*btempcloud); // convert pcl buildings pxyz to pxyzi
              pcl::PointCloud<PointT>::Ptr otempcloud(new pcl::PointCloud<PointT>);
              pcl::copyPointCloud(*odomCloud,*otempcloud); // convert pcl odom pxyz to pxyzi
              Eigen::MatrixXd information_matrix_6 = Eigen::MatrixXd::Identity(6, 6);

              Eigen::Isometry3d t_s_bs_iso = Eigen::Isometry3d::Identity();
              t_s_bs_iso.matrix() = transformation.cast<double>();
              information_matrix_6 = inf_calclator->calc_information_matrix_buildings(btempcloud, otempcloud, t_s_bs_iso);
            
              information_matrix.block<2,2>(0,0) = information_matrix_6.block<2,2>(0,0);
              information_matrix(2,2) = information_matrix_6(5,5);
            }
            std::cout << "buildings inf: " << information_matrix << std::endl;

            Eigen::Isometry2d t_s_bs_iso = Eigen::Isometry2d::Identity();
            t_s_bs_iso.matrix() = t_s_bs;

            Eigen::Isometry2d temp = t_s_bs_iso*(keyframe->odom);
            if(fix_first_building) {
              if(private_nh.param<bool>("fix_first_building", true)) {
                std::cout << "fixed building!" << std::endl;
                auto edge = graph_slam->add_se2_edge_prior(keyframe->node, temp, information_matrix);
                graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));
              }
              fix_first_building = false;
            }
            std::cout << "prior gps: " << *keyframe->utm_coord << std::endl;
            std::cout << "prior quat: " << (*keyframe->orientation).toRotationMatrix() << std::endl;
            std::cout << "prior se2: " << (temp).matrix() << std::endl;
            //auto edge = graph_slam->add_se2_edge_prior(keyframe->node, t_s_bs_iso*(keyframe->odom), information_matrix);
            //graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));

            // t_s_bs in map frame
            //geometry_msgs::TransformStamped ts = matrix2transform(keyframe->stamp, (t_s_bs*keyframe->odom.matrix()).cast<float>(), "map", "kf_"+std::to_string(keyframe->index));
            //b_tf_broadcaster.sendTransform(ts);

            // t_s_bs in kf frame
            //geometry_msgs::TransformStamped ts4 = matrix2transform(keyframe->stamp, (t_s_bs).cast<float>(), "kf_"+std::to_string(keyframe->index), "t_s_bs"+std::to_string(keyframe->index));
            //b_tf_broadcaster.sendTransform(ts4);
            
            // add edges
            for(auto it1 = bnodes.begin(); it1 != bnodes.end(); it1++)
            {
              BuildingNode::Ptr bntemp = *it1;
              Eigen::Matrix3d t_bs_b = Eigen::Matrix3d::Identity();
              t_bs_b.block<2,1>(0, 2) = bntemp->local_origin;
              Eigen::Matrix3d temp1 = t_s_bs*keyframe->odom.matrix();
              
              Eigen::Matrix3d t_s_b = (t_bs_b.inverse())*temp1;
              Eigen::Isometry2d t_s_b_iso = Eigen::Isometry2d::Identity();
              t_s_b_iso.matrix() = t_s_b;

              // buildings tf
              //geometry_msgs::TransformStamped ts2 = matrix2transform(keyframe->stamp,  t_bs_b.cast<float>(), "map", "b_"+bntemp->building.id);
              //b_tf_broadcaster.sendTransform(ts2);

              // edge tf (from keyframe to building)
              //geometry_msgs::TransformStamped ts3 = matrix2transform(keyframe->stamp,  t_s_b.inverse().cast<float>(), "kf_"+std::to_string(keyframe->index), "tf_"+bntemp->building.id);
              //b_tf_broadcaster.sendTransform(ts3);
              
              //auto edge1 = graph_slam->add_se3_edge_prior(keyframe->node, t_s_b_iso, information_matrix);
              //graph_slam->add_robust_kernel(edge1, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));

              auto edge = graph_slam->add_se2_edge(bntemp->node, keyframe->node, t_s_b_iso, information_matrix);
              graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("map_edge_robust_kernel", "NONE"), private_nh.param<double>("map_edge_robust_kernel_size", 1.0));
            }
            keyframe->buildings_nodes = bnodes;
          } else {
            std::cout << "No buildings found!" << std::endl;
            b_updated = false;
          }
        } 
      }
    }

    // if there are new keyframes, but all of them do not have buildings (b_updated = false) 
    // then re-initialize the first guess for next keyframe that will have buildings
    if(new_kf && !b_updated) {
      std::cout << "RE-INITIATE INITIAL GUESS!!!!!!!!!!!!!" << std::endl;
      first_guess = true;
    }
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

  std::vector<Eigen::Matrix4d> loadPoses(std::string file_name) {
    std::vector<Eigen::Matrix4d> poses;
    FILE *fp = fopen(file_name.c_str(),"r");
    if (!fp)
      return poses;
    while (!feof(fp)) {
      Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
      if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                     &P(0,0), &P(0,1), &P(0,2), &P(0,3),
                     &P(1,0), &P(1,1), &P(1,2), &P(1,3),
                     &P(2,0), &P(2,1), &P(2,2), &P(2,3) )==12) {
        Eigen::Matrix3d pose_3d = Eigen::Matrix3d::Identity();
        pose_3d = matrix4dto3d(P);
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
    if(!keyframe_updated & !flush_gps_queue() & !flush_imu_queue() & !update_buildings_nodes()) {
      //std::cout << "no opt!" << std::endl;
      return;
    }

    // loop detection
    std::vector<Loop::Ptr> loops = loop_detector->detect(keyframes, new_keyframes, *graph_slam);
    for(const auto& loop : loops) {
      Eigen::Isometry2d relpose(loop->relative_pose.cast<double>());
      Eigen::MatrixXd information_matrix_6 = inf_calclator->calc_information_matrix(loop->key1->cloud, loop->key2->cloud, isometry2dto3d(relpose));
      Eigen::MatrixXd information_matrix = Eigen::MatrixXd::Identity(3, 3);
      information_matrix.block<2,2>(0,0) = information_matrix_6.block<2,2>(0,0);
      information_matrix(2,2) = information_matrix_6(5,5);

      auto edge = graph_slam->add_se2_edge(loop->key1->node, loop->key2->node, relpose, information_matrix);
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("loop_closure_edge_robust_kernel", "NONE"), private_nh.param<double>("loop_closure_edge_robust_kernel_size", 1.0));
    }

    std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
    new_keyframes.clear();

    // move the first node anchor position to the current estimate of the first node pose
    // so the first node moves freely while trying to stay around the origin
    if(anchor_node && private_nh.param<bool>("fix_first_node_adaptive", true)) {
      Eigen::Isometry2d anchor_target = static_cast<g2o::VertexSE2*>(anchor_edge->vertices()[1])->estimate().toIsometry();
      anchor_node->setEstimate(anchor_target);
    }

    // optimize the pose graph
    int num_iterations = private_nh.param<int>("g2o_solver_num_iterations", 1024);
    graph_slam->optimize(num_iterations);

    // publish tf
    const auto& keyframe = keyframes.back();
    Eigen::Isometry2d trans = keyframe->node->estimate().toIsometry() * keyframe->odom.inverse();
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
      geometry_msgs::TransformStamped ts = matrix2transform2d(keyframe->stamp, trans.matrix().cast<float>(), map_frame_id, odom_frame_id);
      odom2map_pub.publish(ts);
    }

    if(markers_pub.getNumSubscribers()) {
      auto markers = create_marker_array(ros::Time::now());
      markers_pub.publish(markers);
    }

    /*****************************************************************************/
    pcl::PointCloud<PointT3>::Ptr estimatedBuildingsCloud(new pcl::PointCloud<PointT3>);
    //pcl::PointCloud<PointT3>::Ptr buildingsCloud(new pcl::PointCloud<PointT3>);
    for(auto it3 = buildings.begin(); it3 != buildings.end(); it3++)
    {
      BuildingNode::Ptr btemp = *it3;
      Eigen::Isometry3d est = isometry2dto3d(btemp->node->estimate().toIsometry());
      pcl::PointCloud<PointT3>::Ptr temp_cloud_7(new pcl::PointCloud<PointT3>);
      pcl::transformPointCloud(*(btemp->referenceSystem), *temp_cloud_7, est.matrix());
      *estimatedBuildingsCloud += *temp_cloud_7;

      //*buildingsCloud += *btemp->building.geometry;
    }

    sensor_msgs::PointCloud2Ptr eb_cloud_msg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*estimatedBuildingsCloud, *eb_cloud_msg);
    eb_cloud_msg->header.frame_id = "map";
    eb_cloud_msg->header.stamp = keyframe->stamp;
    estimated_buildings_pub.publish(eb_cloud_msg);

    /*sensor_msgs::PointCloud2Ptr b_cloud_msg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*buildingsCloud, *b_cloud_msg);
    b_cloud_msg->header.frame_id = "map";
    b_cloud_msg->header.stamp = keyframe->stamp;
    buildings_pub.publish(b_cloud_msg);*/
    /*****************************************************************************/

    /********************************************************************************************/
    // errors computation
    // RPE always computed
    // ATE computed only if the parameter "enable_ate_calculation" is true (because gt is not always present)
    std::cout << "starting errors computation" << std::endl;
    Eigen::IOFormat lineFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");

    tf::StampedTransform transform_t;
    tf_listener.lookupTransform("base_link", "camera_gray_left", ros::Time(0), transform_t);
    Eigen::Quaterniond q;
    tf::quaternionTFToEigen(transform_t.getRotation(), q);
    Eigen::Vector3d v;
    tf::vectorTFToEigen(transform_t.getOrigin(), v);
    Eigen::Matrix4d base_camera = Eigen::Matrix4d::Identity();
    base_camera.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    //base_camera.block<3, 1>(0, 3) = v;

    // RPE start
    Eigen::Matrix4d prev_pose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d prev_gt = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Matrix4d> rpe_poses;
    std::vector<Eigen::Matrix4d> rpe_gt;
    std::ofstream myfile6;
    myfile6.open ("rpe.txt");
    bool first_enter = true;
    for(int i = 0; i < gt.size(); i++) {
      int pos = getIndexPosition(i);
      if(pos >= 0) {
        Eigen::Matrix4d pose = isometry2dto3d(keyframes[pos]->node->estimate().toIsometry()).matrix();
        if(!first_enter) {
          rpe_poses.push_back(computeRelativeTrans(prev_pose, ((base_camera.inverse())*pose)));
          rpe_gt.push_back(computeRelativeTrans(prev_gt, state6to3(gt[i])));
        } 
        first_enter = false;
        prev_pose = (base_camera.inverse())*pose;
        prev_gt = state6to3(gt[i]);
      }
    }
    
    double t_rpe_sum = 0.0;
    double r_rpe_sum = 0.0;
    for(int i = 0; i < rpe_poses.size(); i++) {
      //std::cout << "rpe gt " << i << ": " << rpe_gt[i] << std::endl;
      //std::cout << "rpe pose " << i << ": " << rpe_poses[i] << std::endl;
      Eigen::Matrix4d rpe_delta = computeRelativeTrans(rpe_gt[i], rpe_poses[i]);
      //std::cout << "delta gt - pos " << i << ": " << delta << std::endl;

      double t = sqrt((rpe_delta(0,3)*rpe_delta(0,3))+(rpe_delta(1,3)*rpe_delta(1,3)));
      double angle_temp = (((rpe_delta(0,0) + rpe_delta(1,1) + 1)-1)/2);
      if(angle_temp > 1.0)
        angle_temp = 1.0;
      if(angle_temp < -1.0)
        angle_temp = -1.0;
      double angle = std::acos(angle_temp); 
      /*std::cout << "t: " << t << std::endl;
      std::cout << "angle: " << angle << std::endl;*/   

      t_rpe_sum += t*t;
      r_rpe_sum += angle*angle;
    }
    
    if(rpe_poses.size() > 0) {
      double t_rpe = sqrt(t_rpe_sum/(rpe_poses.size()));
      //std::cout << "r rpe sum: " << r_rpe_sum << std::endl;
      //std::cout << "t rpe sum: " << t_rpe_sum << std::endl;
      //std::cout << "size: " << rpe_poses.size() << std::endl;
      double r_rpe = sqrt(r_rpe_sum/(rpe_poses.size()));
      myfile6 << t_rpe << " " << r_rpe << std::endl;
      std::cout << "t_rpe: " << t_rpe << std::endl;
      std::cout << "r_rpe: " << r_rpe << std::endl;
    }
    myfile6.close();
    
    // RPE end

    // ATE start
    if(private_nh.param<bool>("enable_ate_calculation", true)) {
      int index = -1, k = 0;
      while(k < keyframes.size()) {
        index = getIndexPosition(k);
        if(index >=0)
          break;
        k++;
      }
      if(index == -1)
        index = 0;

      Eigen::Matrix4d estimate = isometry2dto3d(keyframes[index]->node->estimate().toIsometry()).matrix();
      //std::cout << "estimate: " << estimate.matrix() << std::endl;
      //std::cout << "base camera: " << base_camera.matrix() << std::endl;
      Eigen::Matrix4d tr = estimate*base_camera;

      // delta transforms gt to keyframes, used for visualization
      Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
      delta = tr * (gt[index].inverse());
      //std::cout << "delta: " << delta << std::endl;
      
      // delta2 transforms keyframes to gt, used for error calculation
      Eigen::Matrix4d delta_2 = Eigen::Matrix4d::Identity();
      delta_2 = gt[index] * (tr.inverse());
      //std::cout << "delta_2: " << delta_2 << std::endl;
      
      std::ofstream myfile5;
      myfile5.open("align.txt");
      myfile5 << delta.format(lineFmt) << std::endl;
      myfile5 << delta_2.format(lineFmt) << std::endl;
      myfile5.close();
      
      // publish gt markers
      visualization_msgs::Marker gt_traj_marker;
      gt_traj_marker.header.frame_id = "map";
      gt_traj_marker.header.stamp = ros::Time::now();
      gt_traj_marker.ns = "gt";
      gt_traj_marker.id = 0;
      gt_traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;
      gt_traj_marker.action = visualization_msgs::Marker::ADD;

      gt_traj_marker.pose.orientation.w = 1.0;
      gt_traj_marker.scale.x = gt_traj_marker.scale.y = gt_traj_marker.scale.z = 0.7;

      for(int i = 0; i < gt.size(); i++) {
        geometry_msgs::Point p;

        p.x = (delta * gt[i])(0,3);
        p.y = (delta * gt[i])(1,3);
        p.z = 0;
        gt_traj_marker.points.push_back(p);
        std_msgs::ColorRGBA c;
        c.r = 1.0;
        c.g = 1.0;
        c.b = 1.0;
        c.a = 1.0;
        gt_traj_marker.colors.push_back(c);
      }

      gt_markers_pub.publish(gt_traj_marker);
      std::ofstream myfile;
      myfile.open ("poses.txt");
      std::ofstream myfile2;
      myfile2.open ("gt.txt");
      std::ofstream myfile3;
      myfile3.open ("dist.txt");
      std::ofstream myfile4;
      myfile4.open ("ate.txt");
      double sum = 0.0;
      double sum_sq = 0.0;
      int j = 0;

      for(int i = 0; i < gt.size(); i++) {
        int pos = getIndexPosition(i);
        if(pos >= 0) {
          j++;
          Eigen::Matrix4d pose = isometry2dto3d(keyframes[pos]->node->estimate().toIsometry()).matrix();
          Eigen::Matrix4d pose_trans = delta_2*pose;
          Eigen::Matrix4d pose_error = (state6to3(gt[i]).inverse())*pose_trans;

          double dx = pose_error(0, 3);
          double dy = pose_error(1, 3);
          double dist = sqrt(dx*dx+dy*dy);
          sum += dist;
          sum_sq += (dist*dist);
          //std::cout << dist << " " << dx << " " << dy << " " << dz << std::endl;
          myfile << pose.format(lineFmt) << std::endl;
          /*std::cout << "pose: " << pose << std::endl;
          Eigen::Vector3d ea = (pose.block<3, 3>(0, 0)).eulerAngles(2, 1, 0); 
          std::cout << "to ypr angles: " << ea << std::endl;
          std::cout << "yawss: " << ea[0] << std::endl;
          Eigen::Vector3d ea2 = (pose.block<3, 3>(0, 0)).eulerAngles(0, 1, 2); 
          std::cout << "to Euler angles: " << ea2 << std::endl;
          Eigen::AngleAxisd newAngleAxis(pose.block<3, 3>(0, 0));
          std::cout << "angle: " << newAngleAxis.angle() << "\naxis: " << newAngleAxis.axis() << std::endl;*/
          myfile2 << state6to3(gt[i]).format(lineFmt) << std::endl;
          myfile3 << i << " " << dist << std::endl;
        }  
      }

      // ate
      double t_mean = sum/(j);
      double t_variance = sqrt((sum_sq/(j))-(t_mean*t_mean));

      std::cout << "ate t_mean: " << t_mean << std::endl;
      std::cout << "ate t_variance: " << t_variance << std::endl;

      myfile4 << t_mean << " " << t_variance << std::endl;

      myfile.close();
      myfile2.close();
      myfile3.close();
      myfile4.close();
    }
    std::cout << "finished error computation" << std::endl;
    //ATE end
    /****************************************************************************************/
  }

  Eigen::Matrix4d computeRelativeTrans(Eigen::Matrix4d pose1, Eigen::Matrix4d pose2) {
    Eigen::Matrix4d delta = (pose1.inverse())*pose2;
    return delta;
  }

  // get the position into the keyframes array of keyframe with index "index"
  int getIndexPosition(int index) {
    for(int i = 0; i < keyframes.size(); i++) {
      if(keyframes[i]->index==index) {
        return i;
      }
    }
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
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.7;

    visualization_msgs::Marker& imu_marker = markers.markers[1];
    imu_marker.header = traj_marker.header;
    imu_marker.ns = "imu";
    imu_marker.id = 1;
    imu_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    imu_marker.pose.orientation.w = 1.0;
    imu_marker.scale.x = imu_marker.scale.y = imu_marker.scale.z = 0.75;

    traj_marker.points.resize(keyframes.size() + buildings.size());
    traj_marker.colors.resize(keyframes.size() + buildings.size());
   
    for(int i = 0; i < keyframes.size(); i++) {
      Eigen::Vector2d pos = keyframes[i]->node->estimate().translation();
      traj_marker.points[i].x = pos.x();
      traj_marker.points[i].y = pos.y();
      traj_marker.points[i].z = 0;

      double p = static_cast<double>(i) / keyframes.size();
      traj_marker.colors[i].r = 1.0 - p;
      traj_marker.colors[i].g = p;
      traj_marker.colors[i].b = 0.0;
      traj_marker.colors[i].a = 1.0;

      if(keyframes[i]->acceleration) {
        Eigen::Vector2d pos = keyframes[i]->node->estimate().translation();
        geometry_msgs::Point point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = 0;

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
        
        Eigen::Vector2d pos = buildings[i]->node->estimate().translation();
        traj_marker.points[i+keyframes.size()].x = pos.x();
        traj_marker.points[i+keyframes.size()].y = pos.y();
        traj_marker.points[i+keyframes.size()].z = 0;

        traj_marker.colors[i+keyframes.size()].r = 255.0/255.0;
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
    edge_marker.scale.x = 0.09;

    edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
    edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

    auto edge_itr = graph_slam->graph->edges().begin();
    for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
      g2o::HyperGraph::Edge* edge = *edge_itr;
      g2o::EdgeSE2* edge_se3 = dynamic_cast<g2o::EdgeSE2*>(edge);
      if(edge_se3) {
        g2o::VertexSE2* v1 = dynamic_cast<g2o::VertexSE2*>(edge_se3->vertices()[0]);
        g2o::VertexSE2* v2 = dynamic_cast<g2o::VertexSE2*>(edge_se3->vertices()[1]);
        Eigen::Vector2d pt1 = v1->estimate().translation();
        Eigen::Vector2d pt2 = v2->estimate().translation();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = 0;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = 0;

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

      std::cout << "before edge" << std::endl;
      g2o::EdgeSE2PriorXY* edge_priori_xy = dynamic_cast<g2o::EdgeSE2PriorXY*>(edge);
      if(edge_priori_xy) {
        std::cout << "entered edge" << std::endl;
        g2o::VertexSE2* v1 = dynamic_cast<g2o::VertexSE2*>(edge_priori_xy->vertices()[0]);
        Eigen::Vector2d pt1 = v1->estimate().translation();
        Eigen::Vector2d pt2 = edge_priori_xy->measurement();

        std::cout << "gps error ao: " << edge_priori_xy->error() << std::endl;
  

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = 0;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = 0;

        edge_marker.colors[i * 2].r = 128.0/255.0;
        edge_marker.colors[i * 2].g = 128.0/255.0;
        edge_marker.colors[i * 2].b = 128.0/255.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 128.0/255.0;
        edge_marker.colors[i * 2 + 1].g = 128.0/255.0;
        edge_marker.colors[i * 2 + 1].b = 128.0/255.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }
      std::cout << "after edge" << std::endl;

      g2o::EdgeSE2Prior* edge_priori = dynamic_cast<g2o::EdgeSE2Prior*>(edge);
      if(edge_priori) {
        g2o::VertexSE2* v1 = dynamic_cast<g2o::VertexSE2*>(edge_priori->vertices()[0]);
        Eigen::Vector2d pt1 = v1->estimate().translation();
        Eigen::Isometry2d pt2 = edge_priori->measurement().toIsometry();

        std::cout << "error ao: " << edge_priori->error() << std::endl;
  

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        //edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2].z = 0;
        edge_marker.points[i * 2 + 1].x = pt2.translation().x();
        edge_marker.points[i * 2 + 1].y = pt2.translation().y();
        edge_marker.points[i * 2 + 1].z = 0;

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
      Eigen::Vector2d pos = keyframes.back()->node->estimate().translation();
      sphere_marker.pose.position.x = pos.x();
      sphere_marker.pose.position.y = pos.y();
      sphere_marker.pose.position.z = 0;
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
      zero_utm_ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % 0.0 << std::endl;
    }

    std::ofstream ofs(directory + "/special_nodes.csv");
    ofs << "anchor_node " << (anchor_node == nullptr ? -1 : anchor_node->id()) << std::endl;
    ofs << "anchor_edge " << (anchor_edge == nullptr ? -1 : anchor_edge->id()) << std::endl;
    
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
        Eigen::Vector3f zero_utm_3d = Eigen::Vector3f::Identity();
        zero_utm_3d.block<2,1>(0,0) = (*zero_utm).cast<float>();
        zero_utm_3d(2,0) = 0;
        pt.getVector3fMap() += zero_utm_3d;
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

  ros::Publisher markers_pub;

  std::string map_frame_id;
  std::string odom_frame_id;

  std::mutex trans_odom2map_mutex;
  Eigen::Matrix3f trans_odom2map;
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
  boost::optional<Eigen::Vector2d> zero_utm;
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

  g2o::VertexSE2* anchor_node;
  g2o::EdgeSE2* anchor_edge;
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
  ros::Publisher estimated_buildings_pub;
  double ground_floor_max_thresh;
  double radius_search;
  int min_neighbors_in_radius;
  int ii;
  std::vector<Eigen::Matrix4d> gt;
  ros::Publisher gt_markers_pub;
  bool first_guess;
  Eigen::Matrix4f prev_guess;
  tf::TransformBroadcaster b_tf_broadcaster;
  bool fix_first_building;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::HdlGraphSlamNodelet, nodelet::Nodelet)
