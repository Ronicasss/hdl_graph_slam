// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/keyframe.hpp>

#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam2d/vertex_se2.h>

namespace hdl_graph_slam {

KeyFrame::KeyFrame(const ros::Time& stamp, const Eigen::Isometry2d& odom, double accum_distance, const pcl::PointCloud<PointT>::ConstPtr& cloud, int index) : stamp(stamp), odom(odom), accum_distance(accum_distance), cloud(cloud), node(nullptr), index(index) {buildings_check = false;}

KeyFrame::KeyFrame(const std::string& directory, g2o::HyperGraph* graph) : stamp(), odom(Eigen::Isometry2d::Identity()), accum_distance(-1), cloud(nullptr), node(nullptr) {
  load(directory, graph);
  buildings_check = false;
}

KeyFrame::~KeyFrame() {buildings_check = false;}

void KeyFrame::save(const std::string& directory) {
  if(!boost::filesystem::is_directory(directory)) {
    boost::filesystem::create_directory(directory);
  }

  std::ofstream ofs(directory + "/data");
  ofs << "stamp " << stamp.sec << " " << stamp.nsec << "\n";

  ofs << "estimate\n";
  ofs << node->estimate().toIsometry().matrix() << "\n";

  ofs << "odom\n";
  ofs << odom.matrix() << "\n";

  ofs << "accum_distance " << accum_distance << "\n";

  if(utm_coord) {
    ofs << "utm_coord " << utm_coord->transpose() << "\n";
  }

  if(acceleration) {
    ofs << "acceleration " << acceleration->transpose() << "\n";
  }

  if(orientation) {
    ofs << "orientation " << orientation->toRotationMatrix() << "\n";
  }

  if(node) {
    ofs << "id " << node->id() << "\n";
  }

  pcl::io::savePCDFileBinary(directory + "/cloud.pcd", *cloud);
}

bool KeyFrame::load(const std::string& directory, g2o::HyperGraph* graph) {
  std::ifstream ifs(directory + "/data");
  if(!ifs) {
    return false;
  }

  long node_id = -1;
  boost::optional<Eigen::Isometry2d> estimate;

  while(!ifs.eof()) {
    std::string token;
    ifs >> token;

    if(token == "stamp") {
      ifs >> stamp.sec >> stamp.nsec;
    } else if(token == "estimate") {
      Eigen::Matrix3d mat;
      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          ifs >> mat(i, j);
        }
      }
      estimate = Eigen::Isometry2d::Identity();
      estimate->linear() = mat.block<2, 2>(0, 0);
      estimate->translation() = mat.block<2, 1>(0, 2);
    } else if(token == "odom") {
      Eigen::Matrix3d odom_mat = Eigen::Matrix3d::Identity();
      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          ifs >> odom_mat(i, j);
        }
      }

      odom.setIdentity();
      odom.linear() = odom_mat.block<2, 2>(0, 0);
      odom.translation() = odom_mat.block<2, 1>(0, 2);
    } else if(token == "accum_distance") {
      ifs >> accum_distance;
    } else if(token == "utm_coord") {
      Eigen::Vector2d coord;
      ifs >> coord[0] >> coord[1];
      utm_coord = coord;
    } else if(token == "acceleration") {
      Eigen::Vector2d acc;
      ifs >> acc[0] >> acc[1];
      acceleration = acc;
    } else if(token == "orientation") {
      Eigen::Matrix2d or_mat = Eigen::Matrix2d::Identity();
      for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
          ifs >> or_mat(i, j);
        }
      }
      orientation = or_mat;
    } else if(token == "id") {
      ifs >> node_id;
    }
  }

  if(node_id < 0) {
    ROS_ERROR_STREAM("invalid node id!!");
    ROS_ERROR_STREAM(directory);
    return false;
  }

  if(graph->vertices().find(node_id) == graph->vertices().end()) {
    ROS_ERROR_STREAM("vertex ID=" << node_id << " does not exist!!");
    return false;
  }

  node = dynamic_cast<g2o::VertexSE2*>(graph->vertices()[node_id]);
  if(node == nullptr) {
    ROS_ERROR_STREAM("failed to downcast!!");
    return false;
  }

  if(estimate) {
    node->setEstimate(*estimate);
  }

  pcl::PointCloud<PointT>::Ptr cloud_(new pcl::PointCloud<PointT>());
  pcl::io::loadPCDFile(directory + "/cloud.pcd", *cloud_);
  cloud = cloud_;

  return true;
}

long KeyFrame::id() const {
  return node->id();
}

Eigen::Isometry2d KeyFrame::estimate() const {
  return node->estimate().toIsometry();
}

KeyFrameSnapshot::KeyFrameSnapshot(const Eigen::Isometry2d& pose, const pcl::PointCloud<PointT>::ConstPtr& cloud) : pose(pose), cloud(cloud) {}

KeyFrameSnapshot::KeyFrameSnapshot(const KeyFrame::Ptr& key) : pose(key->node->estimate().toIsometry()), cloud(key->cloud) {}

KeyFrameSnapshot::~KeyFrameSnapshot() {}

}  // namespace hdl_graph_slam
