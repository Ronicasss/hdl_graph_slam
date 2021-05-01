// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/map_cloud_generator.hpp>

#include <pcl/octree/octree_search.h>

namespace hdl_graph_slam {

MapCloudGenerator::MapCloudGenerator() {}

MapCloudGenerator::~MapCloudGenerator() {}

static Eigen::Matrix4f matrix3fto4f(Eigen::Matrix3f m3f) {
  Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
  m4f.block<2,2>(0,0) = m3f.block<2,2>(0,0);
  m4f.block<2,1>(0,3) = m3f.block<2,1>(0,2);
  return m4f;
} 

pcl::PointCloud<MapCloudGenerator::PointT>::Ptr MapCloudGenerator::generate(const std::vector<KeyFrameSnapshot::Ptr>& keyframes, double resolution) const {
  if(keyframes.empty()) {
    std::cerr << "warning: keyframes empty!!" << std::endl;
    return nullptr;
  }

  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
  cloud->reserve(keyframes.front()->cloud->size() * keyframes.size());

  for(const auto& keyframe : keyframes) {
    Eigen::Matrix4f pose = matrix3fto4f(keyframe->pose.matrix().cast<float>());
    for(const auto& src_pt : keyframe->cloud->points) {
      PointT dst_pt;
      dst_pt.getVector4fMap() = pose * src_pt.getVector4fMap();
      dst_pt.intensity = src_pt.intensity;
      cloud->push_back(dst_pt);
    }
  }

  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = false;

  if (resolution <=0.0)
    return cloud; // To get unfiltered point cloud with intensity

  pcl::octree::OctreePointCloud<PointT> octree(resolution);
  octree.setInputCloud(cloud);
  octree.addPointsFromInputCloud();

  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  octree.getOccupiedVoxelCenters(filtered->points);

  filtered->width = filtered->size();
  filtered->height = 1;
  filtered->is_dense = false;

  return filtered;
}

}  // namespace hdl_graph_slam
