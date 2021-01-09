#ifndef BUILDING_HPP
#define BUILDING_HPP

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>

namespace hdl_graph_slam {
	typedef pcl::PointXYZ pt3;

	class Building {
		public: 
			std::string id;
			std::map<std::string,std::string> tags;
			pcl::PointCloud<pt3>::Ptr geometry; // already interpolated and referred to zero utm
			Building(void);
	};
}
#endif