// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/graph_slam.hpp>

#include <boost/format.hpp>
#include <g2o/stuff/macros.h>
#include <g2o/core/factory.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/edge_se2_priorxy.hpp>
#include <g2o/edge_se2_priorquat.hpp>
#include <g2o/robust_kernel_io.hpp>

#include <g2o/types/slam2d/types_slam2d.h>
#include <g2o/types/slam2d_addons/types_slam2d_addons.h>

G2O_USE_OPTIMIZATION_LIBRARY(pcg)
G2O_USE_OPTIMIZATION_LIBRARY(cholmod)  // be aware of that cholmod brings GPL dependency
G2O_USE_OPTIMIZATION_LIBRARY(csparse)  // be aware of that csparse brings LGPL unless it is dynamically linked

namespace g2o {
G2O_REGISTER_TYPE(EDGE_SE2_PRIORXY, EdgeSE2PriorXY)
G2O_REGISTER_TYPE(EDGE_SE2_PRIORQUAT, EdgeSE2PriorQuat)
}  // namespace g2o

namespace hdl_graph_slam {

/**
 * @brief constructor
 */
GraphSLAM::GraphSLAM(const std::string& solver_type) {
  graph.reset(new g2o::SparseOptimizer());
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());

  /*g2o::ParameterSE3Offset* offset = new g2o::ParameterSE3Offset;
  offset->setId(0);
  graph->addParameter(offset);*/

  std::cout << "construct solver: " << solver_type << std::endl;
  g2o::OptimizationAlgorithmFactory* solver_factory = g2o::OptimizationAlgorithmFactory::instance();
  g2o::OptimizationAlgorithmProperty solver_property;
  g2o::OptimizationAlgorithm* solver = solver_factory->construct(solver_type, solver_property);
  graph->setAlgorithm(solver);

  if(!graph->solver()) {
    std::cerr << std::endl;
    std::cerr << "error : failed to allocate solver!!" << std::endl;
    solver_factory->listSolvers(std::cerr);
    std::cerr << "-------------" << std::endl;
    std::cin.ignore(1);
    return;
  }
  std::cout << "done" << std::endl;

  robust_kernel_factory = g2o::RobustKernelFactory::instance();
}

/**
 * @brief destructor
 */
GraphSLAM::~GraphSLAM() {
  graph.reset();
}

void GraphSLAM::set_solver(const std::string& solver_type) {
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());

  /*g2o::ParameterSE3Offset* offset = new g2o::ParameterSE3Offset;
  offset->setId(0);
  graph->addParameter(offset);*/

  std::cout << "construct solver: " << solver_type << std::endl;
  g2o::OptimizationAlgorithmFactory* solver_factory = g2o::OptimizationAlgorithmFactory::instance();
  g2o::OptimizationAlgorithmProperty solver_property;
  g2o::OptimizationAlgorithm* solver = solver_factory->construct(solver_type, solver_property);
  graph->setAlgorithm(solver);

  if(!graph->solver()) {
    std::cerr << std::endl;
    std::cerr << "error : failed to allocate solver!!" << std::endl;
    solver_factory->listSolvers(std::cerr);
    std::cerr << "-------------" << std::endl;
    std::cin.ignore(1);
    return;
  }
  std::cout << "done" << std::endl;
}

int GraphSLAM::num_vertices() const {
  return graph->vertices().size();
}
int GraphSLAM::num_edges() const {
  return graph->edges().size();
}

g2o::EdgeSE2PriorXY* GraphSLAM::add_se2_prior_xy_edge(g2o::VertexSE2* v_se2, const Eigen::Vector2d& xy, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE2PriorXY* edge(new g2o::EdgeSE2PriorXY());
  edge->setMeasurement(xy);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se2;
  graph->addEdge(edge);

  /*
  std::cout << "gps error: " << edge->error() << std::endl;
  std::cout << "gps inf mat: " << edge->information() << std::endl;
  std::cout << "gps meas: " << edge->measurement() << std::endl;
  
  std::cout << "gps est: " << (v_se2->estimate()).toIsometry().matrix() << std::endl;

  std::cout << "gps comp error: " << (((v_se2->estimate()).translation()) - (edge->measurement())) << std::endl;
  */
  return edge;
}

g2o::EdgeSE2PriorQuat* GraphSLAM::add_se2_prior_quat_edge(g2o::VertexSE2* v_se2, const Eigen::Rotation2D<double>& quat, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE2PriorQuat* edge(new g2o::EdgeSE2PriorQuat());
  edge->setMeasurement(quat);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v_se2;
  graph->addEdge(edge);

  /*std::cout << "quat error: " << edge->error() << std::endl;
  std::cout << "quat inf mat: " << edge->information() << std::endl;
  std::cout << "quat meas: " << edge->measurement().toRotationMatrix() << std::endl;
  
  std::cout << "quat est: " << (v_se2->estimate()).toIsometry().matrix() << std::endl;

 
  std::cout << "quat comp error: " << ((v_se2->estimate().rotation().angle())-(edge->measurement().angle())) << std::endl;
  */

  return edge;
}

g2o::EdgeSE2Prior* GraphSLAM::add_se2_edge_prior(g2o::VertexSE2* v1, const Eigen::Isometry2d& relative_pose, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE2Prior* edge(new g2o::EdgeSE2Prior());
  edge->setMeasurement(relative_pose);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v1;
  //edge->setParameterId(0, 0);
  graph->addEdge(edge);
  
  /*std::cout << "error: " << edge->error() << std::endl;
  std::cout << "inf mat: " << edge->information() << std::endl;
  std::cout << "meas: " << edge->measurement().toIsometry().matrix() << std::endl;
  
  std::cout << "est: " << (v1->estimate()).toIsometry().matrix() << std::endl;

  std::cout << "inv: " << edge->measurement().toIsometry().matrix().inverse() << std::endl;
  std::cout << "comp error: " << ((edge->measurement().toIsometry().matrix().inverse())*((v1->estimate()).toIsometry().matrix())) << std::endl;
  */
  return edge;
}

g2o::VertexSE2* GraphSLAM::add_se2_node(const Eigen::Isometry2d& pose) {
  g2o::VertexSE2* vertex(new g2o::VertexSE2());
  vertex->setId(static_cast<int>(graph->vertices().size()));
  vertex->setEstimate(pose);
  graph->addVertex(vertex);

  return vertex;
}

g2o::EdgeSE2* GraphSLAM::add_se2_edge(g2o::VertexSE2* v1, g2o::VertexSE2* v2, const Eigen::Isometry2d& relative_pose, const Eigen::MatrixXd& information_matrix) {
  g2o::EdgeSE2* edge(new g2o::EdgeSE2());
  edge->setMeasurement(relative_pose);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  graph->addEdge(edge);

  return edge;
}

 g2o::EdgeSE2PointXY* GraphSLAM::add_se2_pointxy_edge(g2o::VertexSE2* v1, g2o::VertexPointXY* v2, const Eigen::Vector2d& relative_pose, const Eigen::MatrixXd& information_matrix){
  g2o::EdgeSE2PointXY* edge(new g2o::EdgeSE2PointXY());
  edge->setMeasurement(relative_pose);
  edge->setInformation(information_matrix);
  edge->vertices()[0] = v1;
  edge->vertices()[1] = v2;
  graph->addEdge(edge);

  return edge;
 }

 g2o::VertexPointXY* GraphSLAM::add_pointxy_node(const Eigen::Vector2d& pose) {
  g2o::VertexPointXY* vertex(new g2o::VertexPointXY());
  vertex->setId(static_cast<int>(graph->vertices().size()));
  vertex->setEstimate(pose);
  graph->addVertex(vertex);

  return vertex;
 }

void GraphSLAM::add_robust_kernel(g2o::HyperGraph::Edge* edge, const std::string& kernel_type, double kernel_size) {
  if(kernel_type == "NONE") {
    return;
  }

  g2o::RobustKernel* kernel = robust_kernel_factory->construct(kernel_type);
  if(kernel == nullptr) {
    std::cerr << "warning : invalid robust kernel type: " << kernel_type << std::endl;
    return;
  }

  kernel->setDelta(kernel_size);

  g2o::OptimizableGraph::Edge* edge_ = dynamic_cast<g2o::OptimizableGraph::Edge*>(edge);
  edge_->setRobustKernel(kernel);
}

int GraphSLAM::optimize(int num_iterations) {
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());
  if(graph->edges().size() < 10) {
    return -1;
  }

  std::cout << std::endl;
  std::cout << "--- pose graph optimization ---" << std::endl;
  std::cout << "active edges: " << graph->activeEdges().size() << std::endl;
  std::cout << "active nodes: " << graph->activeVertices().size() << std::endl;
  std::cout << "nodes: " << graph->vertices().size() << "   edges: " << graph->edges().size() << std::endl;
  std::cout << "optimizing... " << std::flush;

  std::cout << "init" << std::endl;
  graph->initializeOptimization();
  graph->setVerbose(true);

  std::cout << "chi2" << std::endl;
  double chi2 = graph->chi2();

  std::cout << "optimize!!" << std::endl;
  auto t1 = ros::WallTime::now();
  int iterations = graph->optimize(num_iterations);

  auto t2 = ros::WallTime::now();
  std::cout << "done" << std::endl;
  std::cout << "iterations: " << iterations << " / " << num_iterations << std::endl;
  std::cout << "chi2: (before)" << chi2 << " -> (after)" << graph->chi2() << std::endl;
  std::cout << "time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

  return iterations;
}

void GraphSLAM::save(const std::string& filename) {
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());

  std::ofstream ofs(filename);
  graph->save(ofs);

  g2o::save_robust_kernels(filename + ".kernels", graph);
}

bool GraphSLAM::load(const std::string& filename) {
  std::cout << "loading pose graph..." << std::endl;
  g2o::SparseOptimizer* graph = dynamic_cast<g2o::SparseOptimizer*>(this->graph.get());

  std::ifstream ifs(filename);
  if(!graph->load(ifs, true)) {
    return false;
  }

  std::cout << "nodes  : " << graph->vertices().size() << std::endl;
  std::cout << "edges  : " << graph->edges().size() << std::endl;

  if(!g2o::load_robust_kernels(filename + ".kernels", graph)) {
    return false;
  }

  return true;
}

}  // namespace hdl_graph_slam
