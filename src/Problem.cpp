#include <iostream>

#include "Problem.h"

namespace Optimizer {

Problem::Problem(ProblemType _peoblemTYpe) : problemType_(_peoblemTYpe) {
}

Problem::~Problem() {
    globalVertexId = 0;
}

bool Problem::addVertex(const std::shared_ptr<Vertex> & _vertex) {
    if (vertices_.find(_vertex->id()) != vertices_.end()) {
        return false;
    } else {
        vertices_.emplace(_vertex->id(), _vertex);
    }

    if (problemType_ == ProblemType::SLAM_PRBLEM) {
        if (isPoseVertex(_vertex)) {
            resizePoseHessian(_vertex);
        }
    }

    return true;
}

bool Problem::removeVertex(const std::shared_ptr<Vertex> & _vertex) {
    if (vertices_.find(_vertex->id()) == vertices_.end()) {
        return false;
    }

    auto removeEdges = getConnectedEdges(_vertex);
    for (auto & edge : removeEdges) {
        removeEdge(edge);
    }

    if (isPoseVertex(_vertex)) {
        indexPoseVertices_.erase(_vertex->id());
    } else {
        indexLandmarkVertices_.erase(_vertex->id());
    }

    vertices_.erase(_vertex->id());
    vertex2Edge_.erase(_vertex->id());

    return true;
}

bool Problem::addEdge(const std::shared_ptr<Edge> & _edge) {
    if (edges_.find(_edge->id()) != edges_.end()) {
        return false;
    } else {
        edges_.emplace(_edge->id(), _edge);
    }

    for (auto & vertex : _edge->getVertices()) {
        vertex2Edge_.emplace(vertex->id(), _edge);
    }

    return true;
}

bool Problem::removeEdge(const std::shared_ptr<Edge> & _edge) {
    if (edges_.find(_edge->id()) == edges_.end()) {
        return false;
    }

    edges_.erase(_edge->id());
    return true;
}

bool Problem::solve(const int _iterations) {
	if (edges_.size() == 0 || vertices_.size() == 0) {
        std::cerr << "Cannot solve problem without edges or vertices." << std::endl;
        return false;		
	}

	setOrdering();
	makeHessian();
	computeLambdaInitLM();

	bool stop = false;
	int iter = 0;
	double lastestChi = 1e20;
	while (!stop && (iter < _iterations)) {
		std::cout << "iteration: " << iter << " , chi= " << currentChi_ << " , lambda= " << currentLambda_ << std::endl;
        bool onceStepSuccess = false;
        int failedCnt = 0;
        while (!onceStepSuccess && failedCnt < 10) {
            // addLambdaToHessianLM();
            solveLinearSystem();
            // removeLambdaHessianLM();

            updateStates();
            onceStepSuccess = isGoodStepInLM();

            if (onceStepSuccess) {
                makeHessian();
                failedCnt = 0;
            } else {
                failedCnt++;
                rollBackStates();
            }
        }
        ++iter;

        if (lastestChi - currentChi_ < 1e-5) {
            std::cout << "lastestChi - currentChi_ < 1e-5" << std::endl;
            stop = true;
        }
        lastestChi = currentChi_;
	}

    return true;
}

void Problem::setOrdering() {
    orderingPoses_ = 0;
    orderingGeneric_ = 0;
    orderingLandmarks_ = 0;

    for (auto vertex : vertices_) {
        orderingGeneric_ += vertex.second->localDimension();

        if (problemType_ == ProblemType::SLAM_PRBLEM) {
            addOrderingSLAM(vertex.second);
        }
    }

    if (problemType_ == ProblemType::SLAM_PRBLEM) {
        unsigned long allPoseDimension = orderingPoses_;
        for (auto landmarkVertex : indexLandmarkVertices_) {
            landmarkVertex.second->setOrderingId(landmarkVertex.second->getOrderingId()+ allPoseDimension);
        }
    }
}

void Problem::addOrderingSLAM(std::shared_ptr<Vertex> & _vertex) {
    if (isPoseVertex(_vertex)) {
        _vertex->setOrderingId(orderingPoses_);
        indexPoseVertices_.emplace(_vertex->id(), _vertex);
        orderingPoses_ += _vertex->localDimension();
    } else if (isLandmarkVertex(_vertex)) {
        _vertex->setOrderingId(orderingLandmarks_);
        indexLandmarkVertices_.emplace(_vertex->id(), _vertex);
        orderingLandmarks_ += _vertex->localDimension();
    }
}

void Problem::makeHessian() {
    unsigned long size = orderingGeneric_;
    Eigen::MatrixXd H(Eigen::MatrixXd::Zero(size, size));
    Eigen::VectorXd b(Eigen::VectorXd::Zero(size));

    for (auto & edge : edges_) {
        edge.second->computeResiduals();
        edge.second->computeJacobians();

        auto jacobians = edge.second->jacobians();
        auto vertices = edge.second->getVertices();
        assert(jacobians.size() == vertices.size());
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            auto vi = vertices[i];
            if (vi->isFixed()) {
                continue;
            }

            auto jacobiani = jacobians[i];
            unsigned long indexi = vi->getOrderingId();
            unsigned long dimensioni = vi->localDimension();

            double drho;
            Eigen::MatrixXd robustInfo(edge.second->information().rows(), edge.second->information().cols());
            edge.second->robustInfo(drho, robustInfo);

            Eigen::MatrixXd JtW = jacobiani.transpose() * robustInfo;
            for (std::size_t j = i; j < vertices.size(); ++j) {
                auto vj = vertices[j];
                if (vj->isFixed()) {
                    continue;
                }

                auto jacobianj = jacobians[j];
                unsigned long indexj = vj->getOrderingId();
                unsigned long dimensionj = vj->localDimension();

                Eigen::MatrixXd hessian = JtW * jacobianj;

                H.block(indexi, indexj, dimensioni, dimensionj).noalias() += hessian;
                if (i != j) {
                    H.block(indexj, indexi, dimensionj, dimensioni).noalias() += hessian.transpose();
                }
            }
            b.segment(indexi, dimensioni).noalias() -= drho * jacobiani.transpose() * edge.second->information() * edge.second->residuals();
        }
    }

    hessian_ = H;
    b_ = b;

    if (hessianPrior_.rows() > 0) {
        Eigen::MatrixXd tempHessianPrior = hessianPrior_;
        Eigen::VectorXd tempbPrior = bPrior_;

        for (auto vertex : vertices_) {
            if (isPoseVertex(vertex.second) && vertex.second->isFixed()) {
                auto index = vertex.second->getOrderingId();
                auto dimension = vertex.second->localDimension();
                tempHessianPrior.block(index, 0, dimension, tempHessianPrior.cols()).setZero();
                tempHessianPrior.block(0, index, tempHessianPrior.rows(), dimension).setZero();
                tempbPrior.segment(index, dimension).setZero();
            }
        }
        hessian_.topLeftCorner(orderingPoses_, orderingPoses_) += tempHessianPrior;
        b_.head(orderingPoses_) += tempbPrior;
    }

    deltaX_ = Eigen::VectorXd::Zero(size);
}

void Problem::solveLinearSystem() {
    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        Eigen::MatrixXd hessian = hessian_;
        for (auto i = 0; i < hessian_.cols(); ++i) {
            hessian(i, i) += currentLambda_;
        }
        deltaX_ = hessian.ldlt().solve(b_);
    } else {
        // Step 1: schur marginalization --> Hpp, bpp
        int reserveSize = orderingPoses_;
        int margSize = orderingLandmarks_;
        Eigen::MatrixXd Hmm = hessian_.block(reserveSize, reserveSize, margSize, margSize);
        Eigen::MatrixXd Hpm = hessian_.block(0, reserveSize, reserveSize, margSize);
        Eigen::MatrixXd Hmp = hessian_.block(reserveSize, 0, margSize, reserveSize);
        Eigen::VectorXd bpp = b_.segment(0, reserveSize);
        Eigen::VectorXd bmm = b_.segment(reserveSize, margSize);

        Eigen::MatrixXd HmmInv(Eigen::MatrixXd::Zero(margSize, margSize));

        for (auto landmarkVertex : indexLandmarkVertices_) {
            int index = landmarkVertex.second->getOrderingId() - reserveSize;
            int size = landmarkVertex.second->localDimension();
            HmmInv.block(index, index, size, size) = Hmm.block(index, index, size, size).inverse();
        }

        Eigen::MatrixXd temp = Hpm * HmmInv;
        hessianPPschur_ = hessian_.block(0, 0 ,orderingPoses_, orderingPoses_) - temp * Hmp;
        bPPschur_ = bpp - temp * bmm;

        // Step 2: solve Hpp * deltaX_ = bpp
        Eigen::VectorXd deltaXpp(Eigen::VectorXd::Zero(reserveSize));
        for (auto i = 0; i < reserveSize; ++i) {
            hessianPPschur_(i, i) += currentLambda_;
        }
        deltaXpp = hessianPPschur_.ldlt().solve(bPPschur_);
        deltaX_.head(reserveSize) = deltaXpp;

        // Step 3: solve Hmm * deltaX_ = bmm - Hmp * deltaXpp
        Eigen::VectorXd deltaXmm(margSize);
        deltaXmm = HmmInv * (bmm - Hmp * deltaXpp);
        deltaX_.tail(margSize) = deltaXmm;
    }
}

void Problem::updateStates() {
    // update vertex
    for (auto vertex : vertices_) {
        vertex.second->backupParameters();

        auto index = vertex.second->getOrderingId();
        auto dimension = vertex.second->localDimension();
        Eigen::VectorXd delta = deltaX_.segment(index, dimension);
        vertex.second->plus(delta);
    }

    // update prior
    if (errPrior_.rows() > 0) {
        bPriorBackup_ = bPrior_;
        errPriorBackup_ = errPrior_;

        bPrior_ -= hessianPrior_ * deltaX_.head(orderingPoses_);
        errPrior_ = -JtPriorInv_ * bPrior_.head(orderingPoses_ - 15);
    }
}

void Problem::rollBackStates(){
    // update vertex
    for (auto vertex : vertices_) {
        vertex.second->rollBackParameters();
    }

    if (errPrior_.rows() > 0) {
        bPrior_ = bPriorBackup_;
        errPrior_ = errPriorBackup_;
    }
}

bool Problem::isPoseVertex(std::shared_ptr<Vertex> _vertex) const {
    auto type = _vertex->typeInfo();
    return type == std::string("VertexPose") || type == std::string("VertexSpeedBias");
}

bool Problem::isLandmarkVertex(std::shared_ptr<Vertex> _vertex) const {
    auto type = _vertex->typeInfo();
    return type == std::string("VertexPointXYZ") || type == std::string("VertexInverseDepth");
}

void Problem::resizePoseHessian(const std::shared_ptr<Vertex> & _vertex) {
    int size = hessianPrior_.rows() + _vertex->localDimension();
    hessianPrior_.conservativeResize(size, size);
    bPrior_.conservativeResize(size);

    bPrior_.tail(_vertex->localDimension()).setZero();
    hessianPrior_.rightCols(_vertex->localDimension()).setZero();
    hessianPrior_.bottomRows(_vertex->localDimension()).setZero();
}

std::vector<std::shared_ptr<Edge>> Problem::getConnectedEdges(const std::shared_ptr<Vertex> & _vertex) const {
    std::vector<std::shared_ptr<Edge>> edges;
    auto range = vertex2Edge_.equal_range(_vertex->id());
    for (auto iter = range.first; iter != range.second; ++iter) {
        if (edges_.find(iter->second->id()) == edges_.end()) {
            continue;
        }
        edges.emplace_back(iter->second);
    }
    return edges;
}

void Problem::computeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1;
    currentChi_ = 0.;

    for (auto edge : edges_) {
        currentChi_ += edge.second->robustChi2();
    }
    if (errPrior_.rows() > 0) {
        currentChi_ += errPrior_.squaredNorm();
    }
    currentChi_ *= 0.5;

    stopThresholdLM_ = 1e-10 * currentChi_;

    double maxDiagonal = 0;
    assert(hessian_.rows() == hessian_.cols() && "Hessian is not square!");
    for (unsigned long i = 0; i < hessian_.cols(); ++i) {
        maxDiagonal = std::max(fabs(hessian_(i, i)), maxDiagonal);
    }

    maxDiagonal = std::min(5e10, maxDiagonal);
    double tau = 1e-5;
    currentLambda_ = tau * maxDiagonal;
}

void Problem::addLambdaToHessianLM() {
    assert(hessian_.rows() == hessian_.cols() && "Hessian is not square!");
    for (unsigned long i = 0; i < hessian_.size(); ++i) {
        hessian_(i, i) += currentLambda_;
    }
}

void Problem::removeLambdaHessianLM() {
    for (unsigned long i = 0; i < hessian_.size(); ++i) {
        hessian_(i, i) -= currentLambda_;
    }   
}

bool Problem::isGoodStepInLM() {
    double scale = 0.;
    scale = 0.5 * deltaX_.transpose() * (currentLambda_ * deltaX_ + b_);
    scale += 1e-6;

    double tempChi = 0.;
    for (auto edge : edges_) {
		// Recompute residuals after update states.
        edge.second->computeResiduals();
		tempChi += edge.second->robustChi2();
    }
	if (errPrior_.size() > 0) {
		tempChi += errPrior_.squaredNorm();
	}
	tempChi *= 0.5;

	double rho = (currentChi_ - tempChi) / scale;
	if (rho > 0 && std::isfinite(tempChi)) {
		double alpha = 1. - pow((2 * rho - 1), 3);
		alpha = std::min(alpha, 2. / 3.);
		double scaleFactor = std::max(1. / 3., alpha);
		currentLambda_ *= scaleFactor;
		ni_ = 2.;
		currentChi_ = tempChi;
		return true;
	} else {
		currentLambda_ *= ni_;
		ni_ *= 2.;
		return false;
	}
}

}   //namespace Optimizer