#include "Edge.h"

namespace Optimizer {

unsigned long globalEdgeId = 0;

Edge::Edge(const int _residualDimension, const int _numVertices, const std::vector<std::string> & _verticesTypes) {
    residuals_.resize(_residualDimension, 1);
    if (!verticesTypes_.empty()) {
        verticesTypes_ = _verticesTypes;
    }
    jacobians_.resize(_numVertices);
    id_ = globalEdgeId++;

    Eigen::MatrixXd information(_residualDimension, _residualDimension);
    information.setIdentity();
    information_ = information;

    lossFunction_ = nullptr;
}

Edge::~Edge() {}

double Edge::chi2() const {
    return residuals_.transpose() * information_ * residuals_;
}

double Edge::robustChi2() const {
    double chi2 = this->chi2();
    if (lossFunction_ != nullptr) {
        Eigen::Vector3d rho;
        lossFunction_->compute(chi2, rho);
        chi2 = rho[0];
    }
    return chi2;
}

void Edge::robustInfo(double & _drho, Eigen::MatrixXd & _info) const {
    if (lossFunction_ != nullptr) {
        double chi2 = this->chi2();
        Eigen::Vector3d rho;
        lossFunction_->compute(chi2, rho);
        const Eigen::VectorXd weightErr = information_ * residuals_;

        Eigen::MatrixXd robustInformation(information_.rows(), information_.cols());
        robustInformation.setIdentity();
        robustInformation *= rho[1] * information_;
        if (rho[1] + 2 * rho[2] * chi2 > 0.) {
            robustInformation += 2* rho[2] * weightErr * weightErr.transpose();
        }
        
        _info = robustInformation;
        _drho = rho[1];
    } else {
        _drho = 1.;
        _info = information_;
    }
}

bool Edge::checkValid() const {
    if (!verticesTypes_.empty()) {
        for (auto i = 0; i < verticesTypes_.size(); ++i) {
            if (verticesTypes_[i] != vertices_[i]->typeInfo()) {
                return false;
            }
        }
    }

    return true;
}

}   //namespace Optimizer