#include "LossFunction.h"

namespace Optimizer {

void HuberLoss::compute(double _err2, Eigen::Vector3d & _rho) const {
    const double dsqr = delta_ * delta_;
    if (_err2 <= dsqr) {    // inlier
        _rho[0] = _err2;
        _rho[1] = 1.;
        _rho[2] = 0.;
    } else {    // outlier
        const double error = sqrt(_err2);
        _rho[0] = 2 * error * delta_ - dsqr;
        _rho[1] = delta_ / error;
        _rho[2] = -0.5 * _rho[1] / error;
    }
}

void CauchyLoss::compute(double _err2, Eigen::Vector3d & _rho) const {
    const double dsqr = delta_ * delta_;
    const double dsqrReci = 1. / dsqr;
    const double aux = dsqrReci * _err2 + 1.0;
    _rho[0] = dsqr * log(aux);
    _rho[1] = 1. / aux;
    _rho[2] = -dsqrReci * std::pow(_rho[1], 2);
}

void TukeyLoss::compute(double _err2, Eigen::Vector3d & _rho) const {
    const double dsqr = delta_ * delta_;
    const double error = sqrt(_err2);
    if (error <= delta_) {
        const double aux = error / dsqr;
        _rho[0] = dsqr * (1. - std::pow((1. - aux), 3)) / 3.;
        _rho[1] = std::pow((1. - aux), 2);
        _rho[2] = -2. * (1. - aux) / dsqr;
    } else {
        _rho[0] = dsqr / 3.;
        _rho[1] = 0.;
        _rho[2] = 0.;
    }
}

}   //namespace Optimizer