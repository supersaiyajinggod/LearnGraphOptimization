#pragma once

#include <Eigen/Core>

namespace Optimizer {


/** \brief the scaling factor for a error:
 * The error is e^T Omega e
 * The output rho is
 * rho[0]: The actual scaled error value
 * rho[1]: First derivative of the scaling function
 * rho[2]: Second derivative of the scaling function
 * \author Eddy
 */
class LossFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual ~LossFunction() {}

    virtual void compute(double _err2, Eigen::Vector3d & _rho) const = 0;
};


class TrivalLoss : public LossFunction {
public:
    virtual void compute(double _err2, Eigen::Vector3d & _rho) const override {
        _rho[0] = _err2;
        _rho[1] = 1;
        _rho[2] = 0;
    }
};

class HuberLoss : public LossFunction {
public:
    explicit HuberLoss(double _delta) : delta_(_delta) {}

    virtual void compute(double _err2, Eigen::Vector3d & _rho) const override;

private:
    double delta_;
};

class CauchyLoss : public LossFunction {
public:
    explicit CauchyLoss(double _delta) : delta_(_delta) {}

    virtual void compute(double _err2, Eigen::Vector3d & _rho) const override;

private:
    double delta_;
};

class TukeyLoss : public LossFunction {
public:
    explicit TukeyLoss(double _delta) : delta_(_delta) {}

    virtual void compute(double _err2, Eigen::Vector3d & _rho) const override;

private:
    double delta_;
};

}   //namespace Optimizer