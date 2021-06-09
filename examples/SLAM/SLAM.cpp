#include <iostream>
#include <random>
#include <map>

#include <Eigen/Geometry>

#include "Problem.h"

constexpr double fx = 380.;
constexpr double fy = 240.;
constexpr double cx = 320.;
constexpr double cy = 240.;

Eigen::Vector2d project(const Eigen::Vector3d & pc) {
	const double invZ = 1.0 / pc[2];
	Eigen::Vector2d uv;
	uv[0] = pc[0] * invZ * fx + cx;
	uv[1] = pc[1] * invZ * fy + cy; 
    return uv;  
}

class VertexPose : public Optimizer::Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // x y z qx qy qz qw
    VertexPose() : Vertex(7, 6) {}

    virtual void plus(const Eigen::VectorXd & _delta) override {
        Eigen::VectorXd & parameters = getParameters();
        Eigen::Quaterniond q(parameters[6], parameters[3], parameters[4], parameters[5]);
        Eigen::Quaterniond dq = deltaQ(Eigen::Vector3d(_delta[3], _delta[4], _delta[5]));
        q = dq * q;
        q.normalize();

        parameters.head<3>() += _delta.head<3>();
        parameters[3] = q.x();
        parameters[4] = q.y();
        parameters[5] = q.z();
        parameters[6] = q.w();
    }

    virtual std::string typeInfo() const override { return "VertexPose"; }

private:
    template <typename Derived>
    Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> & _omega) {
        Eigen::Quaternion<Derived::Scalar> dq;
        Eigen::Matrix<Derived::Scalar, 3, 1> halfTheta = _omega;
        halfTheta /= static_cast<typename Derived::Scalar>(2.0);
        dq.w() = static_cast<typename Derived::Scalar>(1.0);
        dq.x() = halfTheta.x();
        dq.y() = halfTheta.y();
        dq.z() = halfTheta.z();
        return dq;
    }
};

class VertexPoint : public Optimizer::Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexPoint() : Vertex(3) {}

    std::string typeInfo() const { return "VertexPoint"; }
};

class EdgeMono : public Optimizer::Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Vertex[0]: point, Vertex[1]: camera.
    EdgeMono(const Eigen::Vector2d & _uv) : Edge(2, 2, std::vector<std::string>{"VertexXYZ", "VertexPose"}) {
        measurement_ = _uv;
    }

    virtual std::string typeInfo() const override { return "EdgeMono"; }

    virtual void computeResiduals() override {
        const Eigen::Vector3d pw = vertices_[0]->getParameters();
        const Eigen::VectorXd Tcw = vertices_[1]->getParameters();
        const Eigen::Vector3d tcw(Tcw[0], Tcw[1], Tcw[2]);
        const Eigen::Quaterniond qcw(Tcw[6], Tcw[3], Tcw[4], Tcw[5]);
        const Eigen::Vector3d pc = qcw * pw + tcw;

        residuals_ = measurement_ - project(pc);
    }

    virtual void computeJacobians() override {
        const Eigen::Vector3d pw = vertices_[0]->getParameters();
        const Eigen::VectorXd Tcw = vertices_[1]->getParameters();
        const Eigen::Vector3d tcw(Tcw[0], Tcw[1], Tcw[2]);
        const Eigen::Quaterniond qcw(Tcw[6], Tcw[3], Tcw[4], Tcw[5]);
        const Eigen::Vector3d pc = qcw * pw + tcw;
        const auto R = qcw.toRotationMatrix();
        const double & x = pc[0];
        const double & y = pc[1];
        const double & z = pc[2];
        const double z_2 = z * z;

        Eigen::Matrix<double, 2, 3> jacobianPw;
        Eigen::Matrix<double, 2, 6> jacobianTcw;

        jacobianPw(0, 0) = -fx * R(0, 0) / z + fx * x * R(2, 0) / z_2;
        jacobianPw(0, 1) = -fx * R(0, 1) / z + fx * x * R(2, 1) / z_2;
        jacobianPw(0, 2) = -fx * R(0, 2) / z + fx * x * R(2, 2) / z_2;

        jacobianPw(1, 0) = -fy * R(1, 0) / z + fy * y * R(2, 0) / z_2;
        jacobianPw(1, 1) = -fy * R(1, 1) / z + fy * y * R(2, 1) / z_2;
        jacobianPw(1, 2) = -fy * R(1, 2) / z + fy * y * R(2, 2) / z_2;

        jacobianTcw(0, 0) = -1. / z * fx;
        jacobianTcw(0, 1) = 0.;
        jacobianTcw(0, 2) = x / z_2 * fx;
        jacobianTcw(0, 3) = x * y / z_2 * fx;
        jacobianTcw(0, 4) = -(1. + (x * x / z_2)) * fx;
        jacobianTcw(0, 5) = y / z * fx;

        jacobianTcw(1, 0) = 0.;
        jacobianTcw(1, 1) = -1. / z * fy;
        jacobianTcw(1, 2) = y / z_2 * fy;
        jacobianTcw(1, 3) = (1. + y * y / z_2) * fy;
        jacobianTcw(1, 4) = -x * y / z_2 * fy;
        jacobianTcw(1, 5) = -x / z * fy;

		jacobians_[0] = jacobianPw;
		jacobians_[1] = jacobianTcw;
    }

private:
    Eigen::Vector2d measurement_;
};

int main() {
    double sigma = 1;
    double Npoint = 10;
    Eigen::Vector3d t(1., 2., 3.);
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)));

    std::vector<std::tuple<Eigen::Vector3d, Eigen::Quaterniond>> poses;
    poses.emplace_back(std::forward_as_tuple(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()));
    poses.emplace_back(std::forward_as_tuple(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()));

    std::default_random_engine generator;
    std::uniform_real_distribution<> region(2, 10);
    std::normal_distribution<double> noise(0., sigma);

	// std::vector<Eigen::Vector2d> measurements0, measurements1;
	std::map<int, std::vector<Eigen::Vector2d>> measurements;
	measurements.emplace(0 , std::vector<Eigen::Vector2d>());
	measurements.emplace(1 , std::vector<Eigen::Vector2d>());
    std::vector<Eigen::Vector3d> points;

    for (int i = 0; i < Npoint; ++i) {
        Eigen::Vector3d point(region(generator), region(generator), region(generator));
		Eigen::Vector2d uv0 = project(point);
        Eigen::Vector2d uv1 = project(q * point + t);
        uv0[0] += noise(generator);
        uv0[1] += noise(generator);		
        uv1[0] += noise(generator);
        uv1[1] += noise(generator);

        points.emplace_back(point);
		measurements.at(0).emplace_back(uv0);
		measurements.at(1).emplace_back(uv1);
    }

	// SOLVE
	Optimizer::Problem problem(Optimizer::Problem::ProblemType::SLAM_PRBLEM);
	
	std::vector<std::shared_ptr<VertexPose>> vPose;
	for (auto i = 0; i < poses.size(); ++i) {
		auto [t, q] = poses[i];
		std::shared_ptr<VertexPose> vertex = std::make_shared<VertexPose>();
		Eigen::Matrix<double, 7, 1> parameter;
		parameter << t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w();
		vertex->setParameters(parameter);
		if (i == 0) {
			vertex->setFixed();
		}

		problem.addVertex(vertex);
		vPose.emplace_back(vertex);
	}

	for (auto i = 0; i < points.size(); ++i) {
		std::shared_ptr<VertexPoint> vertex = std::make_shared<VertexPoint>();
		vertex->setParameters(points[i]);
		vertex->setFixed();
		problem.addVertex(vertex);

		for (auto cam : measurements) {
			std::shared_ptr<EdgeMono> edge = std::make_shared<EdgeMono>(cam.second[i]);
			std::vector<std::shared_ptr<Optimizer::Vertex>> vertices;
			vertices.emplace_back(vertex);
			vertices.emplace_back(vPose[cam.first]);
			edge->setVertices(vertices);

			problem.addEdge(edge);
		}
	}

	problem.solve(30);
	
	std::cout << "solve result: " << vPose[1]->getParameters().transpose() << std::endl;
	std::cout << "ground truth:     " << t.transpose() << "    " << q.coeffs().transpose() << std::endl;

    return 0;
}