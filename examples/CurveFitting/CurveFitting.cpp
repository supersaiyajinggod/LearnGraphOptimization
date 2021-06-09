#include <iostream>
#include <random>

#include "Problem.h"

class CurveFittingVertex : public Optimizer::Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingVertex() : Vertex(3) {}
    
    virtual std::string typeInfo() const { return "abc"; }
};

class CurveFittingEdge : public Optimizer::Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double _x, double _y) : Edge(1, 1, std::vector<std::string>{"abc"}), x_(_x), y_(_y) {}

    virtual void computeResiduals() override {
        const Eigen::Vector3d abc = vertices_[0]->getParameters();
        residuals_[0] = std::exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]) - y_;
    }

    virtual void computeJacobians() override {
        const Eigen::Vector3d abc = vertices_[0]->getParameters();
        const double y = std::exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);

        Eigen::Matrix<double, 1, 3> jacobian;
        jacobian << x_ * x_ * y, x_ * y, y;
        jacobians_[0] = jacobian;
    }

    virtual std::string typeInfo() const override {  return "CurveFittingEdge"; }

private:
    double x_, y_;
};



int main() {
    double a = 1., b =2., c = 3.;
    double sigma = 1.;
    int N = 100;

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0., sigma);

    Optimizer::Problem problem(Optimizer::Problem::ProblemType::GENERIC_PROBLEM);

    std::shared_ptr<CurveFittingVertex> vertex = std::make_shared<CurveFittingVertex>();
    vertex->setParameters(Eigen::Vector3d(0., 0., 0.));
    problem.addVertex(vertex);

    for (int i = 0; i < N; ++i) {
        double n = noise(generator);
        double x = i / 100.;
        double y = std::exp(a * x * x + b * x + c) + n;
        
        std::shared_ptr<CurveFittingEdge> edge = std::make_shared<CurveFittingEdge>(x, y);
        std::vector<std::shared_ptr<Optimizer::Vertex>> vertices;
        vertices.emplace_back(vertex);
        edge->setVertices(vertices);

        problem.addEdge(edge);
    }

    problem.solve(20);

    std::cout << "solve result: " << vertex->getParameters().transpose() << std::endl;
    std::cout << "ground truth: 1.0 2.0 3.0" << std::endl;

    return 0;
}