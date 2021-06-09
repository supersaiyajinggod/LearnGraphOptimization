#pragma once

#include <unordered_map>
#include <map>
#include <memory>

#include "Vertex.h"
#include "Edge.h"

namespace Optimizer {

typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    /** \brief Distingush between SLAM and generic problem.
     *  If the problem is SLAM, pose and landmark will be classified.
     *  \author eddy
     */
    enum class ProblemType {
        SLAM_PRBLEM,
        GENERIC_PROBLEM
    };

    Problem(ProblemType _peoblemTYpe);

    ~Problem();

    /** \brief Add vertex to the problem.
     *  \param[in] vertex The vertex.
     *  \return True: success, False: failed.
     *  \author eddy
     */
    bool addVertex(const std::shared_ptr<Vertex> & _vertex);


    /** \brief Remove vertex in the problem.
     *  \param[in] vertex The vertex.
     *  \return True: success, False: failed.
     *  \author eddy
     */
    bool removeVertex(const std::shared_ptr<Vertex> & _vertex);

    /** \brief Add edge to the problem.
     *  \param[in] edge The edge.
     *  \return True: success, False: failed.
     *  \author eddy
     */
    bool addEdge(const std::shared_ptr<Edge> & _edge);

    /** \brief Remove edge in the problem.
     *  \param[in] edge The edge.
     *  \return True: success, False: failed.
     *  \author eddy
     */
    bool removeEdge(const std::shared_ptr<Edge> & _edge);

    /** \brief Solve this problem.
     *  \param[in] iterations The max times to iterate.
     *  \author eddy
     */
    bool solve(const int _iterations = 10);

    /** \brief Extend the hessian matrix.
     *  \param[in] dimension The dimension to extend.
     *  \author eddy
     */
    bool extendHessianPriorSize(const int _dimension);

    // Marginalization
    bool marginalize(const std::shared_ptr<Vertex> & _frameVertex, const std::vector<std::shared_ptr<Vertex>> & _landmarkVertices);
    bool marginalize(const std::shared_ptr<Vertex> & _frameVertex);
    bool marginalize(const std::vector<std::shared_ptr<Vertex>> _frameVertex, const int _poseDim);

    Eigen::MatrixXd getHessianPrior(){ return hessianPrior_; }
    Eigen::VectorXd getbPrior(){ return bPrior_; }
    Eigen::VectorXd getErrPrior() { return errPrior_; }
    Eigen::MatrixXd getJtPrior() { return JtPriorInv_; }

    void setHessianPrior(const Eigen::MatrixXd & _H) { hessianPrior_ = _H; }
    void setbPrior(const Eigen::VectorXd & _b) { bPrior_ = _b; }
    void setErrPrior(const Eigen::VectorXd & _err) { errPrior_ = _err; }
    void setJtPrior(const Eigen::MatrixXd & _Jt) { JtPriorInv_ = _Jt; }

private:

    /** \brief Set ordering index of all vertices.
     *	\author eddy
     */
    void setOrdering();

    /** \brief Set ordering for new vertex in slam problem.
     *	\param[in] vertex The vertex.
     *	\author eddy
     */
    void addOrderingSLAM(std::shared_ptr<Vertex> & _vertex);

    /** \brief Construct the hessian matrix(JtJ).
     *	\author eddy
     */
    void makeHessian();

    /** \brief Solve the linear system, Hx=b.
     *	\author eddy
     */
    void solveLinearSystem();

    /** \brief Update solver states.
     *	\author eddy
     */
    void updateStates();

    /** \brief If the residuals become larger after update, roll back the former states.
     *	\author eddy
     */
    void rollBackStates();

    /** \brief Determine whether a vetex is pose vertex.
     *  \return True: pose vertex.
     *	\author eddy
     */
    bool isPoseVertex(std::shared_ptr<Vertex> _vertex) const;

    /** \brief Determine whether a vetex is landmark vertex.
     *  \return True: landmark vertex.
     *	\author eddy
     */
    bool isLandmarkVertex(std::shared_ptr<Vertex> _vertex) const;

    /** \brief Resize the pose hessian when adding pose.
     *  \param[in] vertex The adding vertex.
     *	\author eddy
     */
    void resizePoseHessian(const std::shared_ptr<Vertex> & _vertex);

    /** \brief Check the ordering.
     *  \return True: check over.
     *	\author eddy
     */
    bool checkOrdering();

    /** \brief Get all edges connect to the giving vertex.
     *  \return All edges.
     *	\author eddy
     */
    std::vector<std::shared_ptr<Edge>> getConnectedEdges(const std::shared_ptr<Vertex> & _vertex) const;

    /** \brief Compute the initial lambda in LM.
     *	\author eddy
     */
    void computeLambdaInitLM();

    /** \brief Add lambda to hessian matrix.
     *	\author eddy
     */
    void addLambdaToHessianLM();

    /** \brief Remove lambda from hessian matrix.
     *	\author eddy
     */
    void removeLambdaHessianLM();

    /** \brief Check is good in iterate.
     *  \return True: good. False: bad.
     *	\author eddy
     */
    bool isGoodStepInLM();

    double currentLambda_;                          // Parameter used in LM.
    double currentChi_;                             // Current chi.
    double stopThresholdLM_;                        // The threshold to exit the iteration.
    double ni_;                                     // Control the scaling of lambda.

    ProblemType problemType_;

    Eigen::MatrixXd hessian_;                       // The whole information matrix. JTR-1J.
    Eigen::VectorXd b_;                             // -rJt.
    Eigen::VectorXd deltaX_;                        // Small change in iteration.

    // Prior information.
    Eigen::MatrixXd hessianPrior_;
    Eigen::VectorXd bPrior_;
    Eigen::VectorXd bPriorBackup_;
    Eigen::VectorXd errPrior_;
    Eigen::VectorXd errPriorBackup_;
    Eigen::MatrixXd JtPriorInv_;

    // Marginalization.
    Eigen::MatrixXd hessianPPschur_;
    Eigen::VectorXd bPPschur_;
    Eigen::MatrixXd Hpp_;
    Eigen::VectorXd bpp_;
    Eigen::MatrixXd Hll_;
    Eigen::VectorXd bll_;

    HashVertex vertices_;                           // All vertices.
    HashEdge edges_;                                // All edges.
    HashVertexIdToEdge vertex2Edge_;

    // Ordering related.
    unsigned long orderingPoses_;
    unsigned long orderingLandmarks_;
    unsigned long orderingGeneric_;

    HashVertex indexPoseVertices_;
    HashVertex indexLandmarkVertices_;

    HashVertex verticesMarg_;                      // Vertices need to marginalization.

    bool bDebug_ = false;
    double tHessianCost_ = 0.;
    double tPCGsolveCost_ = 0.;

};

}   //namespace Optimizermap