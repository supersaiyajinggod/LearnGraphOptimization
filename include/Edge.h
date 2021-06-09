#pragma once

#include <vector>
#include <string>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Vertex.h"
#include "LossFunction.h"

namespace Optimizer {

class Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Edge() = delete;

    Edge(const int _residualDimension, const int _numVertices, const std::vector<std::string> & _verticesTypes = std::vector<std::string>());

    virtual ~Edge();

	/**	\brief Return the id of edge.
	 *	\return The id.
	 *	\author eddy
	 */
    unsigned long id() const { return id_; }

	/**	\brief Return the ordering id of edge.
	 *	\return The ordering id.
	 *	\author eddy
	 */
    int orderingId() const { return orderingId_; }

	/**	\brief Set the ordering id of edge.
	 *	\param[in] orderingId The ordering id.
	 *	\author eddy
	 */
    void setOrderingId(const int _id) { orderingId_ = _id; }

	/** \brief Add a vertex to this edge.
	 * 	\param[in] vertex The vertex wants to be added.
	 *	\author eddy
	 */
    bool addVertex(std::shared_ptr<Vertex> & _vertex) {
        vertices_.emplace_back(_vertex);
        return true;
    }

	/** \brief Add vertices to this edge.
	 * 	\param[in] vertex The vertices want to be added.
     *  \return true: success, false: failed.
	 *	\author eddy
	 */
    bool setVertices(const std::vector<std::shared_ptr<Vertex>> & _vertices) {
        vertices_ = _vertices;
        return true;
    }

	/** \brief Get a vertex.
	 * 	\param[in] i The index of vertex.
     *  \return The vertex.
	 *	\author eddy
	 */
    std::shared_ptr<Vertex> getVertex(const int _i) const { return vertices_[_i]; }

	/** \brief Get all vertices.
     *  \return All vertices.
	 *	\author eddy
	 */
    std::vector<std::shared_ptr<Vertex>> getVertices() const { return vertices_; }

	/** \brief Get the number of vertices.
     *  \return The count of vertices.
	 *	\author eddy
	 */
    int numVertices() const { return vertices_.size(); }

	/** \brief Return the type of edge.
     *  \return The type.
	 *	\author eddy
	 */  
    virtual std::string typeInfo() const = 0;

	/** \brief Compute the residuals.
	 *	\author eddy
	 */  
    virtual void computeResiduals() = 0;

	/** \brief Compute the jacobians.
	 *	\author eddy
	 */ 
    virtual void computeJacobians() = 0;

	/** \brief Calculate the square of residuals.
     *  \return The Chi2.
	 *	\author eddy
	 */  
    double chi2() const;

	/** \brief Calculate residuals processed by robust kernel.
     *  \return The rho(Chi2).
	 *	\author eddy
	 */  
    double robustChi2() const;

	/** \brief Get the residuals.
     *  \return The residuals.
	 *	\author eddy
	 */  
    Eigen::VectorXd & residuals() { return residuals_; }

	/** \brief Get the jacobians.
     *  \return The jacobians.
	 *	\author eddy
	 */  
    std::vector<Eigen::MatrixXd> & jacobians() { return jacobians_; }

	/** \brief Get the infofmation.
     *  \return The infofmation.
	 *	\author eddy
	 */  
    Eigen::MatrixXd & information() { return information_; }

	/** \brief Set the sqrtInfofmation.
     *  \param[in] information The sqrtInfofmation.
	 *	\author eddy
	 */  
    void setInformation(const Eigen::MatrixXd & _information) {
        information_ = _information;
        sqrtInformation_ = Eigen::LLT<Eigen::MatrixXd>(information_).matrixL().transpose();
    }

	/** \brief Get the sqrtInfofmation.
     *  \return The sqrtInfofmation.
	 *	\author eddy
	 */  
    Eigen::MatrixXd & sqrtInformation() { return sqrtInformation_; }

	/** \brief Set the lossfunction.
     *  \param[in] ptr The lossfunction.
	 *	\author eddy
	 */  
    void setLossFunction(std::shared_ptr<LossFunction> & _ptr) { lossFunction_ = _ptr; }

	/** \brief Get the lossfunction.
     *  \return The lossfunction.
	 *	\author eddy
	 */  
    std::shared_ptr<LossFunction> lossFunction() { return lossFunction_; }

	/** \brief Get the robust information.
     *  \param[out] drho The robust kernel.
     *  \param[out] info The robust information. 
	 *	\author eddy
	 */ 
    void robustInfo(double & _drho, Eigen::MatrixXd & _info) const;

	/** \brief Set the observaton of this edge.
     *  \param[in] observation The observation.
	 *	\author eddy
	 */ 
    void setObservation(const Eigen::VectorXd & _observation) { observation_ = _observation; }

	/** \brief Get the observation of this edge.
     *  \return The observation.
	 *	\author eddy
	 */  
    Eigen::VectorXd & observaton() { return observation_; }

	/** \brief Check all necessary components are setted in this edge.
     *  \return true: all setted, otherwise, false.
	 *	\author eddy
	 */  
    bool checkValid() const;

protected:
    unsigned long id_;                                  // Edge id.
    int orderingId_;                                    // Edge id in the problem.
    std::vector<std::string> verticesTypes_;            // Vertices type info, using in debug.
    std::vector<std::shared_ptr<Vertex>> vertices_;     // Vertices in this edge.
    Eigen::VectorXd residuals_;                         // Residuals.
    std::vector<Eigen::MatrixXd> jacobians_;            // Jacobians.
    Eigen::MatrixXd information_;                       // Information matrix.
    Eigen::MatrixXd sqrtInformation_;
    Eigen::VectorXd observation_;                       // Observation information.
    std::shared_ptr<LossFunction> lossFunction_;
};

}   //namespace Optimizer