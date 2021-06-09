#pragma once

#include <Eigen/Core>

namespace Optimizer {

extern unsigned long globalVertexId;


class Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Vertex() = delete;

    Vertex(int _numDimension, int _localDimention = -1);

    virtual ~Vertex();

	/**	\brief Return the dimension of variable stored.
	 *	\return The dimension.
	 *	\author eddy
	 */
    int dimension() const;

	/** \brief Return the dimension(freedom) of vertex.
	 * 	\return The dimension.
	 *	\author eddy
	 */
	int localDimension() const;

	/** \brief Return the id of vertex.
	 * 	\return The id.
	 *	\author eddy
	 */
	unsigned long id() const { return id_; }

	/** \brief Return the parameters of vertex.
	 * 	\return The parameters.
	 *	\author eddy
	 */
	Eigen::VectorXd & getParameters() { return parameters_; };

	/** \brief Set the parameters of vertex.
	 * 	\param[in]	parameters The parameters to be setted.
	 *	\author eddy
	 */
	void setParameters(const Eigen::VectorXd & _parameters) { parameters_ = _parameters; }

	/** \brief Backup the parameters of vertex.
	 *	\author eddy
	 */
	void backupParameters() { parameters_backup_ = parameters_; }

	/** \brief Roll back the parameters of vertex.
	 *	\author eddy
	 */
	void rollBackParameters() {parameters_ = parameters_backup_; }

	/** \brief Define the plus operator of vertex.
	 * 	\param[in] delta The delta changes wanted to be added on vertex.
	 *	\author eddy
	 */
	virtual void plus(const Eigen::VectorXd & _delta);

	/** \brief Return the type of vertex.
	 * 	\return The type.
	 *	\author eddy
	 */
	virtual std::string typeInfo() const = 0;

	/** \brief Return the ordering id of vertex.
	 * 	\return The ordering id.
	 *	\author eddy
	 */
	int getOrderingId() const { return orderingId_; }

	/** \brief Set the ordering id of vertex.
	 * 	\param[in] id The the ordering id of vertex.
	 *	\author eddy
	 */
	void setOrderingId(unsigned long _id) { orderingId_ = _id; }

	/** \brief Get the state that whether the vertex is fixed.
	 * 	\return The fixed state.
	 *	\author eddy
	 */
	bool isFixed() const { return fixed_; }

	/** \brief Set the fixed state of vertex.
	 * 	\param[in] finxed The fixed state.
	 *	\author eddy
	 */
	void setFixed(bool _fixed = true) { fixed_ = _fixed; }

protected:
	Eigen::VectorXd parameters_;			// Vertex variable stored actually.
	Eigen::VectorXd parameters_backup_;		// Backup for rolling back in iteration. 
	int localDimension_;					// Local dimension of parameters. 
	unsigned long id_;						// The id of vertex.

	// Id used in problem, with dimension information.
	// For instance, if orderingId_ == 6, implies corresponding to the 6th col in hessian matrix.
	unsigned long orderingId_ = 0;
	bool fixed_ = false;					// Fix the vertex or not.
};

}   //namespace Optimizer