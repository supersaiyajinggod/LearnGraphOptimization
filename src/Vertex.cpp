#include "Vertex.h"

namespace Optimizer {

unsigned long globalVertexId = 0;

Vertex::Vertex(int _numDimension, int _localDimention) {
    parameters_.resize(_numDimension);
    localDimension_ = _localDimention > 0 ? _localDimention : _numDimension;
    id_ = globalVertexId++; 
}

Vertex::~Vertex() {}

int Vertex::dimension() const { return parameters_.rows(); }

int Vertex::localDimension() const { return localDimension_; }

void Vertex::plus(const Eigen::VectorXd & _delta) { parameters_ += _delta; }

}   //namespace Optimizer