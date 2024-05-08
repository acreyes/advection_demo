#ifndef ADVECTION_H
#define ADVECTION_H

#include <parthenon/package.hpp>

using parthenon::MeshBlock;
using parthenon::ParameterInput;
using parthenon::AmrTag;
using parthenon::MeshBlockData;
using parthenon::Real;
/* void initBlocks(MeshBlock *pmb); */
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
void Driver();

#endif
