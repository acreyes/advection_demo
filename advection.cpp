#include "advection.hpp"
#include "forthenon.hpp"

#include <parthenon/package.hpp>
#include "amr_criteria/refinement_package.hpp"
#include <Kokkos_Core.hpp>

#include <cstdio>


using parthenon::MeshBlock;
using parthenon::ParameterInput;
using parthenon::MeshBlockData;
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
   using namespace parthenon;

   // this will get the MeshBlockData
   auto &data = pmb->meshblock_data.Get();
   // now we can get the Kokkos::View held by the "unk" variables
   auto solnVec = data->GetVarPtr("unk")->data;
   auto cellbounds = pmb->cellbounds;
   IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
   IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
   IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

   auto coords = pmb->coords;
   pmb->par_for(
         PARTHENON_AUTO_LABEL, 0, 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
         KOKKOS_LAMBDA(const int n, const int k, const int j, const int i)
         {
            Real x = coords.Xc<1>(i);
            Real y = coords.Xc<2>(j);
            Real r2 = x*x + y*y;
            solnVec(0,k,j,i) = 1.;
            if (r2 < 1./20.) solnVec(0,k,j,i) = 2.;
         });
}

// we can turn this into something that calls a function with a signature more familiar to FLASH if we want to use the pre-existing
// routines in the meantime
AmrTag CheckRefinement(MeshBlockData<Real> *rc) {
   using namespace parthenon;
   // refine on advected, for example.  could also be a derived quantity
   auto pmb = rc->GetBlockPointer();
   auto pkg = pmb->packages.Get("advection_package");
   std::vector<std::string> vars = {"unk"};
   // type is parthenon::VariablePack<Variable<Real>>
   auto v = rc->PackVariables(vars);

   IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
   IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
   IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

   auto coords = pmb->coords;
   const double idx = 1./coords.Dxc<1>();
   const double idy = 1./coords.Dxc<2>();
   const double idz = 1./coords.Dxc<3>();

   // in FLASH these would be runtime parameters
   const double refine_tol = 0.005;
   const double derefine_tol = 0.0002;

   parthenon::AMRBounds bnds(ib,jb,kb);
   const int ndim = 1 + (bnds.je > bnds.js) + (bnds.ke > bnds.ks);

   // in FLASH we would loop over the refine_var_n's
   const int var = 0;
   auto q = Kokkos::subview(rc->Get("unk").data, 0, 0, var, Kokkos::ALL(),
         Kokkos::ALL(), Kokkos::ALL());
   return Refinement::SecondDerivative(bnds, q, refine_tol, derefine_tol);
}

void Driver()
{
   double tmax = 1.;
   double time = 0.;
   // this is hard coded, but really it should be computed from a cfl
   double dt = 0.001;
   int count = 0;
   const int nend = 1000;

   const double vx = 1.;
   const double vy = 1.;

   using BlockData = Kokkos::View<double****>;

   Grid_writeData(time, count);
   double tout_interval = 0.01;
   double tout = 0.;
   printf("step dt time\n");
   printf("%d %e %e\n", count, dt, time);
   while (time < tmax)
   {
      startCommBoundBufs();
      exchangeGuardCells();
      auto blockList = getBlockList();
      /* for (const int blockID : blockList) */
      // these are *not* parallel loops purposely to demonstrate how this all might map to use in FLASH
      for (int blockID = 0; blockID < blockList.size(); blockID++)
      {
         auto deltas = getBlockDeltas(blockID);
         BlockData solnVec = getBlockView(blockID, CENTER);
         BlockData flux_X  = getBlockView(blockID, FLUX_X);
         BlockData flux_Y  = getBlockView(blockID, FLUX_Y);
         IndexLimits blkLimits, blkLimitsGC;
         getBlockIndexLimits(blockID, blkLimits, blkLimitsGC);
         // x-flux
         for(int k = blkLimits(0,2); k <= blkLimits(1,2); k++)
         {
            for(int j = blkLimits(0,1); j <= blkLimits(1,1); j++)
            {
               for(int i = blkLimits(0,0); i <= blkLimits(1,0)+1; i++)
               {
                  //flux @ i-1/2
                  flux_X(0,k,j,i) = vx*solnVec(0,k,j,i-1);
               }
            }
         }
         // y-flux
         for(int k = blkLimits(0,2); k <= blkLimits(1,2); k++)
         {
            for(int j = blkLimits(0,1); j <= blkLimits(1,1)+1; j++)
            {
               for(int i = blkLimits(0,0); i <= blkLimits(1,0); i++)
               {
                  //flux @ i-1/2
                  flux_Y(0,k,j,i) = vy*solnVec(0,k,j-1,i);
               }
            }
         }
      }
      conserveFlux();
      for (int blockID = 0; blockID < blockList.size(); blockID++)
      {
         auto deltas = getBlockDeltas(blockID);
         const double dtidx = dt/deltas[0];
         const double dtidy = dt/deltas[1];
         BlockData solnVec = getBlockView(blockID, CENTER);
         BlockData flux_X  = getBlockView(blockID, FLUX_X);
         BlockData flux_Y  = getBlockView(blockID, FLUX_Y);
         IndexLimits blkLimits, blkLimitsGC;
         getBlockIndexLimits(blockID, blkLimits, blkLimitsGC);
         // x-flux
         for(int k = blkLimits(0,2); k <= blkLimits(1,2); k++)
         {
            for(int j = blkLimits(0,1); j <= blkLimits(1,1); j++)
            {
               for(int i = blkLimits(0,0); i <= blkLimits(1,0); i++)
               {
                  solnVec(0,k,j,i) += -dtidx*(flux_X(0,k,j,i+1) - flux_X(0,k,j,i)) 
                                      -dtidy*(flux_Y(0,k,j+1,i) - flux_Y(0,k,j,i));
               }
            }
         }
      }


      updateRefinement();
      count++;
      time+=dt;
      tout+=dt;
      printf("%d %e %e\n", count, dt, time);
      if (count >= nend) {break;}
      if (tout >= tout_interval)
      {
         Grid_writeData(time, count);
         tout = 0.;
      }
   }
   Grid_writeData(time, count);
}
