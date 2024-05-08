#ifndef FORTHENON_HPP
#define FORTHENON_HPP

#include "parthenon_manager.hpp"
#include "outputs/outputs.hpp"
#include "utils/utils.hpp"
#include <sstream>

using parthenon::ParthenonManager;
using parthenon::ParthenonStatus;
using parthenon::Outputs;
using parthenon::Mesh;
using parthenon::ParameterInput;
using parthenon::SimTime;

enum GridDataStruct{
   CENTER,
   FLUX_X,
   FLUX_Y,
   FLUX_Z
};

struct IndexLimits
{
   std::array<int, 6> data;
   int &operator()(const int lowHi, const int axis)
   {
      return data[lowHi + 2*axis];
   }
};

// these could all be extern "C" and communicate with FLASH
int initParthenon(int argc, char *argv[]);
void finalizeParthenon();
void Grid_parthInit(int nvars);
void Grid_writeData(double tm, int nstep);
void Grid_getNumBlocks(int &nblks);
void gr_getBlkData(int &id, double *limits, double *limitsGC, double *coord, double *delta,
                   double *unk, double *F, double *G, double *H);
std::vector<int> getBlockList();
parthenon::MeshBlock *getMeshBlockPointer(const int id);
Kokkos::View<double****> getBlockView(const int blockID, GridDataStruct dataStruct=CENTER);
void getBlockIndexLimits(const int blockID, IndexLimits &blkLimits, IndexLimits &blkLimitsGC);
std::array<double,3> getBlockDeltas(const int blockID);
void startCommBoundBufs();
void exchangeGuardCells();
void updateRefinement();
void conserveFlux();

class forthOutputs: public Outputs
{
   public:
   forthOutputs(Mesh *pm, ParameterInput *pin, SimTime *tm);
   void writeData(Mesh *pm, ParameterInput *pin, double tm, int nstep);
   SimTime *_tm;
};

class ForthenonManager: public ParthenonManager
{
   public:
      ParthenonStatus ParthenonInitEnv(int argc, char *argv[]);
      bool IsRestart() { return (arg.restart_filename == nullptr ? false : true); }


   private:
      bool called_init_env_ = false;
      parthenon::ArgParse arg;

};

struct ParthInBlock
{
   ParthInBlock(std::string blockName){parthBlock << "<" << blockName << ">\n";}
   std::stringstream parthBlock;

   template<typename T>
      void addPar(std::string par, T value)
      {
         parthBlock << par << "=" << value << "\n";
      };

};

#else
#endif

