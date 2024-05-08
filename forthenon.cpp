//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include "forthenon.hpp"
#include "advection.hpp"

#include "parthenon_manager.hpp"
#include <parthenon/package.hpp>
#include "outputs/restart.hpp"
#include "outputs/restart_hdf5.hpp"
#include "amr_criteria/refinement_package.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "utils/utils.hpp"
#include "globals.hpp"


using parthenon::ParthenonManager;
using parthenon::ParthenonStatus;
using parthenon::Outputs;
using parthenon::Mesh;
using parthenon::ParameterInput;
using parthenon::SimTime;



ForthenonManager *pman;
forthOutputs *pouts;

// here we make a stringstream to mimic the format parthenon expects for
// the input parameters. FLASH would do this on its own
std::stringstream inputPars()
{
   std::vector<ParthInBlock> inputBlocks;
   inputBlocks.emplace_back( "parthenon/job");
   inputBlocks.back().addPar("problem_id", "advection");

   inputBlocks.emplace_back("parthenon/mesh");
   inputBlocks.back().addPar("refinement", "adaptive");
   inputBlocks.back().addPar("numlevel", 3);
   inputBlocks.back().addPar("nx1", 64);
   inputBlocks.back().addPar("x1min", -0.5);
   inputBlocks.back().addPar("x1max",  0.5);
   inputBlocks.back().addPar("ix1_bc",  "periodic");
   inputBlocks.back().addPar("ox1_bc",  "periodic");
   inputBlocks.back().addPar("nx2", 64);
   inputBlocks.back().addPar("x2min", -0.5);
   inputBlocks.back().addPar("x2max",  0.5);
   inputBlocks.back().addPar("ix2_bc",  "periodic");
   inputBlocks.back().addPar("ox2_bc",  "periodic");
   inputBlocks.back().addPar("nx3", 1);
   inputBlocks.back().addPar("x3min", -0.5);
   inputBlocks.back().addPar("x3max",  0.5);
   inputBlocks.back().addPar("ix3_bc",  "periodic");
   inputBlocks.back().addPar("ox3_bc",  "periodic");
   inputBlocks.back().addPar("derefine_count", 1);

   inputBlocks.emplace_back("parthenon/meshblock");
   inputBlocks.back().addPar("nx1", 16);
   inputBlocks.back().addPar("nx2", 16);
   inputBlocks.back().addPar("nx3", 1);

   inputBlocks.emplace_back("parthenon/time");
   inputBlocks.back().addPar("nlim", -1);
   inputBlocks.back().addPar("tlim", 1.0);
   inputBlocks.back().addPar("integrator", "rk2");
   inputBlocks.back().addPar("ncycle_out_mesh",-1000);

   inputBlocks.emplace_back("parthenon/Advection");
   inputBlocks.back().addPar("cfl", 0.45);
   inputBlocks.back().addPar("vx", 1.0);
   inputBlocks.back().addPar("vy", 1.0);
   inputBlocks.back().addPar("vz", 1.0);
   inputBlocks.back().addPar("profile", "hard_sphere");
   inputBlocks.back().addPar("refine_tol", 0.3);
   inputBlocks.back().addPar("derefine_tol", 0.03);
   inputBlocks.back().addPar("compute_error", false);
   inputBlocks.back().addPar("num_vars", 1);
   inputBlocks.back().addPar("vec_size", 1);
   inputBlocks.back().addPar("fill_derived", false);

   inputBlocks.emplace_back("parthenon/output");
   inputBlocks.back().addPar("file_type", "hdf5");
   inputBlocks.back().addPar("dt", 0.05);
   inputBlocks.back().addPar("id", "hdf5");
   inputBlocks.back().addPar("variables", "unk");

   std::stringstream pars;
   for (auto &block : inputBlocks)
   {
      pars << block.parthBlock.str();
   }
   return pars;
}

// we need to wrap the parthenon manager's env initializer so that we can pass input paramteres as a stream
// rather than from a file specified as a cli argument.
ParthenonStatus ForthenonManager::ParthenonInitEnv(int argc, char *argv[]) {
  namespace Globals = parthenon::Globals;
  namespace Env = parthenon::Env;
  namespace SignalHandler = parthenon::SignalHandler;
  using parthenon::ArgStatus;
  using parthenon::RestartReaderHDF5;
    if (called_init_env_) {
    PARTHENON_THROW("ParthenonInitEnv called twice!");
  }
  called_init_env_ = true;

  // initialize MPI
#ifdef MPI_PARALLEL
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI Initialization failed." << std::endl;
    return ParthenonStatus::error;
  }
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank))) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    // MPI_Finalize();
    return ParthenonStatus::error;
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_size failed." << std::endl;
    // MPI_Finalize();
    return ParthenonStatus::error;
  }
#else  // no MPI
  Globals::my_rank = 0;
  Globals::nranks = 1;
#endif // MPI_PARALLEL

  Kokkos::initialize(argc, argv);

  // pgrete: This is a hack to disable allocation tracking until the Kokkos
  // tools provide a more fine grained control out of the box.
  bool unused;
  if (Env::get<bool>("KOKKOS_TRACK_ALLOC_OFF", false, unused)) {
    Kokkos::Profiling::Experimental::set_allocate_data_callback(nullptr);
    Kokkos::Profiling::Experimental::set_deallocate_data_callback(nullptr);
  }

  // parse the input arguments
  // as far as I can tell this is used to get the input file
  // as well as to determine if we are doing a restart or not
  /* ArgStatus arg_status = arg.parse(argc, argv); */
  /* if (arg_status == ArgStatus::error) { */
  /*   return ParthenonStatus::error; */
  /* } else if (arg_status == ArgStatus::complete) { */
  /*   return ParthenonStatus::complete; */
  /* } */

  // Set up the signal handler
  SignalHandler::SignalHandlerInit();
  if (Globals::my_rank == 0 && arg.wtlim > 0) SignalHandler::SetWallTimeAlarm(arg.wtlim);

  // Populate the ParameterInput object
  /* if (arg.input_filename != nullptr) { */
  /*   pinput = std::make_unique<ParameterInput>(arg.input_filename); */
  if (arg.res_flag == 0) 
  {
     // initialize the arguments ourselves
     auto pars = inputPars();
     pinput = std::make_unique<ParameterInput>();
     pinput->LoadFromStream(pars);
  }
  else if (arg.res_flag != 0) { // this could also be handled by using our own restart flag!
    // Read input from restart file
    // in FLASH we can provide the restart file name
    /* restartReader = std::make_unique<RestartReader>(arg.restart_filename); */
    restartReader = std::make_unique<RestartReaderHDF5>(arg.restart_filename);


    // Load input stream
    pinput = std::make_unique<ParameterInput>();
    /* auto inputString = restartReader->GetAttr<std::string>("Input", "File"); */
    auto inputString = restartReader->GetInputString();
    std::istringstream is(inputString);
    pinput->LoadFromStream(is);
  }

  // Modify based on command line inputs
  // probably don't want this in FLASH?
  pinput->ModifyFromCmdline(argc, argv);
  // Set the global number of ghost zones
  Globals::nghost = pinput->GetOrAddInteger("parthenon/mesh", "nghost", 2);

  // set sparse config
  Globals::sparse_config.enabled = pinput->GetOrAddBoolean(
      "parthenon/sparse", "enable_sparse", Globals::sparse_config.enabled);
#ifndef ENABLE_SPARSE
  PARTHENON_REQUIRE_THROWS(
      !Globals::sparse_config.enabled,
      "Sparse is compile-time disabled but was requested to be enabled in input file");
#endif
  Globals::sparse_config.allocation_threshold = pinput->GetOrAddReal(
      "parthenon/sparse", "alloc_threshold", Globals::sparse_config.allocation_threshold);
  Globals::sparse_config.deallocation_threshold =
      pinput->GetOrAddReal("parthenon/sparse", "dealloc_threshold",
                           Globals::sparse_config.deallocation_threshold);
  Globals::sparse_config.deallocation_count = pinput->GetOrAddInteger(
      "parthenon/sparse", "dealloc_count", Globals::sparse_config.deallocation_count);

  // set timeout config
  Globals::receive_boundary_buffer_timeout =
      pinput->GetOrAddReal("parthenon/time", "recv_bdry_buf_timeout_sec", -1.0);

  // set boundary comms buffer switch trigger
  Globals::refinement::min_num_bufs =
      pinput->GetOrAddReal("parthenon/mesh", "refinement_in_one_min_nbufs", 64);

  return ParthenonStatus::ok;
}



// this creates a set of packages that can own their own variables. This is where we would 
// register all our unk vars, fluxes, EMFs, scratch etc...
parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin)
{
   using parthenon::Metadata;
   using parthenon::StateDescriptor;
   auto pkg = std::make_shared<StateDescriptor>("advection_package");
   Metadata m;
   std::vector<std::string> advected_labels= {"advected"};
   const int nvars_unk = 2;
   m = Metadata({Metadata::Cell, Metadata::Independent,
         Metadata::FillGhost},
         std::vector<int>({nvars_unk}), std::vector<std::string>{"phi","phi2"});
   pkg->AddField("unk", m);

   m = Metadata({Metadata::Face, Metadata::Derived, Metadata::Flux},
         std::vector<int>({nvars_unk}), std::vector<std::string>{"flux_phi","flux_phi2"});
   pkg->AddField("flux", m);

   parthenon::Packages_t packages;
   packages.Add(pkg);

   pkg->CheckRefinementBlock = CheckRefinement;
   return packages;
}


// first call that will initialize parthenon & kokkos
int initParthenon(int argc, char *argv[])
{
   pman = new ForthenonManager();
   pman->app_input->ProcessPackages = ProcessPackages;
   pman->app_input->ProblemGenerator = ProblemGenerator;
   auto manager_status = pman->ParthenonInitEnv(argc, argv);
   if (manager_status == ParthenonStatus::complete) {
      pman->ParthenonFinalize();
      return 0;
   }
   if (manager_status == ParthenonStatus::error) {
      pman->ParthenonFinalize();
      return 1;
   }
   return 2;
}

void Grid_getNumBlocks(int &nblks)
{
   auto mesh = pman->pmesh.get();
   auto blockList = mesh->block_list;
   nblks = blockList.size();

}

//initializes the actual grid
void Grid_parthInit(int nvars)
{
   using parthenon::Real;
   pman->ParthenonInitPackagesAndMesh();
   auto pinput = pman->pinput.get();
   auto pm = pman->pmesh.get();

   //init mesh vars


   // output initialization
   Real start_time = pinput->GetOrAddReal("parthenon/time", "start_time", 0.0);
   Real tstop = pinput->GetOrAddReal("parthenon/time", "tlim",
         std::numeric_limits<Real>::infinity());
   Real dt =
      pinput->GetOrAddReal("parthenon/time", "dt", std::numeric_limits<Real>::max());
   const auto ncycle = pinput->GetOrAddInteger("parthenon/time", "ncycle", 0);
   const auto nmax = pinput->GetOrAddInteger("parthenon/time", "nlim", -1);
   const auto nout = pinput->GetOrAddInteger("parthenon/time", "ncycle_out", 1);
   // disable mesh output by default
   const auto nout_mesh =
      pinput->GetOrAddInteger("parthenon/time", "ncycle_out_mesh", 0);
   /* auto tm = parthenon::SimTime(0., 1., 1000, cycle, 1, 0, 0.1); */
   auto tm = new parthenon::SimTime(start_time, tstop, nmax, ncycle, nout, nout_mesh, dt);


   pouts = new forthOutputs(pman->pmesh.get(), pman->pinput.get(), tm);
}

void finalizeParthenon()
{
   pman->ParthenonFinalize();
}

void Grid_writeData(double tm, int nstep)
{

   printf("wrote output\n");
   pouts->writeData(pman->pmesh.get(), pman->pinput.get(), tm, nstep);
}

forthOutputs::forthOutputs(Mesh *pm, ParameterInput *pin, SimTime *tm) :
   Outputs(pm, pin, tm), _tm(tm)
{
}

// we wrap the outputs class so that we can pass in the time and cycle number from our own driver
// rather than using those provided by parthenon
void forthOutputs::writeData(Mesh *pm, ParameterInput *pin, double tm, int nstep)
{
   parthenon::SignalHandler::OutputSignal signal = parthenon::SignalHandler::OutputSignal::none;


   _tm->time = tm;
   _tm->ncycle = nstep;
   MakeOutputs(pm, pin, _tm, signal);
}

std::vector<int> getBlockList()
{
   // parthenon only keeps leaf blocks in the blocklist so just
   // return an array of 0,1,2,3,....,nblocks
   const int blockCount = pman->pmesh.get()->GetNumMeshBlocksThisRank();
   std::vector<int> blockList(blockCount);
   std::iota(blockList.begin(), blockList.end(), 1);
   return blockList;
}

parthenon::MeshBlock *getMeshBlockPointer(const int id)
{
   return pman->pmesh.get()->block_list[id].get();
}


Kokkos::View<double****> getBlockView(const int blockID, GridDataStruct dataStruct)
{
   auto *mb = getMeshBlockPointer(blockID);
   auto &data = mb->meshblock_data.Get();
   // these methods get the Kokkos::View underlying the data in a shape that is familiar to FLASH
   // the pointer can ge extracted for example with the `.data()` method
   // passing to fortran along with the extens you can use c_f_pointer() to shape the pointer into a fortran array like thing
   // If the data exits on device we would make a host mirror of the view, copy it and pass to FLASH the host mirror's pointer
   // Parthenon views are *always* LayoutRight and the views returned here are all (var,k,j,i) which would translate to a fortran array(layoutleft) (i,j,k,var)
   if (dataStruct == CENTER)
   {
      auto solnView = data->GetVarPtr("unk")->data;
      return Kokkos::subview(solnView, 0,0,0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
   }else if (dataStruct == FLUX_X)
   {
      auto flux = data->GetVarPtr("flux")->data;
      return Kokkos::subview(flux, 0,0,0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
   }else if (dataStruct == FLUX_Y)
   {
      auto flux = data->GetVarPtr("flux")->data;
      return Kokkos::subview(flux, 1,0,0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
   }else if (dataStruct == FLUX_Z)
   {
      auto flux = data->GetVarPtr("flux")->data;
      return Kokkos::subview(flux, 2,0,0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
   }
   // make the compiler happy
   auto solnView = data->GetVarPtr("unk")->data;
   return Kokkos::subview(solnView, 0,0,0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

void getBlockIndexLimits(const int blockID, IndexLimits &blkLimits, IndexLimits &blkLimitsGC)
{
   auto *mb = getMeshBlockPointer(blockID);
   auto cellbounds = mb->cellbounds;
   parthenon::IndexRange ib = cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
   parthenon::IndexRange jb = cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
   parthenon::IndexRange kb = cellbounds.GetBoundsK(parthenon::IndexDomain::interior);
   blkLimits.data = {ib.s, ib.e, jb.s, jb.e, kb.s, kb.e};

   ib = cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
   jb = cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
   kb = cellbounds.GetBoundsK(parthenon::IndexDomain::entire);
   blkLimitsGC.data = {ib.s, ib.e, jb.s, jb.e, kb.s, kb.e};
}

std::array<double,3> getBlockDeltas(const int blockID)
{
   auto *mb = getMeshBlockPointer(blockID);
   auto coords = mb->coords;
   return {coords.Dxc<1>(), coords.Dxc<2>(), coords.Dxc<3>()};
}


// these calls are not strictly required but starting the comm buffers
// at the beginning of the time step can have performance implications for the subsequent
// mesh communications
void startCommBoundBufs()
{
   parthenon::ThreadPool tp(1);
   parthenon::TaskID none(0);
   const int num_partitions = pman->pmesh->DefaultNumPartitions();
   parthenon::TaskRegion commBoundRegion(num_partitions);
   for (int i = 0; i < num_partitions; i++) {
      auto &tl = commBoundRegion[i];
      auto &mc0 = pman->pmesh->mesh_data.GetOrAdd("base", i);

      const auto any = parthenon::BoundaryType::any;

      tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mc0);
      tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);
   }
   commBoundRegion.Execute(tp);
}

// this does the refinement tagging and the mesh refinement in one call
void updateRefinement()
{
   // note that parthenon only derefines a block after it has been tagged for derefinement consecutively 
   // for <parthenon/mesh>::derefine_count times. I don't think FLASH/paramesh does this...
   if (!pman->pmesh.get()->adaptive) {return;}
   parthenon::ThreadPool tp(1);
   parthenon::TaskID none(0);
   auto blocks = pman->pmesh.get()->block_list;
   parthenon::TaskRegion refineRegion(blocks.size());
   for (int i=0; i < blocks.size(); i++)
   {
      auto &pmb = blocks[i];
      auto &tl  = refineRegion[i];
      auto &soln = pmb->meshblock_data.Get();
      auto tag_refine = tl.AddTask(none, parthenon::Refinement::Tag<MeshBlockData<parthenon::Real>>, soln.get());
   }

   parthenon::TaskListStatus status = refineRegion.Execute(tp);
   if (status != parthenon::TaskListStatus::complete)
   {
      std::stringstream msg;
      msg << "### somethign wrong in updateRefinement\n";
      PARTHENON_FAIL(msg.str().c_str());
   }
   pman->pmesh->LoadBalancingAndAdaptiveMeshRefinement(pman->pinput.get(), pman->app_input.get());
   pman->pmesh->boundary_comm_map.clear();
   const int num_partitions = pman->pmesh->DefaultNumPartitions();
   for (int i = 0; i < num_partitions; i++) {
      auto &mbase = pman->pmesh->mesh_data.GetOrAdd("base", i);
      parthenon::BuildBoundaryBuffers(mbase);
      // this should be for multi-grid only
      for (auto &[gmg_level, mdc] : pman->pmesh->gmg_mesh_data) {
         auto &mdg = mdc.GetOrAdd(gmg_level, "base", i);
         parthenon::BuildBoundaryBuffers(mdg);
         parthenon::BuildGMGBoundaryBuffers(mdg);
      }
   }
}

// parthenon's flux correction will communicate all variables tagged with the MetaData::Flux property
// see the Grid initialization routine above
void conserveFlux()
{
   if (!pman->pmesh.get()->adaptive) {return;}
   parthenon::ThreadPool tp(1);
   parthenon::TaskID none(0);

   int num_partitions = pman->pmesh->DefaultNumPartitions();
   parthenon::TaskRegion flxCor_region(num_partitions);
   for (int i=0; i<num_partitions; i++)
   {
      auto &mc0 = pman->pmesh->mesh_data.GetOrAdd("base", i);
      auto &tl = flxCor_region[i];
      auto set_flxcor =
         parthenon::AddFluxCorrectionTasks(none, tl, mc0, pman->pmesh->multilevel);
   }
   parthenon::TaskListStatus status = flxCor_region.Execute(tp);

}

void exchangeGuardCells()
{
   parthenon::ThreadPool tp(1);
   parthenon::TaskID none(0);

   int num_partitions = pman->pmesh->DefaultNumPartitions();
   parthenon::TaskRegion exchRegion(num_partitions);
   for (int i=0; i<num_partitions; i++)
   {
      auto &mc0 = pman->pmesh->mesh_data.GetOrAdd("base", i);
      auto &tl = exchRegion[i];
      auto set_flxcor =
         parthenon::AddBoundaryExchangeTasks(none, tl, mc0, pman->pmesh->multilevel);
   }
   parthenon::TaskListStatus status = exchRegion.Execute(tp);

}
