#include "advection.hpp"
#include "forthenon.hpp"


int main(int argc, char *argv[])
{
   int err = initParthenon(argc, argv);
   if (err < 2)
   {
      return err;
   }

   Grid_parthInit(1);

   Driver();

   finalizeParthenon();

   return 0;
}
