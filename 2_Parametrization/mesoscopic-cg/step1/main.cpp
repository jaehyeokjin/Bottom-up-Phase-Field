#include "FitGLE.h"

using namespace FITGLE_NS;

int main(int argc, char** argv)
{
   FitGLE* fg = new FitGLE(argc, argv);
   fg->exec();
   delete fg;
   return 0;
}
