#include <iostream>

#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>

#include "0-0-constants.h"

int main(int argn, char** argv){
  
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "--------------CONSTANTS-TEST-START--------------" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "N_nod   = "  << N_nod                             << std::endl;
    std::cout << "N_traj  = "  << N_traj                            << std::endl;
    std::cout << "a       = "  << a                                 << std::endl;
    std::cout << "d       = "  << d                                 << std::endl;
    std::cout << "D       = "  << D                                 << std::endl;
    std::cout << "n_att   = "  << n_att                             << std::endl;
    std::cout << "meash   = "  << meash                             << std::endl;
    std::cout << "step    = "  << step                              << std::endl;
    std::cout << "sweeps  = "  << sweeps                            << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Exp     = "  << Exp                               << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "MASS    = "  << MASS                              << std::endl;
    std::cout << "OMEGA   = "  << OMEGA                             << std::endl;
    std::cout << "HBAR    = "  << HBAR                              << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Bims    = "  << Bins                              << std::endl;
    std::cout << "X_Left  = "  << X_Left                            << std::endl;
    std::cout << "X_Right = "  << X_Right                           << std::endl;
    std::cout << "H       = "  << H                                 << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "G       = "  << G                                 << std::endl; 
    std::cout << "Q       = "  << Q                                 << std::endl;  
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "---------------CONSTANTS-TEST-END---------------" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;
      
    return 0;
}
