const int    N_nod   = 500; //N_nod is a number of nods of trajectory
const int    N_traj  = 500;
const double a       = 20./N_nod;
const double d       = 2*sqrt(a);
const double D       = 1.;
const int    n_att   = 10;
const int    meash   = 1;
const int    step    = 500;
const int    sweeps  = meash*step;
 
const int    Exp     = 4;   //Number of experiments
 
const double MASS    = 1.;
const double OMEGA   = 1.;
const double HBAR    = 1.;
  
const int    Bins    =  600;
const double X_Left  = -6.;
const double X_Right =  6.;
const double H       = (X_Right-X_Left)/Bins;
 
const double G       = 1.;
const double Q       = 1.;
