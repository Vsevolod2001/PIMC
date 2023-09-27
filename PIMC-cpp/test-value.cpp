#include <iostream>

#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>

#include "0-0-constants.h"

#include "1-0-model.h"
#include "1-1-model-oscillator.h"
#include "1-2-model-morse.h"

#include "2-0-value.h"

int main(int argn, char** argv){

    oscillator osc = oscillator(1.,2.,3.);
    value<oscillator> val = value<oscillator>(&osc,&x_squared,1);
    double* x = new double[2];
    x[0] = 2.;
    x[1] = 3.;  
    std::cout << val.function(x) << std::endl;
    delete [] x;
}
