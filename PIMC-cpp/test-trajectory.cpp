#include <iostream>

#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>

#include "0-0-constants.h"

#include "1-0-model.h"
#include "1-1-model-oscillator.h"
#include "1-2-model-morse.h"

#include "2-0-value.h"

gsl_rng* random_generator; //global gsl random generator

#include "3-0-trajectory.h"


int main(int argn, char** argv){

    std::srand(std::time(nullptr));

    const gsl_rng_type* T;
    gsl_rng_env_setup();

    T                = gsl_rng_default;
    random_generator = gsl_rng_alloc(T);

    model my_model = model(1.,2.);
    double* x = new double[24];
    
    for(int i = 0; i < 25; i++){

        x[i] = -1.2 + 0.1*i;
    };

//    trajectory<oscillator> t = trajectory<oscillator>(basic_oscillator, x, 25);
//    t.preliminary_test();
//    trajectory<model> new_t = t;
//    new_t.preliminary_test();



    trajectory<oscillator> t = trajectory<oscillator>(basic_oscillator, 1, N_nod);

    for(int i = 0; i < 10; i++){

        std::cout << "---------" << i << "---------" << std::endl; 
        t.show();
        t.markov_chain_step();

    };

    std::cout << "---------" << 10 << "---------" << std::endl; 
 
    t.show();

    delete [] x;
    
    gsl_rng_free(random_generator);

    return 0;
}
