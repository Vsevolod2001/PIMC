#include <iostream>

#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>

#include "0-0-constants.h"

#include "1-0-model.h"
#include "1-1-model-oscillator.h"
#include "1-2-model-morse.h"

int main(int argn, char** argv){

    double mass, hbar,
           omega,
           g, q;

    double* x = new double[2];
    x[0] = 1.0;
    x[1] = 1.1;

    mass  = 2.; hbar = 1.;
    omega = 3.;
    g     = 2.; q    = 3.;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "----------BASE-MODEL-CLASS-TEST-START-----------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;
 
    model my_model(mass, hbar);
    my_model.test(x);

    model new_model = my_model;
    new_model.test(x);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "-----------BASE-MODEL-CLASS-TEST-END------------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;


    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "-------OSCILLATOR-MODEL-CLASS-TEST-START--------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;

    oscillator my_oscillator(mass, hbar, omega);
    my_oscillator.test(x);

    oscillator new_oscillator = my_oscillator;
    new_oscillator.test(x);

    basic_oscillator.test(x);

    new_oscillator = basic_oscillator;
    new_oscillator.test(x);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "--------OSCILLATOR-MODEL-CLASS-TEST-END---------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "----------MORSE-MODEL-CLASS-TEST-START----------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;

    morse my_morse(mass, hbar, g, q);
    my_morse.test(x);

    morse new_morse = my_morse;
    my_morse.test(x);

    basic_morse.test(x);

    new_morse = basic_morse;
    new_morse.test(x);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "-----------MORSE-MODEL-CLASS-TEST-END-----------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;

    delete [] x;

    return 0;
}
