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
           g, q,
           current_point, next_point;

    mass  = 2.; hbar = 1.;
    omega = 3.;
    g     = 2.; q    = 3.;

    current_point = 1.5;
    next_point    = 2.0;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "----------BASE-MODEL-CLASS-TEST-START-----------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;
 
    model my_model(mass, hbar);
    my_model.test(current_point, next_point);

    model new_model = my_model;
    new_model.test(current_point, next_point);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "-----------BASE-MODEL-CLASS-TEST-END------------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;


    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "-------OSCILLATOR-MODEL-CLASS-TEST-START--------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;

    oscillator my_oscillator(mass, hbar, omega);
    my_oscillator.test(current_point, next_point);

    oscillator new_oscillator = my_oscillator;
    new_oscillator.test(current_point, next_point);

    basic_oscillator.test(current_point, next_point);

    new_oscillator = basic_oscillator;
    new_oscillator.test(current_point, next_point);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "--------OSCILLATOR-MODEL-CLASS-TEST-END---------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "----------MORSE-MODEL-CLASS-TEST-START----------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;

    morse my_morse(mass, hbar, g, q);
    my_morse.test(current_point, next_point);

    morse new_morse = my_morse;
    my_morse.test(current_point, next_point);

    basic_morse.test(current_point, next_point);

    new_morse = basic_morse;
    new_morse.test(current_point, next_point);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "-----------MORSE-MODEL-CLASS-TEST-END-----------" << std::endl; 
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "                                                " << std::endl;

    return 0;
}
