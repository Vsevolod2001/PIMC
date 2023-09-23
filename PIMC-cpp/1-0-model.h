class model{
 
    public:
 
        double mass, hbar;
 
        //Constructor//
        model(double mass, 
              double hbar){

            this->mass = mass;
            this->hbar = hbar;
   
        }

        //Kinetic Energy//
        double T(double current_point,
                 double    next_point){

            double v = (current_point - next_point)/a;
            return mass*v*v/2;

        }

        //Potential Energy//
        virtual double V(double current_point){

            return 0.;

        }

        //Action//
        double S(double current_point,
                 double    next_point){

            return a*(T(current_point, next_point) + V(current_point))/hbar;

        }

        //Test//
        void test(double current_point,
                  double    next_point){

            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "----------------MODEL-TEST-START----------------" << std::endl; 
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "mass          = " << mass                         << std::endl;
            std::cout << "hbar          = " << hbar                         << std::endl;
            std::cout << "a             = " << a                            << std::endl;  
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "current_point = " << current_point                << std::endl;
            std::cout << "next_point    = " << next_point                   << std::endl; 
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "T             = " << T(current_point, next_point) << std::endl;
            std::cout << "V             = " << V(current_point)             << std::endl;
            std::cout << "S             = " << S(current_point, next_point) << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "-----------------MODEL-TEST-END-----------------" << std::endl; 
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "                                                " << std::endl;
         
        }

};
