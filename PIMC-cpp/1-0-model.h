class model{
 
    public:
 
        double mass, hbar;

        //Constructor//
        model(double mass = 1., 
              double hbar = 1.){

            this->mass = mass;
            this->hbar = hbar;
   
        }

        //Kinetic Energy//
        double T(double x0, double x1){

            double v = (x1 - x0)/a;
            return mass*v*v/2;

        }

        //Potential Energy//
        virtual double V(double x0){

            return 0.;

        }

        //Action//
        double S(double x0, double x1){

            return a*(T(x0, x1) + V(x0))/hbar;

        }

        //Test//
        void test(double x0, double x1){

            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "----------------MODEL-TEST-START----------------" << std::endl; 
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "mass          = " << mass                         << std::endl;
            std::cout << "hbar          = " << hbar                         << std::endl;
            std::cout << "a             = " << a                            << std::endl;  
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "current_point = " << x0                           << std::endl;
            std::cout << "next_point    = " << x1                           << std::endl; 
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "T             = " << T(x0, x1)                    << std::endl;
            std::cout << "V             = " << V(x0)                        << std::endl;
            std::cout << "S             = " << S(x0, x1)                     << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "-----------------MODEL-TEST-END-----------------" << std::endl; 
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "                                                " << std::endl;
         
        }

};
