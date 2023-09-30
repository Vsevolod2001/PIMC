class oscillator : public model{

    public:
 
        double omega;
 
        //Constuctor//
        oscillator(double mass  = MASS,
                   double hbar  = HBAR,
                   double omega = OMEGA) : model(mass, hbar){

            this->omega = omega;

        }

        //Potential Energy//
        double V(double x0){

            double u = omega*x0;
            return mass*u*u/2;
        }

        //Theor Psi2 //
        double theor_psi2(double x){
        
            double L = sqrt(hbar/(mass*omega));
            double z = x/L; 
            return exp(-z*z/2)/(L*sqrt(M_PI));
            
        }

};

oscillator basic_oscillator = oscillator();
