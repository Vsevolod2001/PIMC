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
        double V(double current_point){

            double u = omega*current_point;
            return mass*u*u/2;
        }

        //Theor Psi2 //
        double theor_psi2(double x){
        
            double L = sqrt(hbar/(mass*omega));
            double z = x/L; 
            return exp(-x*x/2)/(L*sqrt(M_PI));
            
        }

};

oscillator basic_oscillator = oscillator();
