class morse : public model{

    public:
 
        double g, q;
 
        //Constuctor//
        morse(double mass = MASS,
              double hbar = HBAR,
              double g    = G,
              double q    = Q) : model(mass, hbar){

            this->g = g;
            this->q = q;

        }

        //Potential Energy//
        double V(double x0){

            double z = exp(-q*x0);
            return 0.5*g*g*((z-1)*(z-1) - 1);
        }

        //Theor Psi2 //
        double theor_psi2(double x){
        
            double L     = hbar/(sqrt(mass)*g);
            double alpha = q*L;
            double z     = x/L;
            double N     = gsl_sf_gamma(2/alpha) * log( exp(2)/alpha ) * ( 2 - alpha )/( log( exp(alpha)*(3 - 2*alpha) ) );
            double power = (-2./alpha) * exp( -alpha*z ) - (2 - alpha)*z;
            double P     = (N/L)*exp(power); 
            return P;
            
        }

};

morse basic_morse = morse(1,1,1,1);
