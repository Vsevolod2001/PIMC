template <class T>
class trajectory{

    public:
      
        T       model;
        double* x;
        int     number_of_points;
      
    trajectory(){

        T default_model = T();
        this->model     = default_model;
        
        this->x = new double[N_nod];
        for(int i = 0; i < N_nod; i++){
            
            this->x[i] = 0.;

        };

        this->number_of_points = N_nod;
    
    }

    trajectory(T   model,
               int method,
               int number_of_points){

        this->model = model;
        
        this->x = new double[number_of_points];
        
        if(method == 0){
           
            for(int i = 0; i < number_of_points; i++){
            
                this->x[i] = 0.;

            };

        }
        else{
           
            for(int i = 0; i < number_of_points; i++){
            
                this->x[i] = (2*D)*gsl_rng_uniform(random_generator) - D;
            
            };
        };

        this->number_of_points = number_of_points;
    
    }

    trajectory(T       model,
               double* x,
               int     number_of_points){
       
        this->model = model;

        this->x = new double[number_of_points];
        for(int i = 0; i < number_of_points; i++){
    
            this->x[i] = x[i];        
        };

        this->number_of_points = number_of_points;
       
    }
    
    trajectory(const trajectory &copied_tr){

        this->model = copied_tr.model;

        delete [] this->x;

        this->x = new double[copied_tr.number_of_points];
        for(int i = 0; i < copied_tr.number_of_points; i++){
    
            this->x[i] = copied_tr.x[i];        
        };

        this->number_of_points = copied_tr.number_of_points;
       
    }
    

    trajectory& operator=(trajectory other){

        std::swap(model           , other.model           );
        std::swap(x               , other.x               );
        std::swap(number_of_points, other.number_of_points);
       
        return *this;

    }

    ~trajectory(){

        delete [] x;
 
    };

    void show(){
        
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "-------------TRAJECTORY-SHOW-START--------------" << std::endl; 
        std::cout << "------------------------------------------------" << std::endl;

        std::cout << "number_of_points = " << number_of_points          << std::endl; 
        std::cout << "------------------------------------------------" << std::endl;

        for(int i = 0; i < number_of_points; i++){
           
        std::cout << "i = " << i << " --- x[" << i << "] = " << x[i]    << std::endl;
         
        };
       
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "-------------TRAJECTORY-SHOW-END----------------" << std::endl; 
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "                                                " << std::endl;
         
    };


    void markov_chain_step(){

        double dS             = 0.;
        double current_point  = 0.;
        double previous_point = 0.;
        double next_point     = 0.;
        double new_point      = 0.;

        for(int j = 0; j < N_nod; j++){

            int i = gsl_rng_uniform_int(random_generator, number_of_points);
           
            current_point  = x[i];
            next_point     = x[(i-1)%N_nod];
            previous_point = x[(i+1)%N_nod];

            for(int k = 0; k < n_att; k++){

                double u  = gsl_rng_uniform(random_generator);
                new_point = current_point + (2*d)*u - d;
                dS        = model.S(previous_point, new_point)     -
                            model.S(previous_point, current_point) + 
                            model.S(new_point     , next_point)    -
                            model.S(current_point , next_point);

                if (dS <= 0){

                    current_point = new_point;
                
                }
                else{

                    double r = gsl_rng_uniform(random_generator);
                    if (r < exp(-dS)){

                        current_point = new_point;

                    };
               
                };

                x[i] = current_point;
                            
            };

        };

    }

};


