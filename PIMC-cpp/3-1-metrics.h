template <class T>
class metric{

    public:

        trajectory* tr;

    metric(trajectory* tr){

        this-tr = tr;

    }

    double mean_x_squared(){
       
        double result;
        
        for(int i = 0; i < (&this->tr).number_of_points; i++){

            double x = (&this->tr).x[i];
            result += x*x;

        };

        result /= (&this->tr).number_of_points; 
        
        return result;
    };

    double mean_delta_x_squared(){
       
        double result;
        
        for(int i = 0; i < (&this->tr).number_of_points; i++){

            double current_x = (&this->tr).x[    i%(&this->tr).number_of_points];
            double    next_x = (&this->tr).x[(i+1)%(&this->tr).number_of_points];
            double   delta_x = (next_x - current_x);

            result += delta_x*delta_x;

        };

        result /= (&this->tr).number_of_points; 
        
        return result;
    };

};
