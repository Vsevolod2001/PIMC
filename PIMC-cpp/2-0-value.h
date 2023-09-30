//template <class T>
typedef double (function_type)(double*, T);

double default_function(double* x){

    return 0.;

};

double x_squared(double* x){

    return x[0]*x[0];

};

double delta_x_squared(double* x){

    return (x[0] - x[1])*(x[0] - x[1]);

};

double p_squared(double* x){

    return (a - (x[0] - x[1])*(x[0] - x[1]))/(a*a);

};

double test_correlation(double* x){
    
    return 2*x[0]*x[1]*exp(a);

};

template<class T>
class value{

    public: 

        T*             model;        
        function_type* function;
        int            number_of_points;

   
    value(){

        T default_model = T();
        this->model            = &default_model;
        this->function         = &default_function;
        this->number_of_points = 1.;

    }

    value(T*             model, 
          function_type* function, 
          int            number_of_points = 1){

        this->model            = model;
        this->function         = function;
        this->number_of_points = number_of_points;

    }

};
