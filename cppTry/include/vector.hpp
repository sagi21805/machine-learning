#ifndef vector_hpp
#define vector_hpp
#include <iostream>
#include <vector>
class Vector{

    public:
        std::vector<double> vectorValues;
        Vector(double vectorValues[]);
        double dot(Vector vector1, Vector vector2);

};


#endif

