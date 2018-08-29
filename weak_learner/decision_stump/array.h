

#ifndef ARRAY_H_
#define ARRAY_H_

#include <vector>

template <class T>
class Array
{
public:
    // Constructor
    Array(std::vector<size_t>& shape, T default_value = 0);
    
    // Accessors
    const std::vector<size_t> get_shape() const
        {return shape_;}
    const size_t get_dim() const
        {return dim_;}

private:
    // Attributes
    std::vector<size_t> shape_;
    size_t dim_;
    std::vector<T> array_
};



#endif // ARRAY_H_