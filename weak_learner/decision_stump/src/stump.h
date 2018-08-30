

#ifndef STUMP_H_
#define STUMP_H_

#include <vector>
#include "xtensor/xarray.hpp"

namespace decision_stump
{

class Stump
{
public:
    // Constructor
    Stump(const xt::xarray<double>& risks, const xt::xarray<double>& moments);

    // Accessors
    const double get_risk() const
        {return xt::sum(risks_)(0);}
    const xt::xarray<double> get_risks() const
        {return risks_;}
    const double get_stump() const
        {return stump_;}
    const size_t get_stump_idx() const
        {return stump_idx_;}
    const size_t get_feature() const
        {return feature_;}
    const xt::xarray<double> get_confidence_rates() const;

    // Public methods
    void compute_stump_value(xt::xarray<double>& sorted_X);
    void update(xt::xarray<double>& risks,
                xt::xarray<double>& moments,
                xt::xarray<size_t>& possible_stumps,
                size_t stump_idx);

private:
    // Attributes
    xt::xarray<double> risks_;
    xt::xarray<double> moments_0_;
    xt::xarray<double> moments_1_;
    double stump_;
    size_t stump_idx_ = 0;
    size_t feature_;
};

} // namespace decision_stump

#endif // STUMP_H_
