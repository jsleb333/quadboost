

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"
#include "stump.h"

namespace decision_stump
{

/***************
 * Constructor
 ***************/
Stump::Stump(const xt::xarray<double> &risks, const xt::xarray<double> &moments)
{
    auto risk = xt::sum(risks, {0});
    feature_ = xt::argmin(risk);
}



}
