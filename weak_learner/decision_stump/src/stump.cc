

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xoperation.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xbuilder.hpp"
#include "stump.h"

namespace decision_stump
{

/***************
 * Constructor
 ***************/
Stump::Stump(const xt::xarray<double> &risks, const xt::xarray<double> &moments)
{
    auto risk = xt::sum(risks, {0});
    feature_ = xt::argmin(risk)(0);
    risks_ = xt::view(risks, xt::all(), feature_);

    moments_0_ = xt::xarray<double>(xt::view(moments, 0, xt::all(), feature_, xt::all()));
    moments_1_ = xt::xarray<double>(xt::view(moments, 1, xt::all(), feature_, xt::all()));
}

/*************
 * Accessors
 *************/
const xt::xarray<double> Stump::get_confidence_rates() const
{   
    auto safe_divisions = xt::not_equal(moments_0_, xt::zeros_like(moments_0_));
    return xt::where(safe_divisions, moments_0_ / moments_1_, moments_0_);
}

/******************
 * Public methods
 ******************/
void Stump::compute_stump_value(xt::xarray<double>& sorted_X)
{
    if (stump_idx_ != 0)
    {
        stump_ = ( sorted_X(stump_idx_, feature_) + sorted_X(stump_idx_-1, feature_) )/2;
    }
    else
    {
        stump_ = sorted_X(stump_idx_, feature_) - 1;
    }
}

void Stump::update(xt::xarray<double>& risks,
            xt::xarray<double>& moments,
            xt::xarray<size_t>& possible_stumps,
            size_t stump_idx)
{
    auto risk = xt::sum(risks, {0});
    size_t sparse_feature_idx = xt::argmin(risk)(0);
    if (risk(sparse_feature_idx) < get_risk())
    {
        feature_ = xt::nonzero(possible_stumps)[sparse_feature_idx][0];
        risks_ = xt::xarray<double>(xt::view(risks, xt::all(), sparse_feature_idx));
        moments_0_ = xt::xarray<double>(xt::view(moments, 0, xt::all(), feature_, xt::all()));
        moments_1_ = xt::xarray<double>(xt::view(moments, 1, xt::all(), feature_, xt::all()));
        stump_idx_ = stump_idx;
    }
}

}
