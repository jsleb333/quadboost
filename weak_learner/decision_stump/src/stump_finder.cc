

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xbuilder.hpp"
#include "stump_finder.h"
#include "stump.h"

namespace decision_stump
{

/***************
 * Constructor
 ***************/
StumpFinder::StumpFinder(const xt::xarray<size_t> &sorted_X_idx,
                         const xt::xarray<double> &sorted_X,
                         const xt::xarray<double> &Y,
                         const xt::xarray<double> &W)
                        : _sorted_X_idx(sorted_X_idx),
                          _sorted_X(_sorted_X),
                          _zeroth_moments(W),
                          _Y(Y)
{
    this->_first_moments = this->_zeroth_moments * this->_Y;
    this->_second_moments = this->_first_moments * this->_Y;
}

// Public methods
const decision_stump::Stump &StumpFinder::find_stump()
{
    const size_t n_classes = this->_zeroth_moments.shape()[1];
    const size_t n_examples = this->_sorted_X.shape()[1];
    const size_t n_features = this->_sorted_X.shape()[1];
    const size_t n_partitions = 2;
    const size_t n_moments = 3;

    xt::xarray<double> moments = xt::zeros<double>({n_moments, n_partitions, n_features, n_classes});

    auto m = xt::index_view(this->_zeroth_moments, _sorted_X_idx(xt::all(), 0) );

    // moments(0, 1, xt::all(), xt::all()) = xt::sum(this->_zeroth_moments(_sorted_X_idx(xt::all(),0)), {0});
}


void StumpFinder::_update_moments(xt::xarray<double> &moments, xt::xarray<size_t> &row_idx)
{

}
const xt::xarray<double>& StumpFinder::_compute_risks(const xt::xarray<double>& moments) const
{

}


} // End namespace
