

#ifndef STUMP_FINDER_H_
#define STUMP_FINDER_H_

#include "xtensor/xarray.hpp"
#include "stump.h"

namespace decision_stump
{

class StumpFinder
{
public:
    // Constructor
    StumpFinder(const xt::xarray<size_t> &sorted_X_idx,
                const xt::xarray<double> &sorted_X,
                const xt::xarray<double> &Y,
                const xt::xarray<double> &W);

    // Public methods
    const decision_stump::Stump& find_stump();

private:
    // Utilitary methods
    void _update_moments(xt::xarray<double>& moments, xt::xarray<size_t>& row_idx);

    const xt::xarray<double>& _compute_risks(const xt::xarray<double>& moments) const;

    // Members
    const xt::xarray<double>& _sorted_X;
    const xt::xarray<double>& _sorted_X_idx;
    const xt::xarray<double>& _Y;
    const xt::xarray<double>& _zeroth_moments;
    xt::xarray<double> _first_moments;
    xt::xarray<double> _second_moments;

};

} // End namespace

#endif // STUMP_FINDER_H_
