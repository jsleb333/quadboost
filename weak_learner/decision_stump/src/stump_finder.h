

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
    StumpFinder() {}

    // Public methods
    const decision_stump::Stump& find_stump(const xt::xarray<size_t>& sorted_X_idx,
                                            const xt::xarray<double>& sorted_X,
                                            const xt::xarray<double>& Y,
                                            const xt::xarray<double>& W);

private:
    // Utilitary methods
    void _update_moments(xt::xarray<double>& moments, xt::xarray<size_t>& row_idx);

    const xt::xarray<double>& _compute_risks(const xt::xarray<double>& moments) const;

};

} // End namespace

#endif // STUMP_FINDER_H_
