

#include "xtensor/xarray.hpp"
#include "stump_finder.h"
#include "stump.h"

namespace decision_stump
{

// Public methods
const decision_stump::Stump& StumpFinder::find_stump(const xt::xarray<size_t>& sorted_X_idx,
                                            const xt::xarray<double>& sorted_X,
                                            const xt::xarray<double>& Y,
                                            const xt::xarray<double>& W)
{

}


// Utilitary methods
void StumpFinder::_update_moments(xt::xarray<double>& moments, xt::xarray<int>& row_idx)
{

}
const xt::xarray<double>& StumpFinder::_compute_risks(const xt::xarray<double>& moments) const
{

}

};

} // End namespace