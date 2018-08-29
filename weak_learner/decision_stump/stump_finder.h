

#ifndef STUMP_FINDER_H_
#define STUMP_FINDER_H_

#include <vector>
#include "stump.h"

namespace decision_stump
{

class StumpFinder
{
public:
    // Constructor
    StumpFinder(const std::vector<int>& sorted_X_idx,
                const std::vector<double>& sorted_X,
                const std::vector<double>& Y,
                const std::vector<double>& W);
    
    // Public methods
    const decision_stump::Stump& find_stump(std::vector<decision_stump::Stump>& stumps_queue);

private:
    // Utilitary methods
    void _update_moments(std::vector<double>& moments, std::vector<int>& row_idx);
    const std::vector<double>& _compute_risks(const std::vector<double>& moments) const;
};

} // End namespace

#endif // STUMP_FINDER_H_