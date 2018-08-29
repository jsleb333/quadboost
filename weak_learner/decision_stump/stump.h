

#ifndef STUMP_H_
#define STUMP_H_

#include <vector>

namespace decision_stump
{
typedef unsigned int uint;

class Stump
{
public:
    // Constructor
    Stump(const std::vector<double>& risks,
          const std::vector<double>& moments);

    // Accessors
    const double get_risk() const;
    const std::vector<double> get_risks() const
        {return risks_;}
    const double get_stump() const
        {return stump_;}
    const uint get_stump_idx() const
        {return stump_idx_;}
    const uint get_feature() const
        {return feature_;}
    const double get_confidence_rates() const;

    // Public methods
    void compute_stump_value(std::vector<double>& sorted_X);
    void update(std::vector<double>& risks,
                std::vector<double>& moments,
                std::vector<uint>& possible_stumps,
                std::vector<uint>& stump_idx);

private:
    // Attributes
    std::vector<double> risks_;
    double stump_;
    uint stump_idx_;
    uint feature_;

    // Utilitary methods
    void _update_moments(std::vector<double>& moments, std::vector<int>& row_idx);
    const std::vector<double>& _compute_risks(const std::vector<double>& moments) const;
};

} // namespace decision_stump

#endif // STUMP_H_