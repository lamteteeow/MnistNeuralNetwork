#pragma once

#include "typedefs.h"
#include <cmath>

static bool fp_almost_equal(real_t a, real_t b) {
    return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
}
