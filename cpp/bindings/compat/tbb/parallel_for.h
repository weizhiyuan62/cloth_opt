#pragma once

// Serial compatibility subset used when oneTBB is unavailable.

namespace tbb {

template <class Range, class Function>
void parallel_for(const Range& range, const Function& function) {
    function(range);
}

}  // namespace tbb
