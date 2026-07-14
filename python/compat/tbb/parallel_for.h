#pragma once

namespace tbb {

template <class Range, class Function>
void parallel_for(const Range& range, const Function& function) {
    function(range);
}

}  // namespace tbb
