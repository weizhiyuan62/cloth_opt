#pragma once

// Serial compatibility subset used when oneTBB is unavailable.

namespace tbb {

template <class Value>
class blocked_range {
public:
    blocked_range(Value begin, Value end) : begin_(begin), end_(end) {}
    Value begin() const { return begin_; }
    Value end() const { return end_; }

private:
    Value begin_;
    Value end_;
};

}  // namespace tbb
