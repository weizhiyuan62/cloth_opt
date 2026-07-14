#pragma once

// Serial compatibility subset used when oneTBB is unavailable.

namespace tbb {

template <class Value>
class combinable {
public:
    Value& local() { return value_; }

    template <class Combine>
    Value combine(const Combine&) const {
        return value_;
    }

private:
    Value value_{};
};

}  // namespace tbb
