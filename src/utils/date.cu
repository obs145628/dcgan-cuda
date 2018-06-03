#include "date.hh"
#include <chrono>

namespace date
{
    long now()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
            ).count();
    }
}
