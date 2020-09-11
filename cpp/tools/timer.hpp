//
// Copyright OptiLab 2020. All rights reserved.
//
// Timer class.
//

#pragma once

#include <chrono>
#include <cstdint> // for uint64_t
#include <memory>  // for std::shared_ptr

#include "system/system_export_defs.hpp"

namespace tools {

class SYS_EXPORT_CLASS Timer final {
public:
    using SPtr = std::shared_ptr<Timer>;

public:
    /**
     * \brief Constructor for Timer class. If the flag "start" is true, starts the timer
     *        on construction.
     */
    explicit Timer(bool start = true);
    ~Timer() = default;

    /**
     * \brief Starts the timer.
     */
    void start();

    /**
     * \brief Stops the timer.
     */
    void stop();

    /**
     * \brief Stops the timer and resets internal counters.
     */
    void reset();

    /**
     * \brief Returns true if the timer is running.
     *        Returns false otherwise.
     */
    inline bool isRunning() const { return m_Running; }

    /**
     * \brief Returns the elapsed time (msec.) between the start of the timer and now.
     *        This value is the cumulative running time, i.e., it does not count the
     *        time when the timer wasn't running.
     */
    uint64_t getWallClockTimeMsec();

    /**
     * \brief Returns the wallclock time in seconds. See also "getWallClockTimeMsec(...)".
     */
    uint64_t getWallClockTimeSec() { return getWallClockTimeMsec() / 1000U; }

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using MSec = std::chrono::milliseconds;

private:
    /**
     * \brief Updates the counter for the elapsed time
     */
    void updateTimeCounter();

    bool m_Running{false};          // Flag indicating whether or not the timer is running
    MSec m_ElapsedTime{};           // Counter of elapsed time between start and update
    TimePoint m_TimePointStart{};   // Point in time when the timer started
    TimePoint m_TimePointCurrent{}; // Current point in time
};

} // namespace tools

