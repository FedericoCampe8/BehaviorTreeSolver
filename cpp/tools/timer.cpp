#include "tools/timer.hpp"

namespace tools {

Timer::Timer(bool start)
{
    reset();
    if (start)
    {
        this->start();
    }
}

void Timer::start()
{
    m_TimePointStart = Clock::now();
    m_TimePointCurrent = m_TimePointStart;
    m_Running = true;
} // start

void Timer::stop()
{
    updateTimeCounter();
    m_Running = false;
} // stop

void Timer::reset()
{
    // Reset elapsed time
    m_ElapsedTime = m_ElapsedTime.zero();
    m_Running = false;
} // reset

void Timer::updateTimeCounter()
{
    // Timer must be running to update the counter
    if (!m_Running)
    {
        return;
    }

    TimePoint timePointNow = Clock::now();

    // Add elapsed time as the difference between now and current time point,
    // i.e., the most recent updated time point
    m_ElapsedTime += std::chrono::duration_cast<MSec>(timePointNow - m_TimePointCurrent);

    // Update the current time point to now
    m_TimePointCurrent = timePointNow;
} // updateTimeCounter

uint64_t Timer::getWallClockTimeMsec()
{
    // Update the elapsed time to this moment
    updateTimeCounter();
    return m_ElapsedTime.count();
} // getWallClockTimeMsec

} // namespace tools

