#ifndef NNLS_QP_SOLVER_TIMERS_H
#define NNLS_QP_SOLVER_TIMERS_H
#include <chrono>

namespace QP_NNLS {
using ticks_t = unsigned long long; //microseconds 
enum class TIMER_TYPES {
	WALL_CLOCK = 0,
	CPU,
};

struct TimeIntervals {
	ticks_t hours;
	ticks_t minutes;
	ticks_t sec;
	ticks_t ms;
	ticks_t mus;
};

class iTimer {
public:
	virtual ~iTimer() = default;
	virtual void Start() = 0;
	virtual ticks_t Ticks() = 0;  //number of ticks from start
	virtual void Reset() = 0;
	virtual ticks_t TimeFromStart() = 0;
	virtual void toIntervals(ticks_t time, TimeIntervals& intervals) = 0;
protected:
	iTimer() = default;
};

class wcTimer : public iTimer {
public:
	wcTimer();
	virtual ~wcTimer() = default;
	void Start() override;
	ticks_t Ticks() override;
	void Reset() override;
	void toIntervals(ticks_t time, TimeIntervals& intervals) override;
	ticks_t TimeFromStart() override;
protected:
	ticks_t ticks = 0;
	std::chrono::time_point<std::chrono::steady_clock> tStart{};
	std::chrono::time_point<std::chrono::steady_clock> tCur{};
	bool active = false;
};
}
#endif





