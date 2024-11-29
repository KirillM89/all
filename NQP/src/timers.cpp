#include "timers.h"

namespace QP_NNLS {

wcTimer::wcTimer() {
	active = false;
}
void wcTimer::Start() {
	active = true;
	tStart = std::chrono::steady_clock::now();
	tCur = tStart;
}

ticks_t wcTimer::Ticks() {
	if (!active) {
		return 0;
	}
	const auto timePoint = std::chrono::steady_clock::now();
	ticks = std::chrono::duration_cast<std::chrono::microseconds>(timePoint - tCur).count();
	tCur = timePoint;
	return ticks;
}

ticks_t wcTimer::TimeFromStart() {
	return active ? std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tStart).count() : 0;
}
void wcTimer::Reset() {
	active = false;
}

void wcTimer::toIntervals(ticks_t time, TimeIntervals& intervals) {
	using ull = unsigned long long;
	constexpr ull musInHour = 1000U * 1000U * 3600U;
	constexpr ull musInMinute = 1000U * 1000U * 60U;
	constexpr ull musInSec = 1000U * 1000U;
	constexpr ull musInMs = 1000U;
	intervals.hours = time / musInHour;
	intervals.minutes = (time - intervals.hours * musInHour) / musInMinute;
	intervals.sec = (time - intervals.hours * musInHour - intervals.minutes * musInMinute) / musInSec;
	intervals.ms = (time - intervals.hours * musInHour - intervals.minutes * musInMinute - intervals.sec * musInSec) / musInMs;
	intervals.mus = time % 1000;
}

}
