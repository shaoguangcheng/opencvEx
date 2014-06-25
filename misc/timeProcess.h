#ifndef TIMEPROCESS_H
#define TIMEPROCESS_H

#include <time.h>
#include <stdio.h>
#include <string>

#ifdef TIMEPROCESS_H
class LogTime
{
public:
   LogTime(const std::string& tag)
	{
		_tag = tag;
//        _t = clock();
        timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        _t = (long int)t1.tv_sec*1000000000 + t1.tv_nsec;
	}
	~LogTime()
	{
       timespec t2;
       clock_gettime(CLOCK_MONOTONIC, &t2);
       long int t = (long int)t2.tv_sec*1000000000 + t2.tv_nsec;
        printf("%s : %.2fms\n", _tag.c_str(), (float)(t - _t)/(float)CLOCKS_PER_SEC * 1000);
	}
private:
	std::string _tag;
    long int _t;
};
 
#define LOG_TIME(tag) LogTime __lt__ = LogTime(tag);
#else
	#define LOG_TIME(tag)
#endif

#endif // TIMEPROCESS_H
