#ifndef TIMER_H
#define TIMER_H
#include <chrono>

class Timer {
public:
    std::string name;
    
    // Time point variables to store start and end times of the timer
    std::chrono::steady_clock::time_point g_start;
    std::chrono::steady_clock::time_point g_end;

    Timer(std::string name) {
        this->name = name;
        start();  
    }

    // Starts the timer 
    void start() {
        g_start = std::chrono::high_resolution_clock::now();
    }

    // Stops the timer, calculates the duration, and prints the result
    void stop() {
        g_end = std::chrono::high_resolution_clock::now();  // end time
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(g_end - g_start);
        printf("%s took %lld milliseconds.\n", name.c_str(), duration.count());
    }
};

#endif // TIMER_H
