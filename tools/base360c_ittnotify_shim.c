#include <stdint.h>

/*
 * Minimal Intel JIT profiling shim.
 * Some PyTorch 2.3.x builds expect these symbols at load time even when
 * no profiler integration is actually used in our workflow.
 */

#ifdef __cplusplus
extern "C" {
#endif

int iJIT_NotifyEvent(int event_type, void *event_data) {
    (void)event_type;
    (void)event_data;
    return 0;
}

int iJIT_IsProfilingActive(void) {
    return 0;
}

unsigned int iJIT_GetNewMethodID(void) {
    static unsigned int next_id = 1U;
    return next_id++;
}

#ifdef __cplusplus
}
#endif
