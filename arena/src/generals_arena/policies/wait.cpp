#include "generals_native.h"
#include <algorithm>

extern "C" std::int64_t arena_wait_act(void *,const GeneralsBatchView *view,std::int64_t *actions) {
    std::fill_n(actions,view->count,0);
    return -1;
}

#include "decision.h"
extern "C" std::int64_t arena_wait_one(void *,const GeneralsBatchView *view,int lane,std::int64_t *metrics) {
    generals_decision::Guard guard(reinterpret_cast<generals_decision::Metrics *>(metrics),10000000,true,
                                  view->observation.tick[lane]);
    guard.done();return 0;
}
// External input is a command source, not an AI decision; it is not timed.
extern "C" std::int64_t arena_input_one(void *context,const GeneralsBatchView *,int lane,std::int64_t *metrics) {
    metrics[7]=0;
    return static_cast<const std::int64_t *>(context)[lane];
}
