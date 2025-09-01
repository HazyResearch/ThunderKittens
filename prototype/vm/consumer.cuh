#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "util.cuh"

MAKE_WORKER(consumer, TEVENT_CONSUMER_START, true)