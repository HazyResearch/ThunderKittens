
#pragma once

#include "util.cuh"

namespace kittens
{
    namespace prototype
    {
        namespace vm
        {

            template <typename config>
            struct NoOp
            {
                static constexpr int opcode = 0;

                struct controller
                {
                    template <typename globals>
                    static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
                    {
                        return query;
                    }
                    template <typename globals>
                    static __device__ int init_semaphores(const globals &g, state<config> &s)
                    {
                        return 0;
                    }
                };
                struct loader
                {
                    template <typename globals>
                    static __device__ void run(const globals &g, state<config> &s)
                    {
                        if (laneid() < config::NUM_PAGES)
                        { // Release all pages, ASAP.
                            auto pid = s.pid(laneid());
                            s.wait_page_ready(pid);
                            s.finish_page(pid, config::NUM_CONSUMER_WARPS);
                        }
                    }
                };
                struct launcher
                { // launches mma's
                    // launcher does nothing here, since this doesn't use tensor cores.
                    template <typename globals>
                    static __device__ void run(const globals &g, state<config> &s)
                    {
                        s.wait_tensor_ready();
                        if (laneid() == 0)
                            arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
                    }
                };
                struct consumer
                {
                    template <typename globals>
                    static __device__ void run(const globals &g, state<config> &s) {}
                };
                struct storer
                {
                    // Uses 4 full pages for outputs.
                    template <typename globals>
                    static __device__ void run(const globals &g, state<config> &s) {}
                };
            };

        } // namespace vm
    } // namespace prototype
} // namespace kittens