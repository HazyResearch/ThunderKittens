#include "mlp.cuh"

#include "down_op.cu"
#include "up_op.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

PYBIND11_MODULE(mlp, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kvm<mlp_config<2>, mlp_micro_globals,
                                 UpOp<mlp_config<2>, mlp_micro_globals>,
                                 DownOp<mlp_config<2>, mlp_micro_globals>>>(m, "mlp_2stage",
                                    &mlp_micro_globals::Bar,
                                    &mlp_micro_globals::instructions,
                                    &mlp_micro_globals::timings,

                                    &mlp_micro_globals::up_weights,
                                    &mlp_micro_globals::down_weights,
                                    
                                    &mlp_micro_globals::inputs,
                                    &mlp_micro_globals::intermediates,
                                    &mlp_micro_globals::outputs);
    kittens::py::bind_kernel<kvm<mlp_config<1>, mlp_micro_globals,
                                 UpOp<mlp_config<1>, mlp_micro_globals>,
                                 DownOp<mlp_config<1>, mlp_micro_globals>>>(m, "mlp_1stage",
                                    &mlp_micro_globals::Bar,
                                    &mlp_micro_globals::instructions,
                                    &mlp_micro_globals::timings,

                                    &mlp_micro_globals::up_weights,
                                    &mlp_micro_globals::down_weights,
                                    
                                    &mlp_micro_globals::inputs,
                                    &mlp_micro_globals::intermediates,
                                    &mlp_micro_globals::outputs);
}