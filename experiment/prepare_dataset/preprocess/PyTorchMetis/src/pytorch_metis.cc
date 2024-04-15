#include "impl.h"
PYBIND11_MODULE(pytorch_metis, m) {
    m.def("metis_assignment", &metis_assignment, "Partition a graph into k-parts using Metis ");
    m.def("expand_indptr", &expand_indptr, "Expand an indptr to the source edges");
    m.def("compact_indptr", &compact_indptr, "Compact an indptr by removing edges with false flags");
    m.def("make_sym", &make_sym, "Convert csr graph to symmetric graph (with edges to both directions)");
};