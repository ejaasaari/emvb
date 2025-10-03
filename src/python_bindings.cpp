#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>

#include "DocumentScorer.hpp"

namespace py = pybind11;

std::vector<std::tuple<size_t, float>> search_query(
    DocumentScorer& scorer,
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    int k,
    size_t nprobe,
    float thresh,
    float thresh_query,
    size_t out_second_stage,
    size_t n_doc_to_score) {

    auto buf = query.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Query array must be 2-dimensional (vec_per_query, dim)");
    }

    float* query_data = static_cast<float*>(buf.ptr);

    // PHASE 1: candidate documents retrieval
    auto candidate_docs = scorer.find_candidate_docs(query_data, nprobe, thresh);

    // PHASE 2: candidate document filtering
    auto selected_docs = scorer.compute_hit_frequency(candidate_docs, thresh, n_doc_to_score);

    // PHASE 3: second stage filtering
    auto selected_docs_2nd = scorer.second_stage_filtering(selected_docs, out_second_stage);

    // PHASE 4: document scoring
    auto query_res = scorer.compute_topk_documents_selected(query_data, selected_docs_2nd, k, thresh_query);

    return query_res;
}

PYBIND11_MODULE(emvb, m) {
    m.doc() = "EMVB: Efficient Multi-Vector Retrieval with Bit Vectors";

    omp_set_num_threads(1);

    py::class_<DocumentScorer>(m, "DocumentScorer")
        .def(py::init<const std::string&, const std::string&, const size_t>(),
             py::arg("doclens_path"),
             py::arg("index_dir_path"),
             py::arg("max_query_terms"),
             "Initialize DocumentScorer with index paths and max query terms")
        .def("find_candidate_docs", &DocumentScorer::find_candidate_docs,
             py::arg("query"),
             py::arg("nprobe"),
             py::arg("thresh"),
             "Phase 1: Find candidate documents using centroid scoring")
        .def("compute_hit_frequency", &DocumentScorer::compute_hit_frequency,
             py::arg("candidate_docs"),
             py::arg("thresh"),
             py::arg("k_centroids"),
             "Phase 2: Filter candidates using hit frequency with bitvectors")
        .def("second_stage_filtering", &DocumentScorer::second_stage_filtering,
             py::arg("doc_ids"),
             py::arg("n_documents"),
             "Phase 3: Second stage filtering with centroid-interaction mechanism")
        .def("compute_topk_documents_selected",
             [](DocumentScorer& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> query,
                const std::vector<numDocsType>& doc_ids,
                size_t k,
                float thresh) {
                 auto buf = query.request();
                 float* query_data = static_cast<float*>(buf.ptr);
                 return self.compute_topk_documents_selected(query_data, doc_ids, k, thresh);
             },
             py::arg("query"),
             py::arg("doc_ids"),
             py::arg("k"),
             py::arg("thresh"),
             "Phase 4: Compute top-k documents with late-interaction scoring")
        .def_readwrite("globalCounter", &DocumentScorer::globalCounter,
                      "Counter for number of document vectors scored");

    m.def("search", &search_query,
          py::arg("scorer"),
          py::arg("query"),
          py::arg("k"),
          py::arg("nprobe"),
          py::arg("thresh"),
          py::arg("thresh_query"),
          py::arg("out_second_stage"),
          py::arg("n_doc_to_score"),
          "Perform search on a query");

    m.def("load_qids", &load_qids,
          py::arg("path"),
          "Load query IDs from a file");

    m.def("set_num_threads", [](int n) { omp_set_num_threads(n); },
          py::arg("n"),
          "Set number of OpenMP threads (default is 1 for best performance)");
}
