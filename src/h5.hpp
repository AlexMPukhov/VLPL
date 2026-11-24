// Thin RAII layer over the HDF5 C API for the openPMD files (src/openpmd.*). Only in builds with
// -DRIP_HDF5 (make HDF5=1, or found by pkg-config). Every failing call throws.
#pragma once
#ifdef RIP_HDF5
#include <hdf5.h>
#if !H5_VERSION_GE(1, 12, 0)
#error "HDF5 1.12 or newer is required (decision 113); the Makefile builds without HDF5 for an older one"
#endif
#include <cstdint>
#include <string>
#include <vector>

namespace h5 {

// Owns one HDF5 identifier (file, group, dataset, dataspace, attribute, type, property list).
class Id {
public:
    Id() = default;
    explicit Id(hid_t id, const char* what);      // throws if id < 0
    ~Id();
    Id(Id&& o) noexcept : id_(o.id_) { o.id_ = -1; }
    Id& operator=(Id&& o) noexcept;
    Id(const Id&) = delete;
    Id& operator=(const Id&) = delete;
    hid_t get() const { return id_; }
    operator hid_t() const { return id_; }
private:
    hid_t id_ = -1;
};

template <class T> hid_t mem_type();             // native memory type of double, float, int64, int32, uint64, uint8
template <class T> hid_t file_type();            // the little-endian file type

Id create_file(const std::string& path);         // truncates
Id open_file(const std::string& path);           // read-only
Id group(hid_t parent, const std::string& path); // opens, or creates with the intermediate groups
Id open_group(hid_t parent, const std::string& path);
bool exists(hid_t parent, const std::string& path);   // every link of the path
std::vector<std::string> children(hid_t group);       // names of the links, in name order
bool is_group(hid_t parent, const std::string& name);

// attributes (strings fixed-length ASCII, null-terminated, as openPMD-api writes them)
void attr(hid_t obj, const std::string& name, const std::string& v);
void attr(hid_t obj, const std::string& name, const char* v);
void attr(hid_t obj, const std::string& name, const std::vector<std::string>& v);
template <class T> void attr(hid_t obj, const std::string& name, T v);
template <class T> void attr(hid_t obj, const std::string& name, const std::vector<T>& v);
bool has_attr(hid_t obj, const std::string& name);
std::string attr_string(hid_t obj, const std::string& name);
std::vector<std::string> attr_strings(hid_t obj, const std::string& name);   // array of fixed-length strings
std::vector<double> attr_reim(hid_t obj, const std::string& name);   // real or complex values as (re, im) pairs
template <class T> std::vector<T> attr_vec(hid_t obj, const std::string& name);   // converted to T
template <class T> T attr_num(hid_t obj, const std::string& name) { return attr_vec<T>(obj, name).at(0); }

// datasets; the file type is F (e.g. float32 output of double data), the memory type is T
template <class F> Id dataset(hid_t parent, const std::string& name, const std::vector<hsize_t>& dims);
Id open_dataset(hid_t parent, const std::string& name);
std::vector<hsize_t> dims(hid_t ds);
// count values starting at start (one entry per dimension), C order in buf
template <class T> void write(hid_t ds, const T* buf, const std::vector<hsize_t>& start, const std::vector<hsize_t>& count);
template <class T> void read(hid_t ds, T* buf, const std::vector<hsize_t>& start, const std::vector<hsize_t>& count);
// a complex dataset (compound of two floats r, i, as h5py and openPMD-api write it) as (re, im) pairs
void read_complex(hid_t ds, double* reim, const std::vector<hsize_t>& start, const std::vector<hsize_t>& count);
bool is_complex(hid_t ds);
std::string type_name(hid_t ds);                 // "f64", "f32", "i64", "i32", "u64", "u8" or "other"

}  // namespace h5
#endif
