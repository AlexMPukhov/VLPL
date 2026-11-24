#include "h5.hpp"
#ifdef RIP_HDF5
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace h5 {

static void fail(const std::string& what) { throw std::runtime_error("HDF5: " + what); }
static void chk(herr_t r, const std::string& what) { if (r < 0) fail(what); }

Id::Id(hid_t id, const char* what) : id_(id) { if (id < 0) fail(what); }
Id::~Id() {
    if (id_ < 0) return;
    switch (H5Iget_type(id_)) {
    case H5I_FILE: H5Fclose(id_); break;
    case H5I_GROUP: H5Gclose(id_); break;
    case H5I_DATASET: H5Dclose(id_); break;
    case H5I_DATASPACE: H5Sclose(id_); break;
    case H5I_ATTR: H5Aclose(id_); break;
    case H5I_DATATYPE: H5Tclose(id_); break;
    case H5I_GENPROP_LST: H5Pclose(id_); break;
    default: break;
    }
}
Id& Id::operator=(Id&& o) noexcept {
    if (this != &o) { Id tmp(std::move(*this)); id_ = o.id_; o.id_ = -1; }
    return *this;
}

template <> hid_t mem_type<double>() { return H5T_NATIVE_DOUBLE; }
template <> hid_t mem_type<float>() { return H5T_NATIVE_FLOAT; }
template <> hid_t mem_type<long long>() { return H5T_NATIVE_LLONG; }
template <> hid_t mem_type<int>() { return H5T_NATIVE_INT; }
template <> hid_t mem_type<unsigned long long>() { return H5T_NATIVE_ULLONG; }
template <> hid_t mem_type<unsigned>() { return H5T_NATIVE_UINT; }
template <> hid_t mem_type<unsigned char>() { return H5T_NATIVE_UCHAR; }
template <> hid_t file_type<double>() { return H5T_IEEE_F64LE; }
template <> hid_t file_type<float>() { return H5T_IEEE_F32LE; }
template <> hid_t file_type<long long>() { return H5T_STD_I64LE; }
template <> hid_t file_type<int>() { return H5T_STD_I32LE; }
template <> hid_t file_type<unsigned long long>() { return H5T_STD_U64LE; }
template <> hid_t file_type<unsigned>() { return H5T_STD_U32LE; }
template <> hid_t file_type<unsigned char>() { return H5T_STD_U8LE; }

Id create_file(const std::string& path) {
    return Id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), ("cannot create " + path).c_str());
}
Id open_file(const std::string& path) {
    return Id(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), ("cannot open " + path).c_str());
}

bool exists(hid_t parent, const std::string& path) {
    size_t p = 0;
    while (true) {
        const size_t q = path.find('/', p);
        const std::string part = path.substr(0, q);
        if (!part.empty() && H5Lexists(parent, part.c_str(), H5P_DEFAULT) <= 0) return false;
        if (q == std::string::npos) return true;
        p = q + 1;
    }
}

// HDF5 1.12 API (decision 113): H5Literate2 / H5L_info2_t, H5Oget_info_by_name3 / H5O_info2_t
static herr_t add_name(hid_t, const char* name, const H5L_info2_t*, void* v) {
    static_cast<std::vector<std::string>*>(v)->push_back(name);
    return 0;
}
std::vector<std::string> children(hid_t g) {
    std::vector<std::string> v;
    chk(H5Literate2(g, H5_INDEX_NAME, H5_ITER_INC, nullptr, add_name, &v), "list group");
    return v;
}
bool is_group(hid_t parent, const std::string& name) {
    H5O_info2_t info;
    chk(H5Oget_info_by_name3(parent, name.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT), "no object " + name);
    return info.type == H5O_TYPE_GROUP;
}

Id group(hid_t parent, const std::string& path) {
    if (exists(parent, path)) return open_group(parent, path);
    Id lcpl(H5Pcreate(H5P_LINK_CREATE), "property list");
    chk(H5Pset_create_intermediate_group(lcpl, 1), "intermediate groups");
    return Id(H5Gcreate2(parent, path.c_str(), lcpl, H5P_DEFAULT, H5P_DEFAULT), ("cannot create group " + path).c_str());
}
Id open_group(hid_t parent, const std::string& path) {
    return Id(H5Gopen2(parent, path.c_str(), H5P_DEFAULT), ("no group " + path).c_str());
}

static void replace_attr(hid_t obj, const std::string& name) {
    if (H5Aexists(obj, name.c_str()) > 0) chk(H5Adelete(obj, name.c_str()), "delete attribute " + name);
}

static Id string_type(size_t len) {
    Id t(H5Tcopy(H5T_C_S1), "string type");
    chk(H5Tset_size(t, len), "string size");
    chk(H5Tset_strpad(t, H5T_STR_NULLTERM), "string pad");
    return t;
}

void attr(hid_t obj, const std::string& name, const std::string& v) {
    replace_attr(obj, name);
    Id t = string_type(v.size() + 1);
    Id sp(H5Screate(H5S_SCALAR), "dataspace");
    Id a(H5Acreate2(obj, name.c_str(), t, sp, H5P_DEFAULT, H5P_DEFAULT), ("attribute " + name).c_str());
    chk(H5Awrite(a, t, v.c_str()), "write attribute " + name);
}
void attr(hid_t obj, const std::string& name, const char* v) { attr(obj, name, std::string(v)); }

void attr(hid_t obj, const std::string& name, const std::vector<std::string>& v) {
    replace_attr(obj, name);
    size_t len = 1;
    for (const std::string& s : v) len = std::max(len, s.size() + 1);
    std::vector<char> buf(len * v.size(), '\0');
    for (size_t i = 0; i < v.size(); ++i) std::memcpy(&buf[i * len], v[i].data(), v[i].size());
    Id t = string_type(len);
    const hsize_t n = v.size();
    Id sp(H5Screate_simple(1, &n, nullptr), "dataspace");
    Id a(H5Acreate2(obj, name.c_str(), t, sp, H5P_DEFAULT, H5P_DEFAULT), ("attribute " + name).c_str());
    chk(H5Awrite(a, t, buf.data()), "write attribute " + name);
}

template <class T> void attr(hid_t obj, const std::string& name, T v) {
    replace_attr(obj, name);
    Id sp(H5Screate(H5S_SCALAR), "dataspace");
    Id a(H5Acreate2(obj, name.c_str(), file_type<T>(), sp, H5P_DEFAULT, H5P_DEFAULT), ("attribute " + name).c_str());
    chk(H5Awrite(a, mem_type<T>(), &v), "write attribute " + name);
}
template <class T> void attr(hid_t obj, const std::string& name, const std::vector<T>& v) {
    replace_attr(obj, name);
    const hsize_t n = v.size();
    Id sp(H5Screate_simple(1, &n, nullptr), "dataspace");
    Id a(H5Acreate2(obj, name.c_str(), file_type<T>(), sp, H5P_DEFAULT, H5P_DEFAULT), ("attribute " + name).c_str());
    chk(H5Awrite(a, mem_type<T>(), v.data()), "write attribute " + name);
}

bool has_attr(hid_t obj, const std::string& name) { return H5Aexists(obj, name.c_str()) > 0; }

std::string attr_string(hid_t obj, const std::string& name) {
    Id a(H5Aopen(obj, name.c_str(), H5P_DEFAULT), ("no attribute " + name).c_str());
    Id t(H5Aget_type(a), "attribute type");
    if (H5Tget_class(t) != H5T_STRING || H5Tis_variable_str(t) > 0) fail("attribute " + name + " is no fixed-length string");
    std::vector<char> buf(H5Tget_size(t) + 1, '\0');
    chk(H5Aread(a, t, buf.data()), "read attribute " + name);
    return std::string(buf.data());
}

std::vector<std::string> attr_strings(hid_t obj, const std::string& name) {
    Id a(H5Aopen(obj, name.c_str(), H5P_DEFAULT), ("no attribute " + name).c_str());
    Id t(H5Aget_type(a), "attribute type");
    Id sp(H5Aget_space(a), "attribute space");
    if (H5Tget_class(t) != H5T_STRING || H5Tis_variable_str(t) > 0) fail("attribute " + name + " is no fixed-length string");
    const size_t len = H5Tget_size(t), m = size_t(H5Sget_simple_extent_npoints(sp));
    std::vector<char> buf(len * m + 1, '\0');
    chk(H5Aread(a, t, buf.data()), "read attribute " + name);
    std::vector<std::string> v;
    for (size_t i = 0; i < m; ++i) v.push_back(std::string(&buf[i * len], strnlen(&buf[i * len], len)));
    return v;
}

std::vector<double> attr_reim(hid_t obj, const std::string& name) {
    Id a(H5Aopen(obj, name.c_str(), H5P_DEFAULT), ("no attribute " + name).c_str());
    Id t(H5Aget_type(a), "attribute type");
    Id sp(H5Aget_space(a), "attribute space");
    const size_t n = size_t(H5Sget_simple_extent_npoints(sp));
    std::vector<double> v(2 * n, 0.0);
    if (H5Tget_class(t) == H5T_COMPOUND && H5Tget_nmembers(t) == 2) {
        char* n0 = H5Tget_member_name(t, 0);
        char* n1 = H5Tget_member_name(t, 1);
        Id mt(H5Tcreate(H5T_COMPOUND, 16), "complex type");
        chk(H5Tinsert(mt, n0, 0, H5T_NATIVE_DOUBLE), "complex member");
        chk(H5Tinsert(mt, n1, 8, H5T_NATIVE_DOUBLE), "complex member");
        H5free_memory(n0);
        H5free_memory(n1);
        chk(H5Aread(a, mt, v.data()), "read attribute " + name);
        return v;
    }
    std::vector<double> r(n);
    chk(H5Aread(a, H5T_NATIVE_DOUBLE, r.data()), "read attribute " + name);
    for (size_t i = 0; i < n; ++i) v[2 * i] = r[i];
    return v;
}

template <class T> std::vector<T> attr_vec(hid_t obj, const std::string& name) {
    Id a(H5Aopen(obj, name.c_str(), H5P_DEFAULT), ("no attribute " + name).c_str());
    Id sp(H5Aget_space(a), "attribute space");
    const hssize_t n = H5Sget_simple_extent_npoints(sp);
    std::vector<T> v(size_t(n > 0 ? n : 0));
    if (n > 0) chk(H5Aread(a, mem_type<T>(), v.data()), "read attribute " + name);
    return v;
}

template <class F> Id dataset(hid_t parent, const std::string& name, const std::vector<hsize_t>& d) {
    Id sp(H5Screate_simple(int(d.size()), d.data(), nullptr), "dataspace");
    Id lcpl(H5Pcreate(H5P_LINK_CREATE), "property list");
    chk(H5Pset_create_intermediate_group(lcpl, 1), "intermediate groups");
    return Id(H5Dcreate2(parent, name.c_str(), file_type<F>(), sp, lcpl, H5P_DEFAULT, H5P_DEFAULT),
              ("cannot create dataset " + name).c_str());
}
Id open_dataset(hid_t parent, const std::string& name) {
    return Id(H5Dopen2(parent, name.c_str(), H5P_DEFAULT), ("no dataset " + name).c_str());
}
std::vector<hsize_t> dims(hid_t ds) {
    Id sp(H5Dget_space(ds), "dataspace");
    const int r = H5Sget_simple_extent_ndims(sp);
    std::vector<hsize_t> d(size_t(r > 0 ? r : 0));
    if (r > 0) H5Sget_simple_extent_dims(sp, d.data(), nullptr);
    return d;
}

static void io(hid_t ds, hid_t mt, void* buf, const std::vector<hsize_t>& start, const std::vector<hsize_t>& count, bool w) {
    hsize_t total = 1;
    for (hsize_t c : count) total *= c;
    if (total == 0) return;
    Id fs(H5Dget_space(ds), "dataspace");
    chk(H5Sselect_hyperslab(fs, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr), "hyperslab");
    Id ms(H5Screate_simple(int(count.size()), count.data(), nullptr), "memory space");
    if (w) chk(H5Dwrite(ds, mt, ms, fs, H5P_DEFAULT, buf), "write dataset");
    else chk(H5Dread(ds, mt, ms, fs, H5P_DEFAULT, buf), "read dataset");
}
template <class T> void write(hid_t ds, const T* buf, const std::vector<hsize_t>& start, const std::vector<hsize_t>& count) {
    io(ds, mem_type<T>(), const_cast<T*>(buf), start, count, true);
}
template <class T> void read(hid_t ds, T* buf, const std::vector<hsize_t>& start, const std::vector<hsize_t>& count) {
    io(ds, mem_type<T>(), buf, start, count, false);
}

bool is_complex(hid_t ds) {
    Id t(H5Dget_type(ds), "dataset type");
    return H5Tget_class(t) == H5T_COMPOUND && H5Tget_nmembers(t) == 2;
}
void read_complex(hid_t ds, double* reim, const std::vector<hsize_t>& start, const std::vector<hsize_t>& count) {
    if (!is_complex(ds)) fail("dataset is not complex (a compound of r and i)");
    Id t(H5Dget_type(ds), "dataset type");
    char* n0 = H5Tget_member_name(t, 0);
    char* n1 = H5Tget_member_name(t, 1);
    const std::string a = n0 ? n0 : "", b = n1 ? n1 : "";
    H5free_memory(n0);
    H5free_memory(n1);
    Id mt(H5Tcreate(H5T_COMPOUND, 16), "complex type");   // members matched by name: the file's own names
    chk(H5Tinsert(mt, a.c_str(), 0, H5T_NATIVE_DOUBLE), "complex member");
    chk(H5Tinsert(mt, b.c_str(), 8, H5T_NATIVE_DOUBLE), "complex member");
    io(ds, mt, reim, start, count, false);
}

std::string type_name(hid_t ds) {
    Id t(H5Dget_type(ds), "dataset type");
    const H5T_class_t c = H5Tget_class(t);
    const size_t s = H5Tget_size(t);
    if (c == H5T_FLOAT) return s == 8 ? "f64" : (s == 4 ? "f32" : "other");
    if (c == H5T_INTEGER) {
        const bool sg = H5Tget_sign(t) == H5T_SGN_2;
        if (s == 8) return sg ? "i64" : "u64";
        if (s == 4) return sg ? "i32" : "other";
        if (s == 1 && !sg) return "u8";
    }
    return "other";
}

#define RIP_H5_TYPES(X) X(double) X(float) X(long long) X(int) X(unsigned long long) X(unsigned) X(unsigned char)
#define RIP_H5_INST(T)                                                                                       \
    template void attr<T>(hid_t, const std::string&, T);                                                     \
    template void attr<T>(hid_t, const std::string&, const std::vector<T>&);                                 \
    template std::vector<T> attr_vec<T>(hid_t, const std::string&);                                          \
    template Id dataset<T>(hid_t, const std::string&, const std::vector<hsize_t>&);                          \
    template void write<T>(hid_t, const T*, const std::vector<hsize_t>&, const std::vector<hsize_t>&);       \
    template void read<T>(hid_t, T*, const std::vector<hsize_t>&, const std::vector<hsize_t>&);
RIP_H5_TYPES(RIP_H5_INST)

}  // namespace h5
#endif
