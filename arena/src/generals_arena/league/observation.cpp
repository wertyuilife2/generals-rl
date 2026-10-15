// Observation construction only. No policy execution or game rules.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdint>
#include <cstring>
#include <vector>

struct Array {
    PyArrayObject *p = nullptr;
    ~Array() { Py_XDECREF(p); }
    bool read(PyObject *value, int dtype) {
        p = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(value, dtype, NPY_ARRAY_CARRAY_RO));
        return p != nullptr;
    }
    bool create(int ndim, npy_intp *shape, int dtype) {
        p = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(ndim, shape, dtype));
        return p != nullptr;
    }
    template<class T> T *data() { return static_cast<T *>(PyArray_DATA(p)); }
    PyObject *release() {
        PyArray_CLEARFLAGS(p, NPY_ARRAY_WRITEABLE);
        PyObject *result = reinterpret_cast<PyObject *>(p); p = nullptr; return result;
    }
};

static PyObject *observe(PyObject *, PyObject *args) {
    PyObject *values[5]; int seat;
    if (!PyArg_ParseTuple(args, "OOOOOi", &values[0], &values[1], &values[2],
                          &values[3], &values[4], &seat)) return nullptr;
    const int types[6] = {NPY_UINT8, NPY_UINT8, NPY_INT16, NPY_INT64, NPY_BOOL, NPY_BOOL};
    Array input[5], output[6];
    for (int i = 0; i < 5; ++i)
        if (!input[i].read(values[i], types[i])) return nullptr;
    if (PyArray_NDIM(input[0].p) != 2 || PyArray_NDIM(input[4].p) != 1 ||
        seat < 0 || seat >= PyArray_SIZE(input[4].p)) {
        PyErr_SetString(PyExc_ValueError, "invalid observation dimensions or seat"); return nullptr;
    }
    npy_intp h = PyArray_DIM(input[0].p, 0), w = PyArray_DIM(input[0].p, 1);
    for (int i = 1; i < 4; ++i) {
        if (PyArray_NDIM(input[i].p) != 2 || PyArray_DIM(input[i].p, 0) != h ||
            PyArray_DIM(input[i].p, 1) != w) {
            PyErr_SetString(PyExc_ValueError, "observation boards must share a shape"); return nullptr;
        }
    }
    for (int i = 0; i < 6; ++i)
        if (!output[i].create(i == 5 ? 1 : 2, PyArray_DIMS(input[i == 5 ? 4 : 0].p), types[i]))
            return nullptr;
    auto owner = input[2].data<std::int16_t>();
    auto terrain = input[0].data<std::uint8_t>(), structure = input[1].data<std::uint8_t>();
    auto army = input[3].data<std::int64_t>();
    auto out_terrain = output[0].data<std::uint8_t>(), out_structure = output[1].data<std::uint8_t>();
    auto out_owner = output[2].data<std::int16_t>();
    auto out_army = output[3].data<std::int64_t>();
    auto visible = output[4].data<npy_bool>();
    try {
        std::vector<std::uint8_t> horizontal(h * w);
        for (npy_intp y = 0; y < h; ++y) for (npy_intp x = 0; x < w; ++x) {
            npy_intp c = y * w + x;
            horizontal[c] = owner[c] == seat || (x && owner[c - 1] == seat) ||
                            (x + 1 < w && owner[c + 1] == seat);
        }
        for (npy_intp y = 0; y < h; ++y) for (npy_intp x = 0; x < w; ++x) {
            npy_intp c = y * w + x;
            bool seen = horizontal[c] || (y && horizontal[c - w]) || (y + 1 < h && horizontal[c + w]);
            visible[c] = seen;
            out_terrain[c] = seen ? terrain[c] : 255;
            out_structure[c] = seen ? structure[c] : 255;
            out_owner[c] = seen ? owner[c] : -3;
            out_army[c] = seen ? army[c] : -1;
        }
    } catch (const std::bad_alloc &) { return PyErr_NoMemory(); }
    std::memcpy(PyArray_DATA(output[5].p), PyArray_DATA(input[4].p), PyArray_NBYTES(input[4].p));
    PyObject *result = PyTuple_New(6);
    if (!result) return nullptr;
    for (int i = 0; i < 6; ++i) PyTuple_SET_ITEM(result, i, output[i].release());
    return result;
}

static PyMethodDef methods[] = {{"observe", observe, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_generals_arena_observation", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__generals_arena_observation() { import_array(); return PyModule_Create(&module); }
