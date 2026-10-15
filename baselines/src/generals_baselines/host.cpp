// CPython/NumPy binding only. Calls the unchanged policy.cpp C ABI.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdint>

using i64 = std::int64_t;
struct Observation {
    const std::uint8_t *terrain, *structure;
    const std::int16_t *owner;
    const i64 *army;
    const std::uint8_t *visible, *alive;
    int height, width, players, player;
    i64 tick;
    int terminated;
};
struct Options {
    int max_sources, max_attempts, lifetime, memory_ttl;
    double temperature, weights[3];
};
struct Ref {
    PyObject *p;
    explicit Ref(PyObject *v = nullptr) : p(v) {}
    ~Ref() { Py_XDECREF(p); }
    explicit operator bool() const { return p != nullptr; }
};
static bool integer(PyObject *object, const char *name, i64 &out) {
    Ref value(PyObject_GetAttrString(object, name));
    if (!value) return false;
    out = PyLong_AsLongLong(value.p);
    return !PyErr_Occurred();
}
static PyObject *act(PyObject *, PyObject *args) {
    PyObject *handle, *calls, *policy, *obs;
    if (!PyArg_ParseTuple(args, "OOOO", &handle, &calls, &policy, &obs)) return nullptr;
    void *instance = PyLong_AsVoidPtr(handle);
    if (PyErr_Occurred()) return nullptr;
    if (!PyTuple_Check(calls) || PyTuple_GET_SIZE(calls) != 3) {
        PyErr_SetString(PyExc_TypeError, "invalid native policy binding"); return nullptr;
    }
    using Act = i64 (*)(void *, const Observation *, const Options *, void *);
    auto call = reinterpret_cast<Act>(PyLong_AsVoidPtr(PyTuple_GET_ITEM(calls, 0)));
    auto error = reinterpret_cast<const char *(*)()>(PyLong_AsVoidPtr(PyTuple_GET_ITEM(calls, 1)));
    void *random = PyLong_AsVoidPtr(PyTuple_GET_ITEM(calls, 2));
    if (PyErr_Occurred()) return nullptr;
    i64 height, width, player, tick, pid, size;
    if (!integer(obs, "height", height) || !integer(obs, "width", width) ||
        !integer(obs, "size", size) || !integer(obs, "player_id", player) ||
        !integer(obs, "tick", tick) || !integer(policy, "player_id", pid)) return nullptr;
    Ref alive(PyObject_GetAttrString(obs, "alive")), terminated(PyObject_GetAttrString(obs, "terminated"));
    if (!alive || !terminated) return nullptr;
    Py_ssize_t players = PyObject_Length(alive.p);
    if (players < 0) return nullptr;
    if (players < 2 || players > 8 || pid < 0 || pid >= players) {
        PyErr_SetString(PyExc_ValueError, "invalid observation player count"); return nullptr;
    }
    if (height < 1 || width < 1 || height > 1024*1024 || width > 1024*1024 ||
        height * width != size || size > 1024*1024) {
        PyErr_SetString(PyExc_ValueError, "unsupported observation dimensions"); return nullptr;
    }
    Ref arrays[6];
    const char *names[6] = {"terrain", "structure", "owner", "army", "visible", "alive"};
    int types[6] = {NPY_UINT8, NPY_UINT8, NPY_INT16, NPY_INT64, NPY_BOOL, NPY_BOOL};
    void *data[6];
    for (int i = 0; i < 6; ++i) {
        Ref value(PyObject_GetAttrString(obs, names[i]));
        if (!value) return nullptr;
        auto *a = reinterpret_cast<PyArrayObject *>(value.p);
        if (!PyArray_Check(value.p) || PyArray_TYPE(a) != types[i] || !PyArray_ISNOTSWAPPED(a) ||
            PyArray_NDIM(a) != (i == 5 ? 1 : 2) ||
            PyArray_DIM(a, 0) != (i == 5 ? players : height) || (i != 5 && PyArray_DIM(a, 1) != width)) {
            PyErr_Format(PyExc_ValueError, "invalid observation array: %s", names[i]); return nullptr;
        }
        arrays[i].p = PyArray_FROM_OTF(value.p, types[i], NPY_ARRAY_CARRAY_RO);
        if (!arrays[i]) return nullptr;
        data[i] = PyArray_DATA(reinterpret_cast<PyArrayObject *>(arrays[i].p));
    }
    Options options{};
    const char *limits[4] = {"max_sources", "max_path_attempts", "path_lifetime", "memory_ttl"};
    int *targets[4] = {&options.max_sources, &options.max_attempts, &options.lifetime, &options.memory_ttl};
    Ref zero(PyLong_FromLong(0));
    if (!zero) return nullptr;
    for (int i = 0; i < 4; ++i) {
        Ref value(PyObject_GetAttrString(policy, limits[i]));
        if (!value) return nullptr;
        int negative = PyObject_RichCompareBool(value.p, zero.p, Py_LT);
        if (negative < 0) return nullptr;
        if (negative) { PyErr_SetString(PyExc_ValueError, "planning limits cannot be negative"); return nullptr; }
        auto raw = PyLong_AsUnsignedLongLongMask(value.p);
        if (PyErr_Occurred()) return nullptr;
        *targets[i] = int(std::uint32_t(raw));  // Preserve ctypes.c_int modulo conversion.
    }
    Ref temperature(PyObject_GetAttrString(policy, "temperature")), weights(PyObject_GetAttrString(policy, "weights"));
    if (!temperature || !weights) return nullptr;
    options.temperature = PyFloat_AsDouble(temperature.p);
    if (PyErr_Occurred()) return nullptr;
    Ref values(PyObject_CallMethod(weights.p, "values", nullptr));
    if (!values) return nullptr;
    Ref sequence(PySequence_Fast(values.p, "weights must be iterable"));
    if (!sequence) return nullptr;
    Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.p);
    if (count > 3) { PyErr_SetString(PyExc_IndexError, "invalid index"); return nullptr; }
    for (Py_ssize_t i = 0; i < count; ++i) {
        options.weights[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(sequence.p, i));
        if (PyErr_Occurred()) return nullptr;
    }
    int ended = PyObject_IsTrue(terminated.p);
    if (ended < 0) return nullptr;
    Observation native{static_cast<const std::uint8_t *>(data[0]), static_cast<const std::uint8_t *>(data[1]),
        static_cast<const std::int16_t *>(data[2]), static_cast<const i64 *>(data[3]),
        static_cast<const std::uint8_t *>(data[4]), static_cast<const std::uint8_t *>(data[5]),
        int(height), int(width), int(players), int(player), tick, ended};
    i64 encoded;
    Py_BEGIN_ALLOW_THREADS
    encoded = call(instance, &native, &options, random);
    Py_END_ALLOW_THREADS
    if (encoded < 0) { PyErr_SetString(PyExc_RuntimeError, error()); return nullptr; }
    return PyLong_FromLongLong(encoded);
}
static PyMethodDef methods[] = {{"act", act, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_generals_baselines_host", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__generals_baselines_host() { import_array(); return PyModule_Create(&module); }
