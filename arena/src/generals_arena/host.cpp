// Bind permitted observations to the existing timed/watchdog C ABI.
// The policy library still owns decision timing, timeouts and diagnostics.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdint>
#include "sdk/arena_policy.h"

using i64 = std::int64_t;
using Move = ArenaMoveResult;
using Observation = ArenaObservation;
struct Ref {
    PyObject *p;
    explicit Ref(PyObject *value = nullptr) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    explicit operator bool() const { return p != nullptr; }
};
static bool integer(PyObject *object, const char *name, i64 &out) {
    Ref value(PyObject_GetAttrString(object, name));
    if (!value) return false;
    out = PyLong_AsLongLong(value.p); return !PyErr_Occurred();
}
static PyObject *act(PyObject *, PyObject *args) {
    PyObject *handle, *address, *obs, *previous, *metrics;
    if (!PyArg_ParseTuple(args, "OOOOO", &handle, &address, &obs, &previous, &metrics)) return nullptr;
    void *instance = PyLong_AsVoidPtr(handle);
    using Act = i64 (*)(void *, const Observation *, i64 *);
    auto call = reinterpret_cast<Act>(PyLong_AsVoidPtr(address));
    if (PyErr_Occurred()) return nullptr;
    auto *diagnostics = reinterpret_cast<PyArrayObject *>(metrics);
    if (!PyArray_Check(metrics) || PyArray_TYPE(diagnostics) != NPY_INT64 ||
        !PyArray_ISCARRAY(diagnostics) || !PyArray_ISNOTSWAPPED(diagnostics) ||
        PyArray_SIZE(diagnostics) != 8) {
        PyErr_SetString(PyExc_ValueError, "invalid native diagnostics buffer"); return nullptr;
    }
    i64 width, height, player, tick, general;
    if (!integer(obs, "width", width) || !integer(obs, "height", height) ||
        !integer(obs, "player_id", player) || !integer(obs, "tick", tick) ||
        !integer(obs, "general_pos", general)) return nullptr;
    Ref alive(PyObject_GetAttrString(obs, "alive")), stats(PyObject_GetAttrString(obs, "stats"));
    if (!alive || !stats) return nullptr;
    Py_ssize_t players = PyObject_Length(alive.p);
    if (players < 0) return nullptr;
    if (players < 2 || players > 8 || width < 1 || height < 1 ||
        width > 1024*1024 || height > 1024*1024 || width * height > 1024*1024) {
        PyErr_SetString(PyExc_ValueError, "invalid native observation dimensions"); return nullptr;
    }
    const char *names[9] = {"terrain", "structure", "owner", "army", "visible", "alive", "army", "territory", "cities"};
    const int types[9] = {NPY_UINT8, NPY_UINT8, NPY_INT16, NPY_INT64, NPY_UINT8, NPY_UINT8, NPY_INT64, NPY_INT64, NPY_INT64};
    Ref arrays[9]; void *data[9];
    for (int i = 0; i < 9; ++i) {
        Ref value(PyObject_GetAttrString(i < 6 ? obs : stats.p, names[i]));
        if (!value) return nullptr;
        arrays[i].p = PyArray_FROM_OTF(value.p, types[i], NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST);
        if (!arrays[i]) return nullptr;
        auto *a = reinterpret_cast<PyArrayObject *>(arrays[i].p);
        if (PyArray_NDIM(a) != (i < 5 ? 2 : 1) || PyArray_DIM(a, 0) != (i < 5 ? height : players) ||
            (i < 5 && PyArray_DIM(a, 1) != width)) {
            PyErr_Format(PyExc_ValueError, "invalid native observation array: %s", names[i]); return nullptr;
        }
        data[i] = PyArray_DATA(a);
    }
    Move move{};
    if (previous != Py_None) {
        Ref sequence(PySequence_Fast(previous, "previous move must be a sequence"));
        if (!sequence) return nullptr;
        if (PySequence_Fast_GET_SIZE(sequence.p) != 9) {
            PyErr_SetString(PyExc_ValueError, "invalid previous move"); return nullptr;
        }
        i64 fields[9];
        for (int i = 0; i < 9; ++i) {
            fields[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(sequence.p, i));
            if (PyErr_Occurred()) return nullptr;
        }
        move = {fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7], fields[8]};
    }
    Observation native{int(width), int(height), int(players), int(player), tick, int(general),
        static_cast<const std::uint8_t *>(data[0]), static_cast<const std::uint8_t *>(data[1]),
        static_cast<const std::int16_t *>(data[2]), static_cast<const i64 *>(data[3]),
        static_cast<const std::uint8_t *>(data[4]), static_cast<const std::uint8_t *>(data[5]),
        static_cast<const i64 *>(data[6]), static_cast<const i64 *>(data[7]), static_cast<const i64 *>(data[8]),
        previous == Py_None ? nullptr : &move};
    i64 encoded;
    Py_BEGIN_ALLOW_THREADS
    encoded = call(instance, &native, static_cast<i64 *>(PyArray_DATA(diagnostics)));
    Py_END_ALLOW_THREADS
    return PyLong_FromLongLong(encoded);
}
static PyMethodDef methods[] = {{"act", act, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_generals_arena_host", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__generals_arena_host() { import_array(); return PyModule_Create(&module); }
