// Bind permitted observations to the existing timed/watchdog C ABI.
// The policy library still owns decision timing, timeouts and diagnostics.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdint>
#include <initializer_list>
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
    const char *names[13] = {"terrain", "structure", "owner", "army", "visible", "alive", "army", "territory", "cities", "fog_obstacles", "has_kill", "warn_afk", "neutralize_in"};
    const int types[13] = {NPY_UINT8, NPY_UINT8, NPY_INT16, NPY_INT64, NPY_UINT8, NPY_UINT8, NPY_INT64, NPY_INT64, NPY_INT64, NPY_UINT8, NPY_UINT8, NPY_UINT8, NPY_DOUBLE};
    Ref arrays[13]; void *data[13];
    for (int i = 0; i < 13; ++i) {
        Ref value(PyObject_GetAttrString(i < 6 || i >= 9 ? obs : stats.p, names[i]));
        if (!value) return nullptr;
        arrays[i].p = PyArray_FROM_OTF(value.p, types[i], NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST);
        if (!arrays[i]) return nullptr;
        auto *a = reinterpret_cast<PyArrayObject *>(arrays[i].p);
        if (PyArray_NDIM(a) != ((i < 5 || i == 9) ? 2 : 1) || PyArray_DIM(a, 0) != ((i < 5 || i == 9) ? height : players) ||
            ((i < 5 || i == 9) && PyArray_DIM(a, 1) != width)) {
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
        static_cast<const std::uint8_t *>(data[9]), static_cast<const std::uint8_t *>(data[10]),
        static_cast<const std::uint8_t *>(data[11]), static_cast<const double *>(data[12]),
        previous == Py_None ? nullptr : &move};
    i64 encoded;
    Py_BEGIN_ALLOW_THREADS
    encoded = call(instance, &native, static_cast<i64 *>(PyArray_DATA(diagnostics)));
    Py_END_ALLOW_THREADS
    return PyLong_FromLongLong(encoded);
}

// Array validation and mapping happen once per group. The timed policy calls
// and per-lane observation construction stay in this serial native loop.
static PyObject *act_batch(PyObject *, PyObject *args) {
    PyObject *handles, *address, *raw, *previous, *metrics, *output, *headers, *active_lane;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &handles, &address, &raw, &previous,
                          &metrics, &output, &headers, &active_lane)) return nullptr;
    using Act = i64 (*)(void *, const Observation *, i64 *);
    auto call = reinterpret_cast<Act>(PyLong_AsVoidPtr(address));
    if (PyErr_Occurred()) return nullptr;
    if (!PyDict_Check(raw) || !PyArray_Check(handles)) {
        PyErr_SetString(PyExc_ValueError, "invalid native batch"); return nullptr;
    }
    const npy_intp count = PyArray_SIZE(reinterpret_cast<PyArrayObject *>(handles));
    if (count < 1) { PyErr_SetString(PyExc_ValueError, "empty native batch"); return nullptr; }
    auto array = [](PyObject *value, int type, std::initializer_list<npy_intp> shape, bool write=false) -> PyObject * {
        if (!value) { PyErr_SetString(PyExc_ValueError, "missing native batch field"); return nullptr; }
        if (write && (!PyArray_Check(value) || PyArray_TYPE(reinterpret_cast<PyArrayObject *>(value)) != type ||
                      !PyArray_ISCARRAY(reinterpret_cast<PyArrayObject *>(value)))) {
            PyErr_SetString(PyExc_ValueError, "invalid native output buffer"); return nullptr;
        }
        PyObject *result = PyArray_FROM_OTF(value, type, NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST);
        if (!result) return nullptr;
        auto *a = reinterpret_cast<PyArrayObject *>(result);
        bool valid = PyArray_NDIM(a) == int(shape.size());
        int axis=0; for (auto size: shape) { if (valid && PyArray_DIM(a,axis) != size) valid=false; ++axis; }
        if (!valid) { Py_DECREF(result); PyErr_SetString(PyExc_ValueError, "invalid native batch array shape"); return nullptr; }
        return result;
    };
    auto ptr = [](Ref &value) { return PyArray_DATA(reinterpret_cast<PyArrayObject *>(value.p)); };
    Ref hs(array(handles,NPY_UINTP,{count})), ms(array(metrics,NPY_UINTP,{count})),
        ps(array(previous,NPY_INT64,{count,10})), os(array(output,NPY_INT64,{count},true)),
        hd(array(headers,NPY_UINTP,{count})), cursor(array(active_lane,NPY_INT64,{1},true));
    if (!hs || !ms || !ps || !os || !hd || !cursor) return nullptr;
    PyObject *state = PyDict_GetItemString(raw,"state");
    PyObject *w=PyDict_GetItemString(raw,"width"), *h=PyDict_GetItemString(raw,"height");
    if (!state || !PyDict_Check(state) || !w || !h) {
        PyErr_SetString(PyExc_ValueError,"invalid native observation batch"); return nullptr;
    }
    i64 width=PyLong_AsLongLong(w), height=PyLong_AsLongLong(h);
    if (PyErr_Occurred()) return nullptr;
    PyObject *alive=PyDict_GetItemString(state,"alive");
    if (!alive || !PyArray_Check(alive) || PyArray_NDIM(reinterpret_cast<PyArrayObject *>(alive)) != 2 ||
        width<1 || height<1 || width>1024*1024 || height>1024*1024 || width*height>1024*1024) {
        PyErr_SetString(PyExc_ValueError,"invalid native batch dimensions"); return nullptr;
    }
    npy_intp players=PyArray_DIM(reinterpret_cast<PyArrayObject *>(alive),1), cells=width*height;
    if (players<2 || players>8) { PyErr_SetString(PyExc_ValueError,"invalid native player count"); return nullptr; }
    const char *names[]={"terrain","structure","owner","army","visible","alive","fog_obstacles",
                         "has_kill","warn_afk","neutralize_in","player_ids","tick","general_pos","active","stats"};
    const int types[]={NPY_UINT8,NPY_UINT8,NPY_INT16,NPY_INT64,NPY_BOOL,NPY_BOOL,NPY_BOOL,
                       NPY_BOOL,NPY_BOOL,NPY_DOUBLE,NPY_INT64,NPY_INT64,NPY_INT64,NPY_BOOL,NPY_INT64};
    Ref arrays[15]; void *data[15];
    for (int i=0;i<15;++i) {
        PyObject *value=PyDict_GetItemString(i==4 || i==10 || i>=12 ? raw : state,names[i]);
        arrays[i].p = i==14 ? array(value,types[i],{count,players,3}) :
            i>=10 ? array(value,types[i],{count}) :
            array(value,types[i],{count,(i<5 || i==6) ? cells : players});
        if (!arrays[i]) return nullptr;
        data[i]=ptr(arrays[i]);
    }
    auto *instances=static_cast<std::uintptr_t *>(ptr(hs)), *diagnostics=static_cast<std::uintptr_t *>(ptr(ms)),
         *journal_headers=static_cast<std::uintptr_t *>(ptr(hd));
    auto *moves=static_cast<i64 *>(ptr(ps)), *actions=static_cast<i64 *>(ptr(os));
    auto *lane_cursor=static_cast<i64 *>(ptr(cursor));
    const auto *ids=static_cast<const i64 *>(data[10]), *ticks=static_cast<const i64 *>(data[11]),
               *generals=static_cast<const i64 *>(data[12]), *stats=static_cast<const i64 *>(data[14]);
    for (npy_intp lane=0;lane<count;++lane) if (!instances[lane] || !diagnostics[lane] || ids[lane]<0 || ids[lane]>=players) {
        PyErr_SetString(PyExc_ValueError,"invalid native batch instance/player"); return nullptr;
    }
    npy_intp failed=-1;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp lane=0;lane<count;++lane) {
        actions[lane]=0;
        if (!static_cast<const std::uint8_t *>(data[13])[lane]) continue;
        const auto base=lane*cells, seatbase=lane*players;
        i64 army[8],land[8],cities[8];
        for (int p=0;p<players;++p) {
            army[p]=stats[(seatbase+p)*3]; land[p]=stats[(seatbase+p)*3+1]; cities[p]=stats[(seatbase+p)*3+2];
        }
        const i64 *r=moves+lane*10;
        Move move{r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9]};
        Observation obs{int(width),int(height),int(players),int(ids[lane]),ticks[lane],int(generals[lane]),
            static_cast<const std::uint8_t *>(data[0])+base,static_cast<const std::uint8_t *>(data[1])+base,
            static_cast<const std::int16_t *>(data[2])+base,static_cast<const i64 *>(data[3])+base,
            static_cast<const std::uint8_t *>(data[4])+base,static_cast<const std::uint8_t *>(data[5])+seatbase,
            army,land,cities,static_cast<const std::uint8_t *>(data[6])+base,
            static_cast<const std::uint8_t *>(data[7])+seatbase,static_cast<const std::uint8_t *>(data[8])+seatbase,
            static_cast<const double *>(data[9])+seatbase,r[0] ? &move : nullptr};
        auto *m=reinterpret_cast<i64 *>(diagnostics[lane]);
        auto *header=reinterpret_cast<i64 *>(journal_headers[lane]);
        *lane_cursor=lane;
        if (header) { header[1]=ids[lane]; header[2]=ticks[lane]; header[3]=2; }
        actions[lane]=call(reinterpret_cast<void *>(instances[lane]),&obs,m);
        if (header) header[1]=-1;
        if (actions[lane]<0 || actions[lane]>cells*8) { ++m[5]; m[7]=2; }
        if (m[7]) { failed=lane; break; }
    }
    Py_END_ALLOW_THREADS
    return PyLong_FromSsize_t(failed);
}
static PyMethodDef methods[] = {{"act", act, METH_VARARGS, nullptr}, {"act_batch", act_batch, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_generals_arena_host", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__generals_arena_host() { import_array(); return PyModule_Create(&module); }
