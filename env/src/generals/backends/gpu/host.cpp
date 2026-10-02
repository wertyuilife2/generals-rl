// Small CPython bridge. Tensor operations use Python's public object protocol;
// CUDA entry points are borrowed from the already loaded runtime/kernel library.
// No libtorch ABI, pybind11, policy code, synchronization, or tensor readback.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include <cstring>
#include <vector>

struct Ref {
    PyObject *p;
    explicit Ref(PyObject *p = nullptr) : p(p) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref &) = delete;
    Ref &operator=(const Ref &) = delete;
    explicit operator bool() const { return p != nullptr; }
};
using Launch = int (*)(void *, void *, int *);
using GetDevice = int (*)(int *);
using SetDevice = int (*)(int);
using ErrorString = const char *(*)(int);
struct Context {
    char *args;
    Launch launch;
    GetDevice get_device;
    SetDevice set_device;
    ErrorString error_string;
    Py_ssize_t action_offset, reset_offset, horizon_offset;
    PyObject *refs = nullptr, *versioned = nullptr, *tensor_type = nullptr;
    PyObject *dtype = nullptr, *shape = nullptr, *device = nullptr, *device_index = nullptr;
    PyObject *stream = nullptr, *arguments = nullptr, *library = nullptr;
    std::vector<long long> versions;
    int index = 0;
    ~Context() {
        for (auto *p : {refs, versioned, tensor_type, dtype, shape, device, device_index,
                        stream, arguments, library}) Py_XDECREF(p);
    }
    bool check(int status) {
        if (!status) return true;
        PyErr_Format(PyExc_RuntimeError, "CUDA simulation: %s", error_string(status));
        return false;
    }
};
constexpr const char *CAPSULE = "generals.cuda.step";
static void destroy(PyObject *capsule) {
    delete static_cast<Context *>(PyCapsule_GetPointer(capsule, CAPSULE));
}
static bool versions_from_python(Context &c, PyObject *kernel) {
    Ref versions(PyObject_GetAttrString(kernel, "_versions"));
    if (!versions) return false;
    c.versions.resize(PyTuple_GET_SIZE(versions.p));
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(versions.p); ++i)
        c.versions[i] = PyLong_AsLongLong(PyTuple_GET_ITEM(versions.p, i));
    return !PyErr_Occurred();
}
static PyObject *bind(PyObject *, PyObject *args) {
    PyObject *kernel, *offsets;
    unsigned long long arg, launch, get, set, error;
    if (!PyArg_ParseTuple(args, "OKKKKKO", &kernel, &arg, &launch, &get, &set, &error, &offsets)) return nullptr;
    auto *c = new Context{};
    c->args = reinterpret_cast<char *>(arg);
    c->launch = reinterpret_cast<Launch>(launch);
    c->get_device = reinterpret_cast<GetDevice>(get);
    c->set_device = reinterpret_cast<SetDevice>(set);
    c->error_string = reinterpret_cast<ErrorString>(error);
    Ref env(PyObject_GetAttrString(kernel, "env"));
    if (env) {
        c->refs = PyObject_GetAttrString(kernel, "_state_refs");
        c->versioned = PyObject_GetAttrString(kernel, "_versioned");
        c->tensor_type = PyObject_GetAttrString(kernel, "_tensor_type");
        c->stream = PyObject_GetAttrString(kernel, "_stream_address");
        c->arguments = PyObject_GetAttrString(kernel, "args");
        c->library = PyObject_GetAttrString(kernel, "lib");
        c->device = PyObject_GetAttrString(env.p, "device");
        Ref wait(PyObject_GetAttrString(env.p, "_wait_actions"));
        if (wait) {
            c->dtype = PyObject_GetAttrString(wait.p, "dtype");
            c->shape = PyObject_GetAttrString(wait.p, "shape");
        }
    }
    if (c->device) c->device_index = PyObject_GetAttrString(c->device, "index");
    if (!c->refs || !c->versioned || !c->tensor_type || !c->stream || !c->arguments || !c->library ||
            !c->dtype || !c->shape || !c->device_index || !versions_from_python(*c, kernel)) {
        delete c; return nullptr;
    }
    c->index = int(PyLong_AsLong(c->device_index));
    if (!PyTuple_Check(offsets) || PyTuple_GET_SIZE(offsets) != 3) {
        delete c; PyErr_SetString(PyExc_ValueError, "expected three argument offsets"); return nullptr;
    }
    c->action_offset = PyLong_AsSsize_t(PyTuple_GET_ITEM(offsets, 0));
    c->reset_offset = PyLong_AsSsize_t(PyTuple_GET_ITEM(offsets, 1));
    c->horizon_offset = PyLong_AsSsize_t(PyTuple_GET_ITEM(offsets, 2));
    if (PyErr_Occurred()) { delete c; return nullptr; }
    PyObject *result = PyCapsule_New(c, CAPSULE, destroy);
    if (!result) delete c;
    return result;
}
struct DeviceGuard {
    Context &c;
    int previous = 0;
    bool changed = false;
    explicit DeviceGuard(Context &c) : c(c) {}
    bool enter() {
        if (!c.check(c.get_device(&previous))) return false;
        changed = previous != c.index;
        return !changed || c.check(c.set_device(c.index));
    }
    bool restore() {
        if (!changed) return true;
        changed = false;
        return c.check(c.set_device(previous));
    }
    ~DeviceGuard() { if (changed) c.set_device(previous); }
};
static bool identities(Context &c, PyObject *env) {
    Py_ssize_t index = 0;
    for (auto *name : {"state", "initial"}) {
        Ref mapping(PyObject_GetAttrString(env, name));
        if (!mapping) return false;
        // State protocol dictionaries are ordinary dicts. A custom mapping uses
        // the original prepare() path instead of bypassing its values() method.
        if (!PyDict_CheckExact(mapping.p)) return false;
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(mapping.p, &pos, &key, &value)) {
            if (index >= PyTuple_GET_SIZE(c.refs) || value != PyTuple_GET_ITEM(c.refs, index++)) {
                PyErr_SetString(PyExc_ValueError, "edit state tensors in place; replacing their storage is unsupported");
                return false;
            }
        }
    }
    if (index == PyTuple_GET_SIZE(c.refs)) return true;
    PyErr_SetString(PyExc_ValueError, "edit state tensors in place; replacing their storage is unsupported");
    return false;
}
static PyObject *step(PyObject *, PyObject *args) {
    PyObject *kernel, *actions;
    if (!PyArg_ParseTuple(args, "OO", &kernel, &actions)) return nullptr;
    Ref capsule(PyObject_GetAttrString(kernel, "_host_context"));
    if (!capsule) return nullptr;
    auto *c = static_cast<Context *>(PyCapsule_GetPointer(capsule.p, CAPSULE));
    if (!c) return nullptr;
    int tensor = PyObject_IsInstance(actions, c->tensor_type);
    if (tensor < 0) return nullptr;
    Ref dtype(tensor ? PyObject_GetAttrString(actions, "dtype") : nullptr);
    if (!tensor || (dtype && dtype.p != c->dtype)) {
        PyErr_SetString(PyExc_TypeError, "actions must be an int64 CUDA tensor"); return nullptr;
    }
    if (!dtype) return nullptr;
    Ref shape(PyObject_GetAttrString(actions, "shape")), device(PyObject_GetAttrString(actions, "device"));
    if (!shape || !device) return nullptr;
    int same_shape = PyObject_RichCompareBool(shape.p, c->shape, Py_EQ);
    int same_device = PyObject_RichCompareBool(device.p, c->device, Py_EQ);
    if (same_shape < 0 || same_device < 0) return nullptr;
    if (!same_shape || !same_device) {
        PyErr_SetString(PyExc_ValueError, "actions must have shape [batch, players] on the environment device"); return nullptr;
    }
    Ref env(PyObject_GetAttrString(kernel, "env"));
    if (!env) return nullptr;
    Ref closed(PyObject_GetAttrString(env.p, "_closed"));
    if (!closed) return nullptr;
    int is_closed = PyObject_IsTrue(closed.p);
    if (is_closed < 0) return nullptr;
    if (is_closed) { PyErr_SetString(PyExc_RuntimeError, "GPU environment is closed"); return nullptr; }
    DeviceGuard guard(*c);
    if (!guard.enter()) return nullptr;
    bool prepared = false;
    if (!identities(*c, env.p)) {
        if (PyErr_Occurred()) return nullptr;
        Ref result(PyObject_CallMethod(kernel, "prepare", nullptr));
        if (!result || !versions_from_python(*c, kernel)) return nullptr;
        prepared = true;
    }
    if (!prepared) {
        Ref dirty(PyObject_GetAttrString(kernel, "dirty"));
        if (!dirty) return nullptr;
        int stale = PyObject_IsTrue(dirty.p);
        if (stale < 0) return nullptr;
        for (Py_ssize_t i = 0; !stale && i < PyTuple_GET_SIZE(c->versioned); ++i) {
            Ref version(PyObject_GetAttrString(PyTuple_GET_ITEM(c->versioned, i), "_version"));
            if (!version) return nullptr;
            auto number = PyLong_AsLongLong(version.p);
            if (PyErr_Occurred()) return nullptr;
            stale = number != c->versions[i];
        }
        if (stale) {
            Ref result(PyObject_CallMethod(kernel, "prepare", nullptr));
            if (!result || !versions_from_python(*c, kernel)) return nullptr;
        }
    }
    Ref reset(PyObject_GetAttrString(env.p, "auto_reset")), horizon(PyObject_GetAttrString(env.p, "episode_ticks"));
    if (!reset || !horizon) return nullptr;
    int32_t reset_value = int32_t(PyLong_AsLong(reset.p));
    int64_t horizon_value = PyLong_AsLongLong(horizon.p);
    if (PyErr_Occurred()) return nullptr;
    Ref contiguous(PyObject_CallMethod(actions, "contiguous", nullptr));
    if (!contiguous) return nullptr;
    Ref data(PyObject_CallMethod(contiguous.p, "data_ptr", nullptr));
    Ref stream(PyObject_CallOneArg(c->stream, c->device_index));
    if (!data || !stream) return nullptr;
    void *data_pointer = PyLong_AsVoidPtr(data.p), *stream_pointer = PyLong_AsVoidPtr(stream.p);
    if (PyErr_Occurred()) return nullptr;
    std::memcpy(c->args + c->action_offset, &data_pointer, sizeof(data_pointer));
    std::memcpy(c->args + c->reset_offset, &reset_value, sizeof(reset_value));
    std::memcpy(c->args + c->horizon_offset, &horizon_value, sizeof(horizon_value));
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = c->launch(c->args, stream_pointer, nullptr);
    Py_END_ALLOW_THREADS
    if (!c->check(status) || !guard.restore()) return nullptr;
    return PyObject_GetAttrString(env.p, "_step_result");
}
static PyMethodDef methods[] = {
    {"bind", bind, METH_VARARGS, nullptr}, {"step", step, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_generals_cuda_host", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__generals_cuda_host() { return PyModule_Create(&module); }
