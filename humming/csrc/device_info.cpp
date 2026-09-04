#define Py_LIMITED_API 0x030A0000
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cuda.h>

#include <cstdint>
#include <vector>

struct DeviceData {
  int64_t index;
  char name[256];
  int64_t sm_count;
  int64_t max_threads_per_block;
  int64_t max_threads_per_sm;
  int64_t max_registers_per_sm;
  int64_t sm_major;
  int64_t sm_minor;
  int64_t l2_cache_size;
  int64_t l1_cache_size;
  int64_t default_smem_size;
  int64_t max_smem_size;
  int64_t memory_clock_khz;
  int64_t memory_bus_width;
  int64_t sm_clock_khz;
  double memory_bandwidth_gbps;
  double base_tensorcore_tops;
};

enum DeviceAttribute : uint32_t {
  INDEX,
  NAME,
  SM_COUNT,
  MAX_THREADS_PER_BLOCK,
  MAX_THREADS_PER_SM,
  MAX_REGISTERS_PER_SM,
  SM_MAJOR,
  SM_MINOR,
  SM_VERSION,
  L2_CACHE_SIZE,
  L2_CACHE_SIZE_MB,
  L1_CACHE_SIZE,
  L1_CACHE_SIZE_KB,
  DEFAULT_SMEM_SIZE,
  MAX_SMEM_SIZE,
  MAX_SMEM_SIZE_KB,
  MEMORY_CLOCK_KHZ,
  MEMORY_BUS_WIDTH,
  SM_CLOCK_KHZ,
  MEMORY_BANDWIDTH_GBPS,
  TENSORCORE_TOPS,
  NUM_DEVICE_ATTRIBUTES,
};

PyObject *device_attributes[NUM_DEVICE_ATTRIBUTES] = {};
uint32_t common_attributes = 0;
int cached_device_count = -1;
bool cuda_initialized = false;

void clear_device_attributes() {
  for (PyObject *&value : device_attributes) Py_CLEAR(value);
  common_attributes = 0;
}

struct PyDeviceInfo {
  PyObject_HEAD
  int64_t index;
};

std::vector<PyObject *> device_infos;
PyObject *dynamic_device_info = nullptr;

bool check_cuda(CUresult result, const char *function) {
  if (result == CUDA_SUCCESS) return true;
  const char *name = "unknown";
  const char *description = "unknown CUDA error";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  PyErr_Format(
      PyExc_RuntimeError,
      "%s failed with CUDA error %d: %s (%s)",
      function,
      static_cast<int>(result),
      name,
      description);
  return false;
}

bool get_attribute(CUdevice device, CUdevice_attribute attribute, int64_t *result) {
  int value;
  if (!check_cuda(cuDeviceGetAttribute(&value, attribute, device), "cuDeviceGetAttribute")) {
    return false;
  }
  *result = value;
  return true;
}

bool initialize_cuda() {
  if (cuda_initialized) return true;
  if (!check_cuda(cuInit(0), "cuInit")) return false;
  cuda_initialized = true;
  return true;
}

bool get_device_count(int *device_count) {
  if (cached_device_count < 0) {
    if (!initialize_cuda()) return false;
    int count;
    if (!check_cuda(cuDeviceGetCount(&count), "cuDeviceGetCount")) return false;
    cached_device_count = count;
  }
  *device_count = cached_device_count;
  return true;
}

bool resolve_current_device_index(int64_t *device_index) {
  if (!initialize_cuda()) return false;
  CUcontext context;
  if (!check_cuda(cuCtxGetCurrent(&context), "cuCtxGetCurrent")) return false;
  if (context == nullptr) {
    *device_index = 0;
    return true;
  }

  CUdevice device;
  if (!check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice")) return false;
  *device_index = device;
  return true;
}

bool validate_device_index(int64_t device_index) {
  int device_count;
  if (!get_device_count(&device_count)) return false;
  if (device_index >= 0 && device_index < device_count) return true;
  PyErr_Format(PyExc_IndexError, "invalid CUDA device index %lld", static_cast<long long>(device_index));
  return false;
}

int64_t get_l1_cache_size(int64_t sm_major, int64_t sm_minor) {
  switch (sm_major * 10 + sm_minor) {
    case 75: return 96 * 1024;
    case 80:
    case 87: return 192 * 1024;
    case 86:
    case 89: return 128 * 1024;
    case 90: return 256 * 1024;
    default:
      if (sm_major == 10 || sm_major == 11) return 256 * 1024;
      if (sm_major == 12) return 128 * 1024;
      return 0;
  }
}

int64_t get_fp16_tensorcore_ops_per_clock(int64_t sm_version) {
  int64_t sm_major = sm_version / 10;
  if (sm_major == 10 || sm_major == 11) return 8192;

  switch (sm_version) {
    case 75: return 1024;
    case 80: return 2048;
    case 86: return 1024;
    case 87: return 2048;
    case 89: return 1024;
    case 90: return 4096;
    case 120:
    case 121: return 1024;
    default: return 0;
  }
}

PyObject *make_tensorcore_tops(const DeviceData &info) {
  struct TensorCoreType {
    const char *name;
    double multiplier;
    int64_t min_sm_version;
  };
  static constexpr TensorCoreType types[] = {
      {"float16", 1.0, 75},
      {"bfloat16", 1.0, 80},
      {"float8e3m4", 2.0, 100},
      {"float8e4m3", 2.0, 89},
      {"float8e5m2", 2.0, 89},
      {"float6e2m3", 2.0, 100},
      {"float6e3m2", 2.0, 100},
      {"float4e0m3", 4.0, 100},
      {"float4e2m1", 4.0, 100},
      {"int8", 2.0, 75},
      {"int4", 4.0, 75},
  };

  PyObject *result = PyDict_New();
  if (result == nullptr) return nullptr;
  int64_t sm_version = info.sm_major * 10 + info.sm_minor;
  bool half_rate_sm = sm_version == 75 || sm_version == 86 || sm_version == 89;
  for (const TensorCoreType &type : types) {
    double tops = sm_version >= type.min_sm_version ? info.base_tensorcore_tops * type.multiplier : 0;
    bool float_type = type.name[0] == 'f' || type.name[0] == 'b';
    if (half_rate_sm && float_type) tops /= 2;
    if (tops == 0) continue;
    PyObject *value = PyFloat_FromDouble(tops);
    if (value == nullptr || PyDict_SetItemString(result, type.name, value) < 0) {
      Py_XDECREF(value);
      Py_DECREF(result);
      return nullptr;
    }
    Py_DECREF(value);
  }
  return result;
}

bool query_device_data(int64_t device_index, DeviceData *info) {
  info->index = device_index;

  CUdevice device;
  bool success =
      check_cuda(cuDeviceGet(&device, device_index), "cuDeviceGet") &&
      check_cuda(cuDeviceGetName(info->name, sizeof(info->name), device), "cuDeviceGetName") &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, &info->sm_count) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &info->max_threads_per_block) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, &info->max_threads_per_sm) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, &info->max_registers_per_sm) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &info->sm_major) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, &info->sm_minor) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, &info->l2_cache_size) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, &info->default_smem_size) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, &info->max_smem_size) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, &info->memory_clock_khz) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, &info->memory_bus_width) &&
      get_attribute(device, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, &info->sm_clock_khz);
  if (!success) return false;

  int64_t sm_version = info->sm_major * 10 + info->sm_minor;
  info->l1_cache_size = get_l1_cache_size(info->sm_major, info->sm_minor);
  // GB10 reports the effective LPDDR5X data rate; discrete GPUs report the
  // memory clock and require the usual DDR factor of two.
  int64_t memory_clock_multiplier = sm_version == 121 ? 1 : 2;
  info->memory_bandwidth_gbps = info->memory_clock_khz * memory_clock_multiplier * info->memory_bus_width / 8.0 / 1e6;
  info->base_tensorcore_tops = info->sm_count * get_fp16_tensorcore_ops_per_clock(sm_version) * info->sm_clock_khz / 1e9;
  return true;
}

PyObject *make_attribute_value(const DeviceData &info, DeviceAttribute attribute) {
  switch (attribute) {
    case INDEX: return PyLong_FromLongLong(info.index);
    case NAME: return PyUnicode_FromString(info.name);
    case SM_COUNT: return PyLong_FromLongLong(info.sm_count);
    case MAX_THREADS_PER_BLOCK: return PyLong_FromLongLong(info.max_threads_per_block);
    case MAX_THREADS_PER_SM: return PyLong_FromLongLong(info.max_threads_per_sm);
    case MAX_REGISTERS_PER_SM: return PyLong_FromLongLong(info.max_registers_per_sm);
    case SM_MAJOR: return PyLong_FromLongLong(info.sm_major);
    case SM_MINOR: return PyLong_FromLongLong(info.sm_minor);
    case SM_VERSION: return PyLong_FromLongLong(info.sm_major * 10 + info.sm_minor);
    case L2_CACHE_SIZE: return PyLong_FromLongLong(info.l2_cache_size);
    case L2_CACHE_SIZE_MB: return PyFloat_FromDouble(info.l2_cache_size / 1024.0 / 1024.0);
    case L1_CACHE_SIZE: return PyLong_FromLongLong(info.l1_cache_size);
    case L1_CACHE_SIZE_KB: return PyFloat_FromDouble(info.l1_cache_size / 1024.0);
    case DEFAULT_SMEM_SIZE: return PyLong_FromLongLong(info.default_smem_size);
    case MAX_SMEM_SIZE: return PyLong_FromLongLong(info.max_smem_size);
    case MAX_SMEM_SIZE_KB: return PyFloat_FromDouble(info.max_smem_size / 1024.0);
    case MEMORY_CLOCK_KHZ: return PyLong_FromLongLong(info.memory_clock_khz);
    case MEMORY_BUS_WIDTH: return PyLong_FromLongLong(info.memory_bus_width);
    case SM_CLOCK_KHZ: return PyLong_FromLongLong(info.sm_clock_khz);
    case MEMORY_BANDWIDTH_GBPS: return PyFloat_FromDouble(info.memory_bandwidth_gbps);
    case TENSORCORE_TOPS: return make_tensorcore_tops(info);
    default: Py_UNREACHABLE();
  }
}

bool initialize_device_attributes() {
  if (device_attributes[0] != nullptr) return true;
  int device_count;
  if (!get_device_count(&device_count)) return false;
  if (device_count == 0) {
    PyErr_SetString(PyExc_RuntimeError, "no CUDA devices found");
    return false;
  }
  std::vector<DeviceData> devices(device_count);
  for (int device_index = 0; device_index < device_count; ++device_index) {
    if (!query_device_data(device_index, &devices[device_index])) return false;
  }

  for (uint32_t attribute = 0; attribute < NUM_DEVICE_ATTRIBUTES; ++attribute) {
    PyObject *values = PyTuple_New(device_count);
    if (values == nullptr) {
      clear_device_attributes();
      return false;
    }
    for (int device_index = 0; device_index < device_count; ++device_index) {
      PyObject *value = make_attribute_value(devices[device_index], static_cast<DeviceAttribute>(attribute));
      if (value == nullptr) {
        Py_DECREF(values);
        clear_device_attributes();
        return false;
      }
      PyTuple_SetItem(values, device_index, value);
    }

    bool common = true;
    for (int device_index = 1; device_index < device_count; ++device_index) {
      int equal = PyObject_RichCompareBool(PyTuple_GetItem(values, 0), PyTuple_GetItem(values, device_index), Py_EQ);
      if (equal < 0) {
        Py_DECREF(values);
        clear_device_attributes();
        return false;
      }
      if (!equal) common = false;
    }
    if (common) {
      device_attributes[attribute] = Py_NewRef(PyTuple_GetItem(values, 0));
      common_attributes |= uint32_t{1} << attribute;
      Py_DECREF(values);
    } else {
      device_attributes[attribute] = values;
    }
  }
  return true;
}

PyObject *get_device_attribute(PyDeviceInfo *self, DeviceAttribute attribute) {
  if (!initialize_device_attributes()) return nullptr;
  PyObject *value = device_attributes[attribute];
  if (!(common_attributes & (uint32_t{1} << attribute))) {
    int64_t device_index = self->index;
    if (device_index == -1 && !resolve_current_device_index(&device_index)) return nullptr;
    value = PyTuple_GetItem(value, device_index);
  }
  return attribute == TENSORCORE_TOPS ? PyDict_Copy(value) : Py_NewRef(value);
}

PyObject *DeviceInfo_get_attribute(PyObject *object, void *closure) {
  auto *self = reinterpret_cast<PyDeviceInfo *>(object);
  auto attribute = static_cast<DeviceAttribute>(reinterpret_cast<intptr_t>(closure));
  return get_device_attribute(self, attribute);
}

PyObject *DeviceInfo_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  static const char *keywords[] = {"index", nullptr};
  PyObject *argument = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:_DeviceInfo", const_cast<char **>(keywords), &argument)) {
    return nullptr;
  }

  int64_t device_index;
  if (argument == Py_None) {
    if (!resolve_current_device_index(&device_index)) return nullptr;
  } else {
    device_index = PyLong_AsLongLong(argument);
    if (PyErr_Occurred()) return nullptr;
  }
  if (device_index < -1) {
    PyErr_Format(PyExc_IndexError, "invalid CUDA device index %lld", static_cast<long long>(device_index));
    return nullptr;
  }
  if (device_index >= 0 && !validate_device_index(device_index)) return nullptr;

  PyObject **cached;
  if (device_index == -1) {
    cached = &dynamic_device_info;
  } else {
    if (device_infos.empty()) device_infos.resize(cached_device_count, nullptr);
    cached = &device_infos[device_index];
  }
  if (*cached != nullptr) return Py_NewRef(*cached);

  PyObject *object = PyType_GenericNew(type, args, kwargs);
  if (object == nullptr) return nullptr;
  auto *self = reinterpret_cast<PyDeviceInfo *>(object);
  self->index = device_index;
  *cached = Py_NewRef(object);
  return object;
}

PyObject *format_device_info(PyDeviceInfo *self) {
  PyObject *values[NUM_DEVICE_ATTRIBUTES] = {};
  for (uint32_t attribute = 0; attribute < NUM_DEVICE_ATTRIBUTES; ++attribute) {
    values[attribute] = get_device_attribute(self, static_cast<DeviceAttribute>(attribute));
    if (values[attribute] == nullptr) {
      for (PyObject *value : values) Py_XDECREF(value);
      return nullptr;
    }
  }
  PyObject *result = PyUnicode_FromFormat(
      "DeviceInfo(index=%R, name=%R, sm_count=%R, max_threads_per_block=%R, "
      "max_threads_per_sm=%R, max_registers_per_sm=%R, "
      "sm_major=%R, sm_minor=%R, sm_version=%R, "
      "l2_cache_size=%R, l2_cache_size_mb=%R, l1_cache_size=%R, l1_cache_size_kb=%R, "
      "default_smem_size=%R, max_smem_size=%R, max_smem_size_kb=%R, "
      "memory_clock_khz=%R, memory_bus_width=%R, "
      "sm_clock_khz=%R, memory_bandwidth_gbps=%R, tensorcore_tops=%R)",
      values[INDEX],
      values[NAME],
      values[SM_COUNT],
      values[MAX_THREADS_PER_BLOCK],
      values[MAX_THREADS_PER_SM],
      values[MAX_REGISTERS_PER_SM],
      values[SM_MAJOR],
      values[SM_MINOR],
      values[SM_VERSION],
      values[L2_CACHE_SIZE],
      values[L2_CACHE_SIZE_MB],
      values[L1_CACHE_SIZE],
      values[L1_CACHE_SIZE_KB],
      values[DEFAULT_SMEM_SIZE],
      values[MAX_SMEM_SIZE],
      values[MAX_SMEM_SIZE_KB],
      values[MEMORY_CLOCK_KHZ],
      values[MEMORY_BUS_WIDTH],
      values[SM_CLOCK_KHZ],
      values[MEMORY_BANDWIDTH_GBPS],
      values[TENSORCORE_TOPS]);
  for (PyObject *value : values) Py_DECREF(value);
  return result;
}

PyObject *DeviceInfo_repr(PyDeviceInfo *self) {
  return PyUnicode_FromFormat("_DeviceInfo(index=%lld)", static_cast<long long>(self->index));
}

PyObject *DeviceInfo_print(PyObject *object, PyObject *) {
  PyObject *value = format_device_info(reinterpret_cast<PyDeviceInfo *>(object));
  if (value == nullptr) return nullptr;
  PyObject *print = PyDict_GetItemString(PyEval_GetBuiltins(), "print");
  PyObject *result = PyObject_CallFunctionObjArgs(print, value, nullptr);
  Py_DECREF(value);
  return result;
}

#define DEVICE_PROPERTY(name, attribute) \
  {name, DeviceInfo_get_attribute, nullptr, nullptr, reinterpret_cast<void *>(attribute)}

PyGetSetDef DeviceInfo_properties[] = {
    DEVICE_PROPERTY("index", INDEX),
    DEVICE_PROPERTY("name", NAME),
    DEVICE_PROPERTY("sm_count", SM_COUNT),
    DEVICE_PROPERTY("max_threads_per_block", MAX_THREADS_PER_BLOCK),
    DEVICE_PROPERTY("max_threads_per_sm", MAX_THREADS_PER_SM),
    DEVICE_PROPERTY("max_registers_per_sm", MAX_REGISTERS_PER_SM),
    DEVICE_PROPERTY("sm_major", SM_MAJOR),
    DEVICE_PROPERTY("sm_minor", SM_MINOR),
    DEVICE_PROPERTY("sm_version", SM_VERSION),
    DEVICE_PROPERTY("l2_cache_size", L2_CACHE_SIZE),
    DEVICE_PROPERTY("l2_cache_size_mb", L2_CACHE_SIZE_MB),
    DEVICE_PROPERTY("l1_cache_size", L1_CACHE_SIZE),
    DEVICE_PROPERTY("l1_cache_size_kb", L1_CACHE_SIZE_KB),
    DEVICE_PROPERTY("default_smem_size", DEFAULT_SMEM_SIZE),
    DEVICE_PROPERTY("max_smem_size", MAX_SMEM_SIZE),
    DEVICE_PROPERTY("max_smem_size_kb", MAX_SMEM_SIZE_KB),
    DEVICE_PROPERTY("memory_clock_khz", MEMORY_CLOCK_KHZ),
    DEVICE_PROPERTY("memory_bus_width", MEMORY_BUS_WIDTH),
    DEVICE_PROPERTY("sm_clock_khz", SM_CLOCK_KHZ),
    DEVICE_PROPERTY("memory_bandwidth_gbps", MEMORY_BANDWIDTH_GBPS),
    DEVICE_PROPERTY("tensorcore_tops", TENSORCORE_TOPS),
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

#undef DEVICE_PROPERTY

PyMethodDef DeviceInfo_methods[] = {
    {"print", DeviceInfo_print, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_device_info", nullptr, -1, nullptr};

PyType_Slot DeviceInfo_slots[] = {
    {Py_tp_doc, const_cast<char *>("CUDA device properties")},
    {Py_tp_new, reinterpret_cast<void *>(DeviceInfo_new)},
    {Py_tp_repr, reinterpret_cast<void *>(DeviceInfo_repr)},
    {Py_tp_getset, DeviceInfo_properties},
    {Py_tp_methods, DeviceInfo_methods},
    {0, nullptr},
};

PyType_Spec DeviceInfo_spec = {
    "humming._device_info._DeviceInfo",
    sizeof(PyDeviceInfo),
    0,
    Py_TPFLAGS_DEFAULT,
    DeviceInfo_slots,
};

PyMODINIT_FUNC PyInit__device_info() {
  PyObject *result = PyModule_Create(&module);
  if (result == nullptr) return nullptr;
  PyObject *device_info_type = PyType_FromSpec(&DeviceInfo_spec);
  if (device_info_type == nullptr) {
    Py_DECREF(result);
    return nullptr;
  }
  if (PyModule_AddObject(result, "_DeviceInfo", device_info_type) < 0) {
    Py_DECREF(device_info_type);
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}
