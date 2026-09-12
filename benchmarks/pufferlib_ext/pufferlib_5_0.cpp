// Python module init stub for the PufferLib 5.0 JIT extension -- mirrors
// pufferlib.cpp's PyInit__C pattern (see that file), just under a distinct
// module name (_C_5_0) so this doesn't collide with the cached 3.0.0 build's
// `_C` module in the same torch extensions build directory.
#include <Python.h>

extern "C" {
  PyObject* PyInit__C_5_0(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C_5_0",
          NULL,
          -1,
          NULL,
      };
      return PyModule_Create(&module_def);
  }
}
