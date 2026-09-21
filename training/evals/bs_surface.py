"""What is BrittainScript's actual builtin surface?

`print` is not defined, so the language is smaller than Python and an eval that
generates Python-shaped code would score badly for the wrong reason. Find the
real builtins and the error strings, since the interpreter exits 0 even on a
syntax error and the eval has to detect failure from text.
"""
import inspect
import os
import pkgutil

import brittainscript

print("package:", os.path.dirname(brittainscript.__file__))
print("version:", getattr(brittainscript, "__version__", "?"))
print("submodules:", [m.name for m in pkgutil.iter_modules(brittainscript.__path__)])
print()

# Builtins are usually a dict or a registry somewhere in the package.
for mod in pkgutil.iter_modules(brittainscript.__path__):
    try:
        m = __import__("brittainscript." + mod.name, fromlist=["*"])
    except Exception:
        continue
    for attr in dir(m):
        if attr.startswith("_"):
            continue
        val = getattr(m, attr)
        if isinstance(val, dict) and len(val) > 3 and all(isinstance(k, str) for k in val):
            keys = sorted(val)
            if any(k in keys for k in ("len", "range", "pyimport", "str", "abs")):
                print("builtins in %s.%s (%d):" % (mod.name, attr, len(keys)))
                print("   ", keys)
                print()

# Error strings the eval will have to match.
src = ""
for mod in pkgutil.iter_modules(brittainscript.__path__):
    try:
        m = __import__("brittainscript." + mod.name, fromlist=["*"])
        src += inspect.getsource(m)
    except Exception:
        pass
import re
errs = sorted(set(re.findall(r'["\'](Syntax error[^"\']*|Undefined \w+[^"\']*|\w*Error[^"\']*)["\']', src)))
print("error message fragments (%d):" % len(errs))
for e in errs[:25]:
    print("   ", e[:90])
