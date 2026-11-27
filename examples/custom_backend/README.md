# Custom backend example

`conjectra_echo_backend.CompileCheckBackend` is a complete backend plugin in ~60 lines. It needs no
model: reviews byte-compile the snapshot and return a structured verdict, write calls append a
marker to the allowed files. Replace the body of `invoke()` with a call to your model or tool.

Three ways to register it (see [docs/backends.md](../../docs/backends.md)):

```bash
# a) installed package with an entry point (pyproject.toml in this folder)
pip install -e examples/custom_backend
# b) no install: environment variable
export PYTHONPATH=examples/custom_backend CONJECTRA_BACKENDS="compile_check=conjectra_echo_backend:CompileCheckBackend"
# c) in code
python -c "from conjectra.backends import register_backend; from conjectra_echo_backend import CompileCheckBackend; register_backend('compile_check', CompileCheckBackend)"
```

Then send every model call to it and run the quickstart:

```bash
PYTHONPATH=examples/custom_backend \
CONJECTRA_BACKENDS="compile_check=conjectra_echo_backend:CompileCheckBackend" \
CONJECTRA_BACKEND_OVERRIDE=compile_check \
./examples/quickstart/run.sh
```

(The quickstart's own `toy` backend is replaced; the run now shows reviews from this plugin.)
