# KohakuRAG setup (uv, GB10/ARM notes)

This is a pragmatic setup guide for getting KohakuRAG running with `uv`.
It reflects the fixes needed on Dell Pro Max GB10 (ARM) where `kohakuvault` may fail to build from PyPI.

This guide intentionally removes `umap-learn` from core dependencies to avoid pulling in `numba -> llvmlite`, which can fail on newer Python versions and some architectures.

Python 3.10+ required. Python 3.11 recommended.

## 1) Clone KohakuRAG

```bash
git clone https://github.com/KohakuBlueleaf/KohakuRAG.git
cd KohakuRAG
```

## 2) Create venv with uv

```bash
uv venv --python 3.11
source .venv/bin/activate
python --version
```

## 3) Pre-step: remove umap-learn (avoid llvmlite/numba issues on GB10 for now...)

Run this from the KohakuRAG repo root.

```bash
python # start python interpreter
```

```python
from pathlib import Path

p = Path("pyproject.toml")
txt = p.read_text().splitlines()

# drop any dependency line containing umap-learn
out = [ln for ln in txt if "umap-learn" not in ln]

p.write_text("\n".join(out) + "\n")
print("Removed umap-learn from pyproject.toml")
exit()
```

Verify work
```bash
cat pyproject.toml
```


If you want UMAP later, install it manually in your venv:

```bash
# uv pip install umap-learn
```

## 4) Install KohakuRAG

```bash
uv pip install -e .
```

On x86 this often works as-is.

On ARM (GB10), this may fail when building `kohakuvault` from PyPI with a Rust error about `expected u8, found i8` (or similar signedness mismatch). If that happens, continue to the next section to build KohakuVault locally.

## 5) GB10/ARM workaround: build KohakuVault locally (patched)

KohakuRAG depends on `kohakuvault`. Normally it is installed from PyPI automatically.
On GB10/ARM, build it from source and apply a small patch.

### 5.1 Clone KohakuVault

From the parent directory that contains `KohakuRAG/`:

```bash
cd ..
git clone https://github.com/KohakuBlueleaf/KohakuVault.git
cd KohakuVault
```

### 5.2 Patch `src/kvault-rust/lib.rs` programmatically

This patch:
- ensures `use std::ffi::c_char;` is present
- normalizes the SQLite extension entrypoint pointer type to `*mut *const c_char`

```bash
python # start python interpreter
```

```python
from pathlib import Path

p = Path("src/kvault-rust/lib.rs")
s = p.read_text()

# Ensure import exists (insert after sqlite_vec import if found)
needle = "use sqlite_vec::sqlite3_vec_init;\n"
ins = "use sqlite_vec::sqlite3_vec_init;\nuse std::ffi::c_char;\n"
if "use std::ffi::c_char;" not in s:
    if needle in s:
        s = s.replace(needle, ins, 1)
    else:
        # fallback: prepend
        s = "use std::ffi::c_char;\n" + s

# Normalize pointer signedness (covers i8/u8 variants)
s = s.replace("*mut *const i8", "*mut *const c_char")
s = s.replace("*mut *const u8", "*mut *const c_char")

p.write_text(s)
print(f"Patched {p}")
exit()
```


### 5.3 Install KohakuVault into the same venv

Important: you want this installed into the KohakuRAG venv you created earlier.
If you are not in that environment, re-activate it from the KohakuRAG folder path.

From the KoakuVault folder (with environment active), run:
```bash
uv pip install -e .
python -c "import kohakuvault; print('kohakuvault ok')"
```

## 6) Install KohakuRAG (now that kohakuvault is present)

```bash
cd ../KohakuRAG
source .venv/bin/activate
uv pip install -e .
python -c "import kohakurag; print('kohakurag ok')"
```

## 7) Install KohakuEngine

The PyPI package name is `kohaku-engine`.
- import name: `kohakuengine`
- CLI tool: `kogine`

```bash
uv pip install kohaku-engine
python -c "import kohakuengine; print('kohakuengine ok')"
kogine --help
```