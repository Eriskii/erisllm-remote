"""Jacobian lens files, read without torch.

The Jacobian lens (Gurnee et al. 2026, "Verbalizable Representations Form a Global
Workspace in Language Models"; reference code github.com/anthropics/jacobian-lens)
holds one matrix J_l per layer: the average Jacobian of the last layer's residual
with respect to layer l's, over a text corpus. Layer l's residual h reads out as

    lens_l(h) = log_softmax(unembed(norm(J_l h)))

with the model's own final norm and unembedding. The reading itself is done by
`erisllm.engine`, which has the model; this module only finds and loads the files.

The reference code saves lenses with torch.save. `load` reads those with a
restricted unpickler that can only rebuild tensors and dictionaries, and keeps a
float16 .safetensors copy under data/jlens/ so later starts skip the conversion.
"""
import collections
import pickle
import sys
import zipfile
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "jlens"

# Pre-fitted lenses for the base models used here: (Hugging Face repo, file).
KNOWN = {
    # Fitted by Neuronpedia on google/gemma-3-12b-it (bfloat16), 844 wikitext prompts.
    "mlx-community/gemma-3-12b-it-qat-4bit": (
        "neuronpedia/jacobian-lens",
        "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt",
    ),
    # Fitted on google/gemma-4-12B-it-qat-q4_0-unquantized, the weights the OptiQ build was made from.
    "mlx-community/gemma-4-12B-it-qat-OptiQ-4bit": (
        "gghfez/gemma-4-it-qat-jacobian-lens",
        "gemma-4-12B-it-qat_jlens.pt",
    ),
}


class Lens:
    """Per-layer J matrices (float32, [d_model, d_model]) and where they came from."""

    def __init__(self, jacobians: dict[int, mx.array], source: str, n_prompts: int | None = None):
        self.J = jacobians
        self.layers = sorted(jacobians)
        self.source = source
        self.n_prompts = n_prompts
        self.d_model = next(iter(jacobians.values())).shape[0]


def _bf16(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)


_STORAGES = {
    "HalfStorage": np.float16,
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "BFloat16Storage": "bfloat16",
}


def _rebuild_tensor(storage, offset, size, stride, *_):
    out = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * storage.itemsize for s in stride)
    )
    return np.ascontiguousarray(out)


class _TensorUnpickler(pickle.Unpickler):
    """Rebuilds the tensors of a torch.save zip and refuses every other global."""

    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" and name in _STORAGES:
            return _STORAGES[name]
        raise pickle.UnpicklingError(f"not a plain tensor file: refusing {module}.{name}")

    def persistent_load(self, pid):
        kind, dtype, key, _location, _numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"unknown persistent id {kind!r}")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        return _bf16(raw) if dtype == "bfloat16" else np.frombuffer(raw, dtype)


def read_pt(path: Path) -> dict:
    """The object a torch.save zip holds, with tensors as numpy arrays."""
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        with archive.open(pkl) as f:
            return _TensorUnpickler(f, archive, prefix).load()


def _from_pt(path: Path) -> tuple[dict[int, np.ndarray], int | None]:
    obj = read_pt(path)
    if "J" not in obj:
        raise ValueError(f"{path} is not a Jacobian lens (keys: {sorted(obj)})")
    return {int(layer): J for layer, J in obj["J"].items()}, obj.get("n_prompts")


def load(spec: str, base_model: str) -> Lens | None:
    """A lens from `spec`: a .pt or .safetensors path, "auto" (the known lens for
    `base_model`, downloaded the first time), or "none"."""
    if spec == "none":
        return None
    if spec == "auto":
        if base_model not in KNOWN:
            print(f"jlens: no known lens for {base_model}; pass --jlens PATH", file=sys.stderr, flush=True)
            return None
        repo, filename = KNOWN[base_model]
        converted = CACHE / f"{Path(filename).stem}.safetensors"
        if not converted.exists():
            from huggingface_hub import hf_hub_download

            print(f"jlens: fetching {repo}/{filename} (about 1.4 GB, once)", file=sys.stderr, flush=True)
            spec = hf_hub_download(repo, filename)
        else:
            spec = str(converted)
        source = f"{repo}/{filename}"
    else:
        source = spec
    path = Path(spec)
    if path.suffix == ".safetensors":
        arrays, meta = mx.load(str(path), return_metadata=True)
        J = {int(k.split(".", 1)[1]): v.astype(mx.float32) for k, v in arrays.items()}
        return Lens(J, meta.get("source", source), int(meta["n_prompts"]) if meta.get("n_prompts") else None)
    jac, n_prompts = _from_pt(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{path.stem}.safetensors"
    meta = {"source": source, **({"n_prompts": str(n_prompts)} if n_prompts else {})}
    mx.save_safetensors(str(out), {f"J.{layer}": mx.array(J.astype(np.float16)) for layer, J in jac.items()}, metadata=meta)
    print(f"jlens: converted to {out}", file=sys.stderr, flush=True)
    return Lens({layer: mx.array(J.astype(np.float32)) for layer, J in jac.items()}, source, n_prompts)
