# ---
# pytest: false
# ---

"""
PTQ (AWQ / GPTQ) for OpenAI's gpt-oss on Modal, then serve with vLLM.

Usage:
  modal run gpt_oss_ptq_modal.py --method=awq
  modal run gpt_oss_ptq_modal.py --method=gptq

Env knobs:
  MODEL_NAME=openai/gpt-oss-20b (default)
  MAX_CAL_SAMPLES=64   # safer default, dial up later
  SEQ_LEN=256          # safer default, dial up later
  N_GPU=2              # PTQ and serving
  PACK_GROUP_SIZE=64   # ensure packer matches calibration group size
  FAST_BOOT=0|1        # vLLM
"""

from __future__ import annotations
import os
from pathlib import Path
import inspect
import modal

# ---------------------- App / Config ----------------------
APP_NAME = "gpt-oss-ptq"
app = modal.App(APP_NAME)

MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")
PTQ_METHOD = os.environ.get("PTQ_METHOD", "awq").lower()  # "awq" or "gptq"

# Conservative defaults; you can override via env on run
MAX_CAL_SAMPLES = int(os.environ.get("MAX_CAL_SAMPLES", "1028"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "2048"))
PACK_GROUP_SIZE = int(os.environ.get("PACK_GROUP_SIZE", "64"))

OUTPUT_SUBDIR = os.environ.get(
    "OUTPUT_SUBDIR", f"{MODEL_NAME.split('/')[-1]}-{PTQ_METHOD}-w4a16"
)

# Server / hardware
N_GPU = int(os.environ.get("N_GPU", "1"))
FAST_BOOT = os.environ.get("FAST_BOOT", "0") == "1"
VLLM_PORT = 8000
MINUTES = 60  # seconds

# Volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("gpt-oss-models", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ---------------------- Container Image ----------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            # reduce fragmentation / OOM risk
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:64",
        }
    )
    .pip_install(
        "transformers>=4.44.0",
        "huggingface_hub[hf_transfer]>=0.23",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "llmcompressor>=0.6.0.1",
    )
    # Torch nightly (cu128)
    .uv_pip_install(
        "torch",
        pre=True,
        extra_options="--extra-index-url https://download.pytorch.org/whl/nightly/cu128",
    )
    # vLLM GPT-OSS build
    .uv_pip_install(
        "vllm==0.10.1+gptoss",
        pre=True,
        extra_options=(
            "--extra-index-url https://wheels.vllm.ai/gpt-oss/ "
            "--extra-index-url https://download.pytorch.org/whl/nightly/cu128 "
            "--index-strategy unsafe-best-match"
        ),
    )
)

# ---------------------- Helpers ----------------------
def _bypass_moe_mlp_for_calibration(model):
    """
    Monkey-patch GPT-OSS MoE MLP forward to identity during calibration only.
    Returns a list of (module, original_forward) so we can restore afterwards.
    """
    import types
    patched = []
    core = getattr(model, "model", model)
    layers = getattr(core, "layers", None)
    if layers is None:
        return patched

    def _identity_forward(self, hidden_states, *args, **kwargs):
        # GPT-OSS MLP normally returns (hidden_states, router_logits)
        return hidden_states, None

    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "forward"):
            patched.append((mlp, mlp.forward))
            mlp.forward = types.MethodType(_identity_forward, mlp)
    return patched


def _restore_patches(patched):
    for mod, orig in patched:
        mod.forward = orig


def _gs_kw(cls, g=64):
    # llmcompressor versions sometimes use 'group_size' or 'groupsize'
    params = inspect.signature(cls).parameters
    if "group_size" in params:
        return {"group_size": g}
    if "groupsize" in params:
        return {"groupsize": g}
    return {}


def force_packer_group_size(g=64):
    """
    compressed_tensors defaults to group_size=128 when packing.
    Force it to 64 so column shapes like 2880 are divisible.
    """
    try:
        import compressed_tensors.compressors.quantized_compressors.pack_quantized as pq  # noqa: E402

        orig_init = pq.PackQuantizedWeightsCompressor.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["group_size"] = g
            return orig_init(self, *args, **kwargs)

        if getattr(pq.PackQuantizedWeightsCompressor.__init__, "_patched_gs", None) != g:
            pq.PackQuantizedWeightsCompressor.__init__ = patched_init  # type: ignore
            pq.PackQuantizedWeightsCompressor.__init__._patched_gs = g  # type: ignore
            print(f"[PTQ] Forced packer group_size={g}")
    except Exception as e:
        print(f"[PTQ] WARNING: could not force packer group_size -> {e}")

# ---------------------- PTQ Recipe Builder ----------------------
def build_recipe(method: str):
    """
    Attention-only PTQ for GPT-OSS.
    - Ignore MoE layers and lm_head.
    - Use group_size=64 to match 2880 etc.
    """
    ignore_regex = [
        r"re:.*mlp\.experts.*",
        r"re:.*mlp\.router.*",
        r"re:.*experts.*",
        r"re:.*router.*",
        r"re:.*lm_head.*",
    ]

    if method == "awq":
        from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

        mappings = [
            AWQMapping(
                smooth_layer=r"re:.*input_layernorm$",
                balance_layers=[r"re:.*q_proj$", r"re:.*k_proj$", r"re:.*v_proj$"],
            ),
            AWQMapping(
                smooth_layer=r"re:.*v_proj$",
                balance_layers=[r"re:.*o_proj$"],
            ),
        ]

        return [
            AWQModifier(
                scheme="W4A16",
                targets=["Linear"],         # attention linears only (MLP is ignored above)
                mappings=mappings,
                ignore=ignore_regex,
                attn_implementation="eager",
                **_gs_kw(AWQModifier, 64),  # <-- critical: make columns divisible
            )
        ]

    elif method == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier

        return [
            GPTQModifier(
                scheme="W4A16",
                targets="Linear",
                ignore=ignore_regex,
                **_gs_kw(GPTQModifier, 64),  # <-- also 64 for GPTQ
            )
        ]

    else:
        raise ValueError(f"Unknown PTQ method: {method}")

# ---------------------- PTQ Task ----------------------
@app.function(
    image=image,
    gpu=f"H200:{N_GPU}",
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/models": models_vol,
    },
)
def ptq_pack(
    model_name: str = MODEL_NAME,
    method: str = PTQ_METHOD,
    output_subdir: str = OUTPUT_SUBDIR,
    max_calibration_samples: int = MAX_CAL_SAMPLES,
    seq_len: int = SEQ_LEN,
    pack_group_size: int = PACK_GROUP_SIZE,
) -> str:
    """
    Run PTQ (AWQ or GPTQ) on GPT-OSS attention layers, ignoring MoE.
    Writes to /models/<output_subdir>. Returns the path.
    """
    # Force eager/no-compile to avoid meta↔cuda tracing issues
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTORCH_JIT"] = "0"

    print(f"[PTQ] Starting {method.upper()} on {model_name}")
    out_dir = Path("/models") / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PTQ] Output -> {out_dir}")
    print(f"[PTQ] Requested: samples={max_calibration_samples}, seq_len={seq_len}")

    recipe = build_recipe(method)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    n_gpus = torch.cuda.device_count()
    print(f"[PTQ] Visible GPUs: {n_gpus}")

    # Load config with eager attention for calibration
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    if hasattr(cfg, "use_flash_attention"):
        cfg.use_flash_attention = False

    # Device map: shard across GPUs if available; cap memory per GPU
    if n_gpus > 1:
        max_memory = {}
        for i in range(n_gpus):
            torch.cuda.set_device(i)
            _, total = torch.cuda.mem_get_info()
            total_gb = int(total / (1024**3))
            max_memory[i] = f"{max(1, total_gb - 10)}GiB"  # leave headroom
        device_map = "auto"
        extra_model_kwargs = dict(max_memory=max_memory)
        print(f"[PTQ] Using device_map=auto with max_memory={max_memory}")
    else:
        torch.cuda.set_device(0)
        device_map = {"": 0}
        extra_model_kwargs = {}
        print("[PTQ] Using single-GPU placement on cuda:0")

    # Load model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        **extra_model_kwargs,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )

    # VRAM-aware downshift (after load)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
    except Exception:
        free_gb = None

    if free_gb is not None:
        # Tighten more when we have very little headroom
        if free_gb < 40:
            max_calibration_samples = min(max_calibration_samples, 48)
            seq_len = min(seq_len, 192)
        elif free_gb < 60:
            max_calibration_samples = min(max_calibration_samples, 64)
            seq_len = min(seq_len, 256)
        elif free_gb < 80:
            max_calibration_samples = min(max_calibration_samples, 96)
            seq_len = min(seq_len, 384)
        print(f"[PTQ] Free VRAM ~{free_gb:.1f} GiB -> using samples={max_calibration_samples}, seq_len={seq_len}")

    # Import oneshot (version-compatible)
    try:
        from llmcompressor import oneshot
    except Exception:
        from llmcompressor.transformers import oneshot

    # *** Force packer to use the same group size we calibrated with (default 64) ***
    force_packer_group_size(pack_group_size)

    # MoE bypass during calibration + low-mem execution context
    patched = _bypass_moe_mlp_for_calibration(model)
    torch.cuda.empty_cache()
    try:
        with torch.inference_mode():
            oneshot(
                model=model,
                tokenizer=tokenizer,
                dataset="open_platypus",
                recipe=recipe,
                output_dir=str(out_dir),
                max_seq_length=seq_len,
                num_calibration_samples=max_calibration_samples,
            )
    finally:
        _restore_patches(patched)
        torch.cuda.empty_cache()

    (out_dir / "PTQ_DONE.txt").write_text(
        f"model={model_name}\nmethod={method}\nsamples={max_calibration_samples}\nseq_len={seq_len}\npack_group_size={pack_group_size}\n"
    )
    print("[PTQ] Done.")
    return str(out_dir)


# ---------------------- Serve Compressed Model ----------------------
@app.function(
    image=image,
    gpu=f"H200:{N_GPU}",
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/models": models_vol,
    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve_quantized():
    """
    Boot a vLLM server for the compressed checkpoint at /models/<OUTPUT_SUBDIR>.
    Uses:
      - PTQ_METHOD ('awq' or 'gptq') for --quantization
      - --kv-cache-dtype fp8 to reduce runtime memory on H100/H200
    """
    import subprocess

    local_model_dir = f"/models/{OUTPUT_SUBDIR}"
    method = PTQ_METHOD
    kv_cache_dtype = "fp8"

    cmd = [
        "vllm",
        "serve",
        local_model_dir,
        "--served-model-name",
        Path(local_model_dir).name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--quantization",
        method,
    ]
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    print("[vLLM] Launch:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


# ---------------------- Local Entrypoint ----------------------
@app.local_entrypoint()
def run(method: str = PTQ_METHOD, serve: bool = True):
    """
    Examples:
      MAX_CAL_SAMPLES=64 SEQ_LEN=256 N_GPU=2 modal run gpt_oss_ptq_modal.py --method=awq
      N_GPU=2 modal run gpt_oss_ptq_modal.py --method=gptq
    """
    method = method.lower()
    os.environ["PTQ_METHOD"] = method  # keep globals consistent in server
    global OUTPUT_SUBDIR
    OUTPUT_SUBDIR = f"{MODEL_NAME.split('/')[-1]}-{method}-w4a16"

    print(f"Quantizing {MODEL_NAME} with {method}...")
    out_dir = ptq_pack.remote(method=method, output_subdir=OUTPUT_SUBDIR)
    print(f"Compressed model at: {out_dir}")

    if serve:
        print("Starting vLLM web server...")
        url = serve_quantized.get_web_url()
        print(f"vLLM running at: {url}")
        print("Use the OpenAI-compatible /v1/chat/completions endpoint.")
