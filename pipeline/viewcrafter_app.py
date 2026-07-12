"""ViewCrafter Modal deployment — AI-generated 360° flythroughs of a scene.

Deployed as a SEPARATE Modal app from the main YASBM pipeline
(``modal_app.py``) because it needs torch 1.13.1 + pytorch3d 0.7.5 which
are incompatible with the analyze pipeline's Python 3.12 + torch 2.5
image. Keeping them apart lets the working `/analyze` endpoint stay
online while we iterate on this one.

Deploy::

    modal deploy pipeline/viewcrafter_app.py

Modal prints a URL like ``https://<user>--yasbm-viewcrafter-web.modal.run``
that the frontend will POST to when the user clicks "generate flythrough".

Model: `Drexubery/ViewCrafter_25` (Apache-2.0)
Paper: TPAMI 2025, https://drexubery.github.io/ViewCrafter/
Docs:  https://github.com/Drexubery/ViewCrafter
"""

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import modal
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse

app = modal.App("yasbm-viewcrafter")


# ---------------------------------------------------------------------------
# ViewCrafter image
# ---------------------------------------------------------------------------
# The pinned versions here mirror ViewCrafter's ``requirements.txt`` exactly
# where possible. Their canonical env is Python 3.9.16 + torch 1.13.1 + CUDA
# 11.7 — Modal's ``debian_slim`` python builds start at 3.10, so we hop up
# one version and use torch 2.0.1 + CUDA 11.8 (pytorch3d has prebuilt wheels
# for this combo). If anything breaks at load time, downgrade both.

CUDA_VER = "cu118"
TORCH_VER = "2.0.1"
TORCHVISION_VER = "0.15.2"

viewcrafter_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libglu1-mesa",
        "libxi6",
        "libxrandr2",
        "libxinerama1",
        "wget",
        # C++ compiler chain — pytorch3d builds native CUDA + C++ extensions,
        # and torch.utils.cpp_extension errors with `which clang++` returned
        # non-zero if no compiler exists on PATH.
        "build-essential",
        "g++",
        # PyAV builds from source and needs pkg-config + the full FFmpeg C
        # dev header set to link against libav*.
        "pkg-config",
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavfilter-dev",
        "libavutil-dev",
        "libswscale-dev",
        "libswresample-dev",
        "libpostproc-dev",
    )
    .pip_install(
        f"torch=={TORCH_VER}",
        f"torchvision=={TORCHVISION_VER}",
        extra_index_url=f"https://download.pytorch.org/whl/{CUDA_VER}",
    )
    # PyTorch3D — install headers + build from source. The prebuilt fbaipub
    # wheels return 403 as of mid-2026. Building from git takes ~5–10 min
    # but is reliable and caches on subsequent deploys.
    #
    # `wheel` and `setuptools` are required for `bdist_wheel` since we pass
    # `--no-build-isolation` (which skips the automatic build-env setup that
    # would otherwise provide them).
    .pip_install(
        "wheel",
        "setuptools>=60",
        "fvcore",
        "iopath",
        "ninja",
    )
    .run_commands(
        # `CC` / `CXX` override torch.utils.cpp_extension's default compiler
        # discovery which otherwise probes clang++ first.
        "CC=/usr/bin/gcc CXX=/usr/bin/g++ "
        "FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.9' "
        "pip install --no-build-isolation "
        "'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5'",
        gpu="L4",
    )
    # ViewCrafter's requirements.txt — pinned versions verbatim where they matter.
    .pip_install(
        "av==10.0.0",
        "decord==0.6.0",
        "einops==0.6.1",
        "imageio==2.27.0",
        "imageio-ffmpeg==0.4.8",
        "kornia",
        "matplotlib==3.9.2",
        "moviepy==1.0.3",
        "numpy==1.23.5",
        "open-clip-torch==2.17.1",
        "opencv-python==4.7.0.72",
        "Pillow==9.4.0",
        "pyglet==1.5.0",
        "pytorch-lightning==1.9.3",
        "PyYAML==6.0",
        "roma==1.5.0",
        "scikit-image==0.20.0",
        "scikit-learn==1.2.2",
        "scipy==1.9.1",
        "tensorboard==2.12.2",
        "timm==0.6.13",
        "tqdm==4.65.0",
        "transformers==4.28.1",
        "trimesh==4.4.3",
        "xformers==0.0.22",
        "omegaconf==2.3.0",
        "huggingface_hub>=0.20",
        "fastapi",
        "python-multipart",
        "pydantic>=2",
    )
    .run_commands(
        "git clone https://github.com/Drexubery/ViewCrafter.git /root/ViewCrafter",
    )
    .workdir("/root/ViewCrafter")
    .env({"PYTHONPATH": "/root/ViewCrafter"})
)


# ---------------------------------------------------------------------------
# Persistent volumes for weights + artifacts
# ---------------------------------------------------------------------------

vc_weights_vol = modal.Volume.from_name("yasbm-viewcrafter-weights", create_if_missing=True)
vc_artifacts_vol = modal.Volume.from_name("yasbm-viewcrafter-artifacts", create_if_missing=True)

_VOLUMES = {
    "/root/ViewCrafter/checkpoints": vc_weights_vol,
    "/artifacts": vc_artifacts_vol,
}


# ---------------------------------------------------------------------------
# Model class — warm-loaded on GPU
# ---------------------------------------------------------------------------

VC_MODEL_URL = "Drexubery/ViewCrafter_25"
VC_CKPT_NAME = "model.ckpt"
DUST3R_URL = (
    "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/"
    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
)
DUST3R_CKPT_NAME = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"


@app.cls(
    image=viewcrafter_image,
    gpu="L4",
    volumes=_VOLUMES,
    timeout=900,
    scaledown_window=180,
)
class ViewCrafterPipeline:
    @modal.enter()
    def load(self) -> None:
        """Download ViewCrafter + DUSt3R weights on first invocation.

        Only runs the download when the target file isn't already present —
        subsequent container starts hit the Modal Volume and skip network I/O.
        """

        import urllib.request

        from huggingface_hub import hf_hub_download

        ckpt_dir = Path("/root/ViewCrafter/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        vc_ckpt = ckpt_dir / VC_CKPT_NAME
        if not vc_ckpt.exists():
            print(f"[viewcrafter] downloading {VC_CKPT_NAME} from {VC_MODEL_URL} ...")
            path = hf_hub_download(
                repo_id=VC_MODEL_URL,
                filename=VC_CKPT_NAME,
                local_dir=str(ckpt_dir),
            )
            # hf_hub_download may create a nested path — normalise.
            if Path(path) != vc_ckpt:
                shutil.copy(path, vc_ckpt)

        dust3r_ckpt = ckpt_dir / DUST3R_CKPT_NAME
        if not dust3r_ckpt.exists():
            print(f"[viewcrafter] downloading {DUST3R_CKPT_NAME} ...")
            urllib.request.urlretrieve(DUST3R_URL, str(dust3r_ckpt))

        vc_weights_vol.commit()
        print("[viewcrafter] weights ready")

    @modal.method()
    def generate_flythrough(
        self,
        image_bytes: bytes,
        request_id: str,
        video_length: int = 25,
        ddim_steps: int = 50,
    ) -> dict:
        """Run ViewCrafter's `single_view_txt` mode on one image.

        Writes the input to a temp dir, shells out to ``inference.py``, then
        copies the resulting MP4 into the shared artifacts volume.

        Uses the 512×320 config (fits comfortably on an L4 at 24 GB VRAM).
        Bumping to 1024×576 requires an A10G+.
        """

        tmp_root = Path(tempfile.mkdtemp(prefix="vc-"))
        input_dir = tmp_root / "in"
        output_dir = tmp_root / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        input_img = input_dir / "input.png"
        input_img.write_bytes(image_bytes)

        cmd = [
            "python", "/root/ViewCrafter/inference.py",
            "--image_dir", str(input_img),
            "--out_dir", str(output_dir),
            "--traj_txt", "/root/ViewCrafter/test/trajs/loop2.txt",
            "--ckpt_path", f"/root/ViewCrafter/checkpoints/{VC_CKPT_NAME}",
            "--model_path", f"/root/ViewCrafter/checkpoints/{DUST3R_CKPT_NAME}",
            "--config", "/root/ViewCrafter/configs/inference_pvd_512.yaml",
            "--mode", "single_view_txt",
            "--elevation", "5",
            "--d_theta", "-30",
            "--d_phi", "45",
            "--d_r", "-.2",
            "--d_x", "50",
            "--d_y", "25",
            "--center_scale", "1.",
            "--ddim_steps", str(ddim_steps),
            "--video_length", str(video_length),
            "--seed", "123",
            "--height", "320",
            "--width", "512",
            "--device", "cuda:0",
        ]

        print(f"[viewcrafter] running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd="/root/ViewCrafter",
            capture_output=True,
            text=True,
            timeout=800,
        )

        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-3000:]
            raise RuntimeError(f"ViewCrafter inference failed:\n{tail}")

        # ViewCrafter drops the MP4 somewhere under out_dir with a variable
        # name. Grab any MP4 we can find.
        mp4s = sorted(output_dir.rglob("*.mp4"))
        if not mp4s:
            raise RuntimeError(
                f"ViewCrafter produced no MP4 files. Output dir contents: "
                f"{list(output_dir.rglob('*'))}"
            )
        best = mp4s[-1]

        # Persist under /artifacts/<request_id>/flythrough.mp4 so the web
        # container can serve it via /flythroughs/<request_id>.
        dest_dir = Path("/artifacts") / request_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "flythrough.mp4"
        shutil.copy(best, dest)
        vc_artifacts_vol.commit()

        # Best-effort cleanup — Modal will nuke /tmp on scaledown anyway.
        try:
            shutil.rmtree(tmp_root)
        except OSError:
            pass

        return {
            "status": "done",
            "flythrough_url": f"/flythroughs/{request_id}/flythrough.mp4",
            "request_id": request_id,
            "video_length": video_length,
            "ddim_steps": ddim_steps,
        }


# ---------------------------------------------------------------------------
# Web ASGI entrypoint
# ---------------------------------------------------------------------------

@app.function(
    image=viewcrafter_image,
    volumes=_VOLUMES,
    timeout=900,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="YASBM ViewCrafter", version="0.0.1")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "yasbm-viewcrafter"}

    @api.get("/flythroughs/{req_id}/{filename}")
    async def artifact(req_id: str, filename: str) -> FileResponse:
        vc_artifacts_vol.reload()
        path = Path("/artifacts") / req_id / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="flythrough not found")
        return FileResponse(str(path), media_type="video/mp4")

    @api.post("/flythrough")
    async def flythrough(request: Request):
        try:
            form = await request.form()
            image = form.get("image")
            if image is None or not hasattr(image, "read"):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "missing 'image' field in multipart body"},
                )
            body = await image.read()
            if not body:
                return JSONResponse(
                    status_code=400, content={"detail": "empty image"}
                )

            request_id = uuid.uuid4().hex
            result = await ViewCrafterPipeline().generate_flythrough.remote.aio(
                body, request_id
            )

            base_url = str(request.base_url).rstrip("/")
            if result.get("flythrough_url", "").startswith("/flythroughs/"):
                result["flythrough_url"] = base_url + result["flythrough_url"]
            return result
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            print(f"[/flythrough] error: {exc}\n{tb}")
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"{type(exc).__name__}: {exc}",
                    "traceback": tb.splitlines()[-15:],
                },
            )

    return api
