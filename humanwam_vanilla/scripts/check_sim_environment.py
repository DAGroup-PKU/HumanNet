#!/usr/bin/env python3
"""Check RoboTwin dependencies and CPU physics, with optional GPU rendering."""
import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render", action="store_true", help="Also render on logical cuda:0; select an idle GPU with CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if not args.render:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # SAPIEN 3.0.0b1 only auto-detects /usr/share, while some containers use /etc.
    icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    if icd.is_file() and not os.environ.get("VK_ICD_FILENAMES"):
        os.environ["VK_ICD_FILENAMES"] = str(icd)
    report = {"python": sys.version, "prefix": sys.prefix, "checks": {},
              "gpu_tested": False, "renderer_tested": False, "policy_rollout_tested": False}

    def check(name, fn):
        try:
            report["checks"][name] = {"passed": True, "details": fn()}
        except Exception as exc:
            report["checks"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    for module in ("sapien", "mplib", "open3d", "cv2", "transforms3d", "toppra",
                   "gymnasium", "trimesh", "warp", "torch", "curobo.curobolib.geom_cu",
                   "curobo.curobolib.kinematics_fused_cu"):
        check("import:" + module, lambda name=module: str(importlib.import_module(name).__file__))

    def cpu_physics():
        import sapien
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(1 / 100)
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
        builder.initial_pose = sapien.Pose([0, 0, 1])
        actor = builder.build(name="environment-check")
        before = float(actor.pose.p[2])
        for _ in range(10):
            scene.step()
        after = float(actor.pose.p[2])
        assert 0 < after < before, (before, after)
        return {"steps": 10, "initial_z": before, "final_z": after, "render_system": False}

    check("sapien_cpu_physics", cpu_physics)

    def vulkan_loader():
        ctypes.CDLL("libvulkan.so.1")
        return "Vulkan loader is present; GPU device/rendering not tested"

    check("vulkan_loader", vulkan_loader)

    def headless_render():
        import numpy as np
        import sapien
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("optix")
        device = sapien.Device("cuda:0")
        assert device.can_render(), str(device)
        renderer = sapien.render.SapienRenderer(device)
        engine = sapien.Engine()
        engine.set_renderer(renderer)
        scene = engine.create_scene(sapien.SceneConfig())
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([-1, -1, -1], [1, 1, 1])
        material = sapien.render.RenderMaterial()
        material.base_color = [1, 0.2, 0.1, 1]
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.3, 0.3, 0.3], material=material)
        builder.initial_pose = sapien.Pose([2, 0, 1])
        builder.build_static(name="render-check")
        camera = scene.add_camera("check", 64, 64, 1.0, 0.1, 10.0)
        camera.set_pose(sapien.Pose([0, 0, 1], [1, 0, 0, 0]))
        scene.step()
        scene.update_render()
        camera.take_picture()
        color = np.asarray(camera.get_picture("Color"))
        assert color.shape == (64, 64, 4) and np.isfinite(color).all()
        assert float(color[:, :, :3].std()) > 0.01
        return {"device": str(device), "pci": device.pci_string, "cuda_id": device.cuda_id,
                "shader": "rt", "denoiser": "optix", "samples_per_pixel": 32,
                "path_depth": 8, "image_shape": list(color.shape), "rgb_std": float(color[:, :, :3].std())}

    if args.render:
        check("optix_headless_render", headless_render)
        report["renderer_tested"] = report["checks"]["optix_headless_render"]["passed"]
        report["gpu_tested"] = report["renderer_tested"]
    report["versions"] = {}
    for package in ("sapien", "mplib", "open3d", "torch", "warp-lang", "numpy", "scipy", "nvidia-curobo"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = None
    report["passed"] = all(item["passed"] for item in report["checks"].values())
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
