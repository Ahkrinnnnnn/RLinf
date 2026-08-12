#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Convert ``scene_assets/001`` STLs into meter-scale Z-up USDA meshes.

Usage (from anywhere):
  python rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/convert_scene_assets.py \\
      --src /path/to/scene_assets/001
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

PARTS = {
    "圆柱.STL": "peg",
    "空心圆柱.STL": "tube",
    "大方形底座.STL": "base",
    "中间件.STL": "middle",
    "顶盖.STL": "cover",
}
PRIM_NAMES = {
    "peg": "Peg",
    "tube": "Tube",
    "base": "Base",
    "middle": "Middle",
    "cover": "Cover",
}


def load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    ascii_stl = data[:5].lower().startswith(b"solid") and b"facet" in data[:400]
    if not ascii_stl and len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        verts = np.empty((n, 3, 3), dtype=np.float64)
        off = 84
        for i in range(n):
            verts[i, 0] = struct.unpack_from("<fff", data, off + 12)
            verts[i, 1] = struct.unpack_from("<fff", data, off + 24)
            verts[i, 2] = struct.unpack_from("<fff", data, off + 36)
            off += 50
        return verts
    tris = []
    cur = []
    for line in data.decode("latin1", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            cur.append([float(x) for x in line.split()[1:4]])
            if len(cur) == 3:
                tris.append(cur)
                cur = []
    return np.asarray(tris, dtype=np.float64)


def orient_z_up(verts: np.ndarray, name: str) -> np.ndarray:
    size = verts.reshape(-1, 3).max(0) - verts.reshape(-1, 3).min(0)
    out = verts.copy()
    # Square base is Y-thin in CAD; rotate so height is +Z.
    if name == "base" and size[1] < min(size[0], size[2]) * 0.6:
        out[..., :] = out[..., [0, 2, 1]]
    return out


def center_geom(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = verts.reshape(-1, 3)
    mn, mx = pts.min(0), pts.max(0)
    return verts - 0.5 * (mn + mx), mx - mn


def write_usda(
    path: Path, prim_name: str, verts_m: np.ndarray, *, kinematic: bool, mass: float
) -> None:
    n = len(verts_m)
    points = verts_m.reshape(-1, 3)
    point_lines = [f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in points]
    # Tube/base keep open cavities via triangle mesh (kinematic only).
    # Dynamic bodies (peg) need convexHull + PhysicsMeshCollisionAPI, otherwise
    # PhysX ignores physics:approximation and falls back from triangle mesh.
    approx = "none" if prim_name in {"Tube", "Base"} else "convexHull"
    kin = 1 if kinematic else 0
    mesh_apis = '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]'
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "{prim_name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{prim_name}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
)
{{
    bool physics:kinematicEnabled = {kin}
    float physics:mass = {mass}
    def Mesh "geometry" (
        prepend apiSchemas = {mesh_apis}
    )
    {{
        uniform token physics:approximation = "{approx}"
        point3f[] points = [{", ".join(point_lines)}]
        int[] faceVertexCounts = [{", ".join(["3"] * n)}]
        int[] faceVertexIndices = [{", ".join(str(i) for i in range(n * 3))}]
        uniform token subdivisionScheme = "none"
    }}
}}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    default_src = Path(__file__).resolve().parents[6] / ".." / "scene_assets" / "001"
    parser.add_argument(
        "--src",
        type=Path,
        default=default_src.resolve(),
        help="Directory containing the Chinese-named STL parts",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "scene_001",
        help="Output directory for USDA meshes",
    )
    args = parser.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    masses = {"peg": 0.08, "tube": 0.25, "base": 0.5, "middle": 0.2, "cover": 0.15}
    meta = {}
    for src_name, key in PARTS.items():
        src = args.src / src_name
        if not src.is_file():
            raise FileNotFoundError(src)
        verts, size_mm = center_geom(orient_z_up(load_stl(src), key))
        size_m = size_mm * 0.001
        write_usda(
            args.dst / f"{key}.usda",
            PRIM_NAMES[key],
            verts * 0.001,
            kinematic=key != "peg",
            mass=masses[key],
        )
        meta[key] = {"height": float(size_m[2]), "size": size_m.tolist(), "source": src_name}
        print(f"{key}: size_m={size_m.tolist()}")

    (args.dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
