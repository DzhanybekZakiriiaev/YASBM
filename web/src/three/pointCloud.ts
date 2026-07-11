import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

/**
 * Fetch a PLY file and return a BufferGeometry with `position` and `color`
 * attributes. If the PLY has no per-vertex color, we synthesize a
 * depth-tinted palette based on Z so the cloud still reads three-dimensional.
 *
 * The pipeline emits colored ASCII PLY (property uchar red/green/blue) but
 * older captures may not, so we handle both.
 */
export async function loadPLYPointCloud(
  url: string,
  signal?: AbortSignal,
): Promise<THREE.BufferGeometry> {
  const res = await fetch(url, { signal });
  if (!res.ok) {
    throw new Error(`Failed to fetch PLY (${res.status}): ${url}`);
  }
  const buf = await res.arrayBuffer();

  const loader = new PLYLoader();
  const geometry = loader.parse(buf);

  // PLYLoader gives us positions, and — if the file had them — colors.
  const positions = geometry.getAttribute("position") as
    | THREE.BufferAttribute
    | undefined;
  if (!positions) {
    throw new Error("PLY has no position attribute");
  }

  const hasColor = !!geometry.getAttribute("color");
  if (!hasColor) {
    // Depth-tint fallback: map Z through a cool→warm ramp so we still read
    // three-dimensional volume in the absence of true color.
    const count = positions.count;
    const colors = new Float32Array(count * 3);

    // Normalize Z into [0,1].
    let zMin = Infinity;
    let zMax = -Infinity;
    for (let i = 0; i < count; i++) {
      const z = positions.getZ(i);
      if (z < zMin) zMin = z;
      if (z > zMax) zMax = z;
    }
    const zRange = Math.max(1e-6, zMax - zMin);

    // Palette anchors: near = warm amber, far = deep teal.
    const near = new THREE.Color("#ffb26b");
    const far = new THREE.Color("#3aa0c9");
    const tmp = new THREE.Color();
    for (let i = 0; i < count; i++) {
      const z = positions.getZ(i);
      const t = (z - zMin) / zRange;
      tmp.copy(near).lerp(far, t);
      colors[i * 3 + 0] = tmp.r;
      colors[i * 3 + 1] = tmp.g;
      colors[i * 3 + 2] = tmp.b;
    }
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(colors, 3),
    );
  } else {
    // PLYLoader stores color as float [0,1] already; nothing to normalize.
  }

  // If the PLY had face indices, PLYLoader sets `geometry.index`. Ensure
  // we have per-vertex normals so lighting works on the mesh render path.
  if (geometry.index) {
    geometry.computeVertexNormals();
  }

  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  return geometry;
}
