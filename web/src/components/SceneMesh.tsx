import { useEffect, useState } from "react";
import * as THREE from "three";
import { loadPLYPointCloud } from "../three/pointCloud";

/**
 * Renders the reconstructed scene as a solid mesh when the PLY has face
 * indices, or falls back to nothing when it's a bare point cloud (the
 * <PointCloud> component takes over in that case).
 *
 * Vertex colors carry the RGB samples from the source video; a standard PBR
 * material picks up the R3F 3-point rig so the surface catches highlights
 * and reads as a real 3D object under orbit — the whole point of moving
 * from `<points>` to `<mesh>` on the reconstruction.
 */
interface SceneMeshProps {
  url: string;
}

export function SceneMesh({ url }: SceneMeshProps) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [hasFaces, setHasFaces] = useState(false);

  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    let cancelled = false;
    loadPLYPointCloud(url, controller.signal)
      .then((geo) => {
        if (cancelled) {
          geo.dispose();
          return;
        }
        setHasFaces(!!geo.index && geo.index.count > 0);
        setGeometry(geo);
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name !== "AbortError") {
          console.warn("[SceneMesh] failed to load PLY", err);
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
      setGeometry((prev) => {
        prev?.dispose();
        return null;
      });
    };
  }, [url]);

  if (!geometry || !hasFaces) return null;

  return (
    <mesh
      geometry={geometry}
      castShadow={false}
      receiveShadow={false}
      frustumCulled={false}
    >
      <meshStandardMaterial
        vertexColors
        roughness={0.85}
        metalness={0.0}
        // Two-sided so the "back" of the depth mesh is visible when the
        // user orbits behind it — no invisible-from-behind glitch.
        side={THREE.DoubleSide}
        toneMapped
      />
    </mesh>
  );
}
