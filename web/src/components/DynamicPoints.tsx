import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { useAnalysisStore } from "../state/analysis";

/**
 * Per-frame moving-pixel point cloud, synced to the video playhead.
 *
 * The pipeline exports `dynamic.json`: for every frame, the back-projected
 * 3D positions + colors of the pixels that were *excluded* from the static
 * room mesh because they move (a walking person, a thrown ball). Rendering
 * the entry that matches `currentTimeS` makes the moving object scrub
 * through 3D space in lockstep with the video — the scene finally obeys
 * the timeline instead of being a frozen frame-0 snapshot.
 */
interface FrameCloud {
  positions: Float32Array;
  colors: Float32Array; // normalized 0..1
}

interface DynamicJson {
  frames: { p: number[]; c: number[] }[];
}

export function DynamicPoints({ url }: { url: string }) {
  const currentTimeS = useAnalysisStore((s) => s.currentTimeS);
  const fps = useAnalysisStore((s) => s.analysisResult?.fps ?? 30);
  const [frames, setFrames] = useState<FrameCloud[] | null>(null);
  const geometryRef = useRef<THREE.BufferGeometry>(null);

  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) return;
        const data = (await res.json()) as DynamicJson;
        if (cancelled) return;
        const parsed: FrameCloud[] = data.frames.map((f) => {
          const positions = new Float32Array(f.p);
          const colors = new Float32Array(f.c.length);
          for (let i = 0; i < f.c.length; i++) colors[i] = f.c[i] / 255;
          return { positions, colors };
        });
        setFrames(parsed);
      } catch (err) {
        if ((err as { name?: string })?.name !== "AbortError") {
          console.warn("[DynamicPoints] failed to load", err);
        }
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
      setFrames(null);
    };
  }, [url]);

  const frameIdx = useMemo(() => {
    if (!frames || frames.length === 0) return 0;
    return Math.min(
      frames.length - 1,
      Math.max(0, Math.round(currentTimeS * (fps || 30))),
    );
  }, [frames, currentTimeS, fps]);

  // Swap buffer attributes when the playhead crosses into a new frame.
  useEffect(() => {
    const geo = geometryRef.current;
    if (!geo || !frames || frames.length === 0) return;
    const f = frames[frameIdx];
    geo.setAttribute("position", new THREE.BufferAttribute(f.positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(f.colors, 3));
    geo.computeBoundingSphere();
  }, [frames, frameIdx]);

  if (!frames || frames.length === 0) return null;

  return (
    <points frustumCulled={false}>
      <bufferGeometry ref={geometryRef} />
      <pointsMaterial
        vertexColors
        size={0.035}
        sizeAttenuation
        transparent
        opacity={0.95}
        depthWrite={false}
      />
    </points>
  );
}
