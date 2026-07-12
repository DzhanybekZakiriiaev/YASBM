import { Component, Suspense, useMemo, type ReactNode } from "react";
import { Html, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import type { PropPlacement, Vec3 } from "../lib/api";
import { useAnalysisStore } from "../state/analysis";

/**
 * 3D props layer — for every detected object with a `PropPlacement`, render
 * a stylized stand-in at its 3D bbox: a remote GLB (Poly Pizza match or a
 * Tripo "hero" upgrade) when available, otherwise a procedural proxy built
 * from three.js primitives (guaranteed offline-safe).
 *
 * Every model is normalized to the object's detected bbox: proxies are
 * built inside a unit 1×1×1 box and stretched by `prop.scale`; GLBs are
 * recentered and uniformly scaled to fit inside `prop.scale` (uniform so
 * real assets don't get squashed).
 */
export function Props({ showLabels = true }: { showLabels?: boolean }) {
  const props = useAnalysisStore((s) => s.analysisResult?.props);
  const objects = useAnalysisStore((s) => s.analysisResult?.objects);
  const showProps = useAnalysisStore((s) => s.showProps);
  const heroUrls = useAnalysisStore((s) => s.heroUrls);

  if (!showProps || !props || props.length === 0) return null;

  const verdictByObject = new Map(
    (objects ?? []).map((o) => [o.object_id, o.verdict]),
  );

  return (
    <group>
      {props.map((prop) => {
        // People are already fully represented by the per-frame dynamic
        // points + track markers; a solid capsule proxy on top duplicates
        // (and clips through) them, so persons never get a stand-in.
        const glbUrl = heroUrls[prop.object_id] ?? prop.glb_url;
        if (prop.label === "person" && !glbUrl) return null;
        return (
          <PropItem
            key={prop.object_id}
            prop={prop}
            // A generated hero asset always wins over the Poly Pizza match.
            glbUrl={glbUrl}
            showLabel={showLabels}
            verdict={verdictByObject.get(prop.object_id) ?? "static"}
          />
        );
      })}
    </group>
  );
}

/** Verdict → hologram tint. Matches the chip palette used elsewhere. */
const HOLOGRAM_COLORS: Record<string, string> = {
  consistent: "#34d399",
  borderline: "#fbbf24",
  implausible: "#f87171",
  morphing: "#e879f9",
  agent: "#38bdf8",
  static: "#9ca3af",
};

interface PropItemProps {
  prop: PropPlacement;
  glbUrl: string | null;
  showLabel: boolean;
  verdict: string;
}

function PropItem({ prop, glbUrl, showLabel, verdict }: PropItemProps) {
  // Guard degenerate bboxes so a zero-size detection never collapses the
  // model (or divides by zero during normalization).
  const scale = useMemo<Vec3>(
    () => [
      Math.max(prop.scale[0], 0.05),
      Math.max(prop.scale[1], 0.05),
      Math.max(prop.scale[2], 0.05),
    ],
    [prop.scale],
  );

  const color = HOLOGRAM_COLORS[verdict] ?? HOLOGRAM_COLORS.static;

  // Without a real model, the stand-in is a holographic bounding box —
  // a forensic ANNOTATION over the photographic mesh rather than fake
  // furniture fighting it. Solid geometry (procedural proxies included)
  // clips through the real bed/couch pixels and reads as a glitch; a
  // translucent verdict-tinted volume with bright edges reads as "the
  // auditor marked this object here".
  const hologram = <HologramBox scale={scale} color={color} />;

  return (
    <group
      position={prop.position}
      rotation={[0, (prop.yaw_deg * Math.PI) / 180, 0]}
    >
      {glbUrl ? (
        // key remounts the boundary when the URL changes (e.g. a hero
        // upgrade replacing a broken Poly Pizza GLB gets a fresh chance).
        <GlbErrorBoundary key={glbUrl} fallback={hologram}>
          <Suspense fallback={hologram}>
            <GlbModel url={glbUrl} scale={scale} />
          </Suspense>
        </GlbErrorBoundary>
      ) : (
        hologram
      )}
      {showLabel ? (
        <Html
          position={[0, scale[1] / 2 + 0.12, 0]}
          center
          zIndexRange={[20, 0]}
        >
          <div
            className="pointer-events-none select-none whitespace-nowrap rounded border bg-neutral-950/85 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-widest"
            style={{ borderColor: `${color}66`, color }}
          >
            {prop.label}
          </div>
        </Html>
      ) : null}
    </group>
  );
}

/** Verdict-tinted translucent volume + bright edges — forensic annotation. */
function HologramBox({ scale, color }: { scale: Vec3; color: string }) {
  const edges = useMemo(() => {
    const geo = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
    const e = new THREE.EdgesGeometry(geo);
    geo.dispose();
    return e;
  }, [scale]);

  return (
    <group>
      <mesh>
        <boxGeometry args={scale} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.07}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={color} transparent opacity={0.85} toneMapped={false} />
      </lineSegments>
    </group>
  );
}

/* ------------------------------------------------------------------ */
/* GLB path                                                            */
/* ------------------------------------------------------------------ */

function GlbModel({ url, scale }: { url: string; scale: Vec3 }) {
  const gltf = useGLTF(url);

  const model = useMemo(() => {
    // Clone so multiple props can share a cached GLB, and so our recenter
    // transform never mutates drei's cache entry.
    const scene = gltf.scene.clone(true);
    const box = new THREE.Box3().setFromObject(scene);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    const wrapper = new THREE.Group();
    wrapper.add(scene);
    scene.position.sub(center);
    // Uniform fit inside the detected bbox — stretching a real asset
    // non-uniformly looks worse than leaving a little air.
    const s = Math.min(
      scale[0] / Math.max(size.x, 1e-6),
      scale[1] / Math.max(size.y, 1e-6),
      scale[2] / Math.max(size.z, 1e-6),
    );
    wrapper.scale.setScalar(s);
    return wrapper;
  }, [gltf, scale]);

  return <primitive object={model} />;
}

/**
 * A bad or unreachable GLB must degrade to the procedural proxy, never
 * crash the canvas. React error boundaries require a class component.
 */
class GlbErrorBoundary extends Component<
  { fallback: ReactNode; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch(error: unknown): void {
    // eslint-disable-next-line no-console
    console.warn("[Props] GLB failed to load — using procedural proxy", error);
  }

  render(): ReactNode {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}
