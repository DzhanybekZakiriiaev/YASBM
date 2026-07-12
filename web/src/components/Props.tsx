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
export function Props({ showLabels = false }: { showLabels?: boolean }) {
  const props = useAnalysisStore((s) => s.analysisResult?.props);
  const showProps = useAnalysisStore((s) => s.showProps);
  const heroUrls = useAnalysisStore((s) => s.heroUrls);

  if (!showProps || !props || props.length === 0) return null;

  return (
    <group>
      {props.map((prop) => (
        <PropItem
          key={prop.object_id}
          prop={prop}
          // A generated hero asset always wins over the Poly Pizza match.
          glbUrl={heroUrls[prop.object_id] ?? prop.glb_url}
          showLabel={showLabels}
        />
      ))}
    </group>
  );
}

interface PropItemProps {
  prop: PropPlacement;
  glbUrl: string | null;
  showLabel: boolean;
}

function PropItem({ prop, glbUrl, showLabel }: PropItemProps) {
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

  const proxy = <ProceduralProxy label={prop.label} scale={scale} />;

  return (
    <group
      position={prop.position}
      rotation={[0, (prop.yaw_deg * Math.PI) / 180, 0]}
    >
      {glbUrl ? (
        // key remounts the boundary when the URL changes (e.g. a hero
        // upgrade replacing a broken Poly Pizza GLB gets a fresh chance).
        <GlbErrorBoundary key={glbUrl} fallback={proxy}>
          <Suspense fallback={proxy}>
            <GlbModel url={glbUrl} scale={scale} />
          </Suspense>
        </GlbErrorBoundary>
      ) : (
        proxy
      )}
      {showLabel ? (
        <Html
          position={[0, scale[1] / 2 + 0.12, 0]}
          center
          zIndexRange={[20, 0]}
        >
          <div className="pointer-events-none select-none whitespace-nowrap rounded border border-neutral-700 bg-neutral-950/85 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-widest text-neutral-300">
            {prop.label}
          </div>
        </Html>
      ) : null}
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

/* ------------------------------------------------------------------ */
/* Procedural proxies                                                  */
/* ------------------------------------------------------------------ */

function ProceduralProxy({ label, scale }: { label: string; scale: Vec3 }) {
  const group = useMemo(() => {
    const key = label.trim().toLowerCase();
    const builder = BUILDERS[key] ?? buildGhost;
    return normalizeToUnitBox(builder());
  }, [label]);

  return <primitive object={group} scale={scale} />;
}

/**
 * Wrap a freeform group so its bounding box becomes exactly 1×1×1 centered
 * at the origin. Scaling the result by the detected bbox dims then fills
 * the detection exactly (proxies are stylized, so non-uniform stretch is
 * fine — it keeps a wide bed wide and a tall person tall).
 */
function normalizeToUnitBox(group: THREE.Group): THREE.Group {
  const box = new THREE.Box3().setFromObject(group);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const wrapper = new THREE.Group();
  wrapper.add(group);
  group.position.set(-center.x, -center.y, -center.z);
  wrapper.scale.set(
    1 / Math.max(size.x, 1e-6),
    1 / Math.max(size.y, 1e-6),
    1 / Math.max(size.z, 1e-6),
  );
  return wrapper;
}

/* Muted stylized palette — consistent look across all proxy classes. */
const PALETTE = {
  body: 0x8a8f98, // neutral grey-blue
  bodyLight: 0xa7adb6,
  fabric: 0x6e7b8a, // desaturated slate
  linen: 0xb0b6bd,
  wood: 0x7a6a58, // muted walnut
  dark: 0x22252a, // near-black plastic / screens
  metal: 0x565b63,
} as const;

function mat(color: number): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.8,
    metalness: 0.05,
  });
}

function box(
  w: number,
  h: number,
  d: number,
  material: THREE.MeshStandardMaterial,
  x = 0,
  y = 0,
  z = 0,
): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  mesh.position.set(x, y, z);
  return mesh;
}

function cyl(
  rTop: number,
  rBot: number,
  h: number,
  material: THREE.MeshStandardMaterial,
  x = 0,
  y = 0,
  z = 0,
): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(rTop, rBot, h, 20),
    material,
  );
  mesh.position.set(x, y, z);
  return mesh;
}

function buildPerson(): THREE.Group {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.18, 0.75, 6, 14),
    mat(PALETTE.fabric),
  );
  body.position.y = 0.74; // capsule spans ~0.18..1.29
  g.add(body);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.15, 18, 14),
    mat(PALETTE.bodyLight),
  );
  head.position.y = 1.47;
  g.add(head);
  return g;
}

function buildBed(): THREE.Group {
  const g = new THREE.Group();
  g.add(box(2.0, 0.32, 1.5, mat(PALETTE.wood), 0, 0.16, 0)); // base
  g.add(box(1.92, 0.2, 1.42, mat(PALETTE.fabric), 0, 0.42, 0)); // mattress
  const pillow = mat(PALETTE.linen);
  g.add(box(0.34, 0.12, 0.5, pillow, -0.72, 0.58, -0.36));
  g.add(box(0.34, 0.12, 0.5, pillow, -0.72, 0.58, 0.36));
  return g;
}

function buildChair(): THREE.Group {
  const g = new THREE.Group();
  const wood = mat(PALETTE.wood);
  g.add(box(0.5, 0.06, 0.5, mat(PALETTE.fabric), 0, 0.45, 0)); // seat
  g.add(box(0.5, 0.55, 0.06, wood, 0, 0.755, -0.22)); // back
  for (const [x, z] of [
    [-0.21, -0.21],
    [0.21, -0.21],
    [-0.21, 0.21],
    [0.21, 0.21],
  ]) {
    g.add(cyl(0.025, 0.025, 0.45, wood, x, 0.225, z));
  }
  return g;
}

function buildCouch(): THREE.Group {
  const g = new THREE.Group();
  const fabric = mat(PALETTE.fabric);
  const cushion = mat(PALETTE.body);
  g.add(box(1.8, 0.38, 0.85, fabric, 0, 0.29, 0)); // base
  g.add(box(1.8, 0.52, 0.22, fabric, 0, 0.72, -0.315)); // back
  g.add(box(0.22, 0.58, 0.85, cushion, -0.79, 0.39, 0)); // left arm
  g.add(box(0.22, 0.58, 0.85, cushion, 0.79, 0.39, 0)); // right arm
  return g;
}

function buildTv(): THREE.Group {
  const g = new THREE.Group();
  const dark = mat(PALETTE.dark);
  g.add(box(1.15, 0.68, 0.06, dark, 0, 0.76, 0)); // panel
  g.add(box(0.09, 0.36, 0.06, mat(PALETTE.metal), 0, 0.24, 0)); // neck
  g.add(box(0.5, 0.04, 0.24, dark, 0, 0.02, 0)); // base
  return g;
}

function buildLaptop(): THREE.Group {
  const g = new THREE.Group();
  const dark = mat(PALETTE.dark);
  g.add(box(0.34, 0.02, 0.24, mat(PALETTE.metal), 0, 0.01, 0)); // keyboard deck
  // Screen hinged at its bottom edge, tilted back ~110° open.
  const screen = box(0.34, 0.24, 0.015, dark);
  screen.geometry.translate(0, 0.12, 0); // pivot at bottom edge
  screen.position.set(0, 0.02, -0.115);
  screen.rotation.x = -0.35;
  g.add(screen);
  return g;
}

function buildCup(): THREE.Group {
  const g = new THREE.Group();
  g.add(cyl(0.045, 0.035, 0.1, mat(PALETTE.linen), 0, 0.05, 0));
  return g;
}

function buildBottle(): THREE.Group {
  const g = new THREE.Group();
  const body = mat(PALETTE.body);
  g.add(cyl(0.04, 0.04, 0.2, body, 0, 0.1, 0)); // body
  g.add(cyl(0.016, 0.028, 0.07, body, 0, 0.235, 0)); // shoulder + neck
  g.add(cyl(0.018, 0.018, 0.025, mat(PALETTE.dark), 0, 0.283, 0)); // cap
  return g;
}

function buildTable(): THREE.Group {
  const g = new THREE.Group();
  const wood = mat(PALETTE.wood);
  g.add(box(1.4, 0.06, 0.85, wood, 0, 0.72, 0)); // top
  for (const [x, z] of [
    [-0.62, -0.36],
    [0.62, -0.36],
    [-0.62, 0.36],
    [0.62, 0.36],
  ]) {
    g.add(cyl(0.035, 0.035, 0.69, wood, x, 0.345, z));
  }
  return g;
}

function buildCar(): THREE.Group {
  const g = new THREE.Group();
  g.add(box(4.0, 0.55, 1.75, mat(PALETTE.fabric), 0, 0.55, 0)); // body
  g.add(box(2.0, 0.45, 1.6, mat(PALETTE.body), -0.2, 1.05, 0)); // cabin
  const wheel = mat(PALETTE.dark);
  for (const [x, z] of [
    [-1.3, -0.88],
    [1.3, -0.88],
    [-1.3, 0.88],
    [1.3, 0.88],
  ]) {
    const w = cyl(0.33, 0.33, 0.22, wheel, x, 0.33, z);
    w.rotation.x = Math.PI / 2; // axle along z (car width)
    g.add(w);
  }
  return g;
}

/** Unknown class — translucent ghost box with visible edges. */
function buildGhost(): THREE.Group {
  const g = new THREE.Group();
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const fill = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({
      color: PALETTE.bodyLight,
      roughness: 0.8,
      metalness: 0.0,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
    }),
  );
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({
      color: PALETTE.bodyLight,
      transparent: true,
      opacity: 0.55,
    }),
  );
  g.add(fill, edges);
  // Shift up so the ghost sits 0..1 like the other builders (normalization
  // recenters anyway; this just keeps builder conventions consistent).
  fill.position.y = 0.5;
  edges.position.y = 0.5;
  return g;
}

const BUILDERS: Record<string, () => THREE.Group> = {
  person: buildPerson,
  bed: buildBed,
  chair: buildChair,
  couch: buildCouch,
  sofa: buildCouch,
  tv: buildTv,
  tvmonitor: buildTv,
  laptop: buildLaptop,
  cup: buildCup,
  bottle: buildBottle,
  table: buildTable,
  "dining table": buildTable,
  diningtable: buildTable,
  car: buildCar,
};
