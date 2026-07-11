import { useEffect, useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import {
  Environment,
  GizmoHelper,
  GizmoViewport,
  Grid,
  Line,
  OrbitControls,
  PerspectiveCamera,
} from "@react-three/drei";
import {
  Bloom,
  EffectComposer,
  ToneMapping,
  Vignette,
} from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";
import * as THREE from "three";
import { NoToneMapping, type Group, type PerspectiveCamera as PerspectiveCameraImpl } from "three";
import { useAnalysisStore } from "../state/analysis";
import type { Track } from "../lib/api";
import { PointCloud } from "./PointCloud";

/** Emissive palette for trajectories — hot enough that Bloom picks them up
 *  at luminanceThreshold=0.3 while still reading as their intended colors. */
const TRACK_COLORS = [
  "#ff4b4b", // observed / red
  "#4bff88", // predicted / green
  "#4bb8ff", // cyan
  "#ffb84b", // amber
  "#c66bff", // violet
];

export function Viewer3D() {
  const analysisResult = useAnalysisStore((s) => s.analysisResult);
  const tracks = analysisResult?.tracks ?? [];
  const pointCloudUrl = analysisResult?.point_cloud_url ?? null;

  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-neutral-800 bg-neutral-950">
      <div className="pointer-events-none absolute left-3 top-3 z-10 font-mono text-[10px] uppercase tracking-widest text-neutral-500">
        3d scene · {tracks.length} track{tracks.length === 1 ? "" : "s"}
        {pointCloudUrl ? " · cloud" : ""}
      </div>
      <Canvas
        dpr={[1, 2]}
        // Tone mapping is handled by the <ToneMapping> effect below. Setting
        // it on the Canvas GL context as well caused Bloom to read+write the
        // same depth-stencil buffer and Chrome spammed `glBlitFramebuffer:
        // Read and write depth stencil attachments cannot be the same image`
        // until the WebGL context died. NoToneMapping keeps the render linear
        // until the composer applies ACES at the very end.
        gl={{
          antialias: false,
          powerPreference: "high-performance",
          toneMapping: NoToneMapping,
          stencil: false,
        }}
      >
        <color attach="background" args={["#050505"]} />

        <PerspectiveCamera makeDefault position={[3, 2, 4]} fov={45} />

        {/* Cinematic 3-point rig: low ambient, warm key, cool rim. */}
        <ambientLight intensity={0.18} />
        <directionalLight
          position={[6, 8, 4]}
          intensity={2.4}
          color="#ffd6a5"
          castShadow={false}
        />
        <directionalLight
          position={[-5, 3, -6]}
          intensity={1.1}
          color="#7fb8ff"
        />

        {/* Subtle sheen from an HDRI without lighting the scene through it. */}
        <Environment preset="city" environmentIntensity={0.35} background={false} />

        <Grid
          infiniteGrid
          cellSize={0.4}
          cellThickness={0.6}
          cellColor="#3a2e26"
          sectionSize={2}
          sectionThickness={1.2}
          sectionColor="#6b4a34"
          fadeDistance={28}
          fadeStrength={1.2}
        />

        <SceneContents tracks={tracks} pointCloudUrl={pointCloudUrl} />

        <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
        <GizmoHelper alignment="top-right" margin={[64, 64]}>
          <GizmoViewport
            axisColors={["#ff4b4b", "#4bff88", "#4bb8ff"]}
            labelColor="#e5e5e5"
          />
        </GizmoHelper>

        {/* N8AO removed for now — it requires a normal pass that fights with
            Bloom's depth-stencil handling on Chrome's WebGL2 driver. Bloom +
            filmic tone map + vignette carry the cinematic feel on their own. */}
        <EffectComposer multisampling={4}>
          <Bloom
            mipmapBlur
            intensity={1.4}
            luminanceThreshold={0.3}
            luminanceSmoothing={0.9}
          />
          <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
          <Vignette darkness={0.4} offset={0.2} />
        </EffectComposer>
      </Canvas>
    </div>
  );
}

interface SceneContentsProps {
  tracks: Track[];
  pointCloudUrl: string | null;
}

function SceneContents({ tracks, pointCloudUrl }: SceneContentsProps) {
  const groupRef = useRef<Group>(null);

  return (
    <>
      <CinematicCamera tracks={tracks} />
      {pointCloudUrl ? <PointCloud url={pointCloudUrl} /> : null}
      <group ref={groupRef}>
        {tracks.map((track, i) => (
          <TrackLine
            key={track.track_id}
            track={track}
            color={TRACK_COLORS[i % TRACK_COLORS.length]}
          />
        ))}
      </group>
    </>
  );
}

interface TrackLineProps {
  track: Track;
  color: string;
}

function TrackLine({ track, color }: TrackLineProps) {
  const points = useMemo<[number, number, number][]>(
    () =>
      track.points.map((p) => [
        p.position[0],
        p.position[1],
        p.position[2],
      ]),
    [track],
  );
  if (points.length < 2) return null;
  // Drei's <Line> is a MeshLine internally — thicker than gl.LINES and its
  // color is picked up by our Bloom pass at the 0.3 luminance threshold.
  return (
    <Line
      points={points}
      color={color}
      lineWidth={3}
      transparent
      opacity={0.95}
      toneMapped={false}
    />
  );
}

/**
 * Auto-frame the camera to the union bounding box of all trajectory points.
 * Runs once on mount and every time the tracks list changes identity.
 * Zooms out ~30% so the trajectory reads with breathing room.
 */
function CinematicCamera({ tracks }: { tracks: Track[] }) {
  const camera = useThree((s) => s.camera) as PerspectiveCameraImpl;
  const controls = useThree((s) => s.controls) as
    | { target: THREE.Vector3; update: () => void }
    | null;

  useEffect(() => {
    if (!tracks.length) return;

    const box = new THREE.Box3();
    const tmp = new THREE.Vector3();
    let empty = true;
    for (const track of tracks) {
      for (const p of track.points) {
        tmp.set(p.position[0], p.position[1], p.position[2]);
        if (empty) {
          box.min.copy(tmp);
          box.max.copy(tmp);
          empty = false;
        } else {
          box.expandByPoint(tmp);
        }
      }
    }
    if (empty) return;

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    // Fit the largest dimension into ~60% of vertical FOV, then push back
    // by another 30% for breathing room.
    const maxDim = Math.max(size.x, size.y, size.z, 0.5);
    const fovRad = ((camera.fov ?? 45) * Math.PI) / 180;
    let distance = (maxDim / 2) / Math.tan(fovRad / 2);
    distance *= 1.3; // breathing room

    // Frame from a cinematic 3/4 angle: slightly high, slightly to the right.
    const dir = new THREE.Vector3(0.9, 0.55, 1.0).normalize();
    const camPos = center.clone().addScaledVector(dir, distance);

    camera.position.copy(camPos);
    camera.near = Math.max(0.01, distance * 0.01);
    camera.far = Math.max(100, distance * 20);
    camera.lookAt(center);
    camera.updateProjectionMatrix();

    if (controls && "target" in controls) {
      controls.target.copy(center);
      controls.update();
    }
  }, [tracks, camera, controls]);

  return null;
}
