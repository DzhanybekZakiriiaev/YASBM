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
import { DynamicPoints } from "./DynamicPoints";
import { PointCloud } from "./PointCloud";
import { Props } from "./Props";
import { SceneMesh } from "./SceneMesh";
import { TrackMarkers } from "./TrackMarkers";

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
  const showProps = useAnalysisStore((s) => s.showProps);
  const toggleProps = useAnalysisStore((s) => s.toggleProps);
  const tracks = analysisResult?.tracks ?? [];
  const pointCloudUrl = analysisResult?.point_cloud_url ?? null;
  const dynamicPointsUrl = analysisResult?.dynamic_points_url ?? null;
  const hasProps = (analysisResult?.props?.length ?? 0) > 0;

  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-neutral-800 bg-neutral-950">
      <div className="pointer-events-none absolute left-3 top-3 z-10 font-mono text-[10px] uppercase tracking-widest text-neutral-500">
        3d scene · {tracks.length} track{tracks.length === 1 ? "" : "s"}
        {pointCloudUrl ? " · cloud" : ""}
      </div>
      {hasProps ? (
        <button
          type="button"
          onClick={toggleProps}
          className="absolute bottom-3 left-3 z-10 rounded-md border border-neutral-800 bg-neutral-900/80 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-neutral-400 transition-colors hover:border-neutral-700 hover:bg-neutral-800 hover:text-neutral-200"
        >
          props: {showProps ? "on" : "off"}
        </button>
      ) : null}
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
        {/* Background is set by the <Environment> HDRI below; keep this
            fallback so scenes without an HDRI aren't blinding white. */}
        <color attach="background" args={["#050505"]} />
        <fog attach="fog" args={["#050505", 8, 42]} />

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

        {/* HDRI for material sheen ONLY — as a visible background it washed
            the whole scene out white and destroyed the mesh's contrast.
            Dark void + fog reads better for a forensic viewer. */}
        <Environment
          preset="apartment"
          environmentIntensity={0.45}
          background={false}
        />

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

        <SceneContents
          tracks={tracks}
          pointCloudUrl={pointCloudUrl}
          dynamicPointsUrl={dynamicPointsUrl}
        />

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
  dynamicPointsUrl: string | null;
}

function SceneContents({
  tracks,
  pointCloudUrl,
  dynamicPointsUrl,
}: SceneContentsProps) {
  const groupRef = useRef<Group>(null);

  return (
    <>
      <CinematicCamera tracks={tracks} />
      {pointCloudUrl ? (
        <>
          {/* Rendered as a solid mesh when the PLY has faces; otherwise
              the SceneMesh returns null and the PointCloud takes over. */}
          <SceneMesh url={pointCloudUrl} />
          <PointCloud url={pointCloudUrl} />
        </>
      ) : null}
      {/* Moving pixels re-rendered per frame, synced to the playhead. */}
      {dynamicPointsUrl ? <DynamicPoints url={dynamicPointsUrl} /> : null}
      {/* Stylized 3D props at each detected object's bbox (GLB or proxy). */}
      <Props />
      <group ref={groupRef}>
        {tracks.map((track, i) => (
          <TrackLine
            key={track.track_id}
            track={track}
            color={TRACK_COLORS[i % TRACK_COLORS.length]}
          />
        ))}
      </group>
      {/* Animated per-frame track markers — colored spheres at each track's
          current 3D position, scaled and coloured by σ. Peak-σ track glows
          red so the physics violation is visible in space at the exact
          timestamp you're scrubbed to. */}
      <TrackMarkers tracks={tracks} />
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
 * Spawn the viewer exactly at the recording camera's pose.
 *
 * The pipeline back-projects with a pinhole model: camera at the origin
 * looking down +Z, focal length fx = frame_width. Reproducing that pose
 * plus the matching vertical FOV (2·atan(H / 2W)) makes the depth mesh
 * fill the viewport exactly like frame 0 of the video — no hunting for
 * the angle where the projection "lines up". Orbit starts from there,
 * pivoting around the scene's depth centroid.
 */
function CinematicCamera({ tracks }: { tracks: Track[] }) {
  const camera = useThree((s) => s.camera) as PerspectiveCameraImpl;
  const controls = useThree((s) => s.controls) as
    | { target: THREE.Vector3; update: () => void }
    | null;
  const frameWidth = useAnalysisStore(
    (s) => s.analysisResult?.frame_width ?? null,
  );
  const frameHeight = useAnalysisStore(
    (s) => s.analysisResult?.frame_height ?? null,
  );

  useEffect(() => {
    if (!tracks.length) return;

    // Median track depth = a robust "middle of the scene" for the orbit
    // pivot. Track positions are already in the camera's world frame.
    const zs: number[] = [];
    for (const track of tracks) {
      for (const p of track.points) zs.push(p.position[2]);
    }
    zs.sort((a, b) => a - b);
    const zMid = zs.length ? zs[Math.floor(zs.length / 2)] : 3;
    const target = new THREE.Vector3(0, 0, Math.max(zMid, 0.5));

    // Match the recording camera's intrinsics: fx = frame width, so the
    // vertical FOV is 2·atan((H/2) / W). Falls back to 45° when the
    // response predates the frame-dimension fields.
    if (frameWidth && frameHeight) {
      camera.fov =
        (2 * Math.atan(frameHeight / (2 * frameWidth)) * 180) / Math.PI;
    } else {
      camera.fov = 45;
    }

    camera.position.set(0, 0, 0);
    camera.near = 0.05;
    camera.far = 200;
    camera.lookAt(target);
    camera.updateProjectionMatrix();

    if (controls && "target" in controls) {
      controls.target.copy(target);
      controls.update();
    }
  }, [tracks, camera, controls, frameWidth, frameHeight]);

  return null;
}
