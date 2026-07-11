import { useMemo } from "react";
import * as THREE from "three";
import type { Track } from "../lib/api";
import { useAnalysisStore } from "../state/analysis";

/**
 * Renders one animated sphere per track at its interpolated 3D position for
 * the current video time. The sphere colour + emissive intensity is driven
 * by the per-frame σ for that track (green under 3σ, amber to 10σ, red over
 * 10σ). Peak-σ track scales up so the offending object is visually obvious
 * during scrubbing.
 */
export function TrackMarkers({ tracks }: { tracks: Track[] }) {
  const currentTimeS = useAnalysisStore((s) => s.currentTimeS);

  const markers = useMemo(() => {
    return tracks.map((track) => {
      const pos = interpolatePosition(track, currentTimeS);
      const sigma = interpolateSigma(track, currentTimeS);
      return { track, pos, sigma };
    });
    // tracks identity + currentTimeS drive recompute
  }, [tracks, currentTimeS]);

  if (markers.length === 0) return null;

  const peakSigma = markers.reduce((m, x) => Math.max(m, x.sigma), 0);

  return (
    <group>
      {markers.map(({ track, pos, sigma }) => {
        const isPeak = sigma > 0 && sigma === peakSigma;
        const color = sigmaToColor(sigma);
        const emissive = sigmaToEmissive(sigma);
        const size = 0.02 + (isPeak ? 0.03 : 0.0) + Math.min(sigma / 40, 0.03);
        return (
          <mesh key={track.track_id} position={pos}>
            <sphereGeometry args={[size, 16, 16]} />
            <meshStandardMaterial
              color={color}
              emissive={emissive}
              emissiveIntensity={0.8 + Math.min(sigma / 5, 3)}
              roughness={0.35}
              metalness={0.0}
              toneMapped={false}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function interpolatePosition(
  track: Track,
  t: number,
): [number, number, number] {
  const points = track.points;
  if (points.length === 0) return [0, 0, 0];
  if (t <= points[0].t_s) return points[0].position;
  const last = points[points.length - 1];
  if (t >= last.t_s) return last.position;
  for (let i = 0; i < points.length - 1; i++) {
    if (t >= points[i].t_s && t <= points[i + 1].t_s) {
      const dt = points[i + 1].t_s - points[i].t_s;
      const alpha = dt <= 0 ? 0 : (t - points[i].t_s) / dt;
      const a = points[i].position;
      const b = points[i + 1].position;
      return [
        a[0] + alpha * (b[0] - a[0]),
        a[1] + alpha * (b[1] - a[1]),
        a[2] + alpha * (b[2] - a[2]),
      ];
    }
  }
  return last.position;
}

function interpolateSigma(track: Track, t: number): number {
  const sigmas = track.sigma_per_frame;
  if (!sigmas || sigmas.length === 0) return 0;
  const points = track.points;
  if (sigmas.length !== points.length) return 0;
  if (t <= points[0].t_s) return sigmas[0];
  const lastIdx = points.length - 1;
  if (t >= points[lastIdx].t_s) return sigmas[lastIdx];
  for (let i = 0; i < points.length - 1; i++) {
    if (t >= points[i].t_s && t <= points[i + 1].t_s) {
      const dt = points[i + 1].t_s - points[i].t_s;
      const alpha = dt <= 0 ? 0 : (t - points[i].t_s) / dt;
      return sigmas[i] + alpha * (sigmas[i + 1] - sigmas[i]);
    }
  }
  return sigmas[lastIdx];
}

function sigmaToColor(sigma: number): THREE.Color {
  // < 3σ = emerald; 3–10σ = amber; > 10σ = red.
  const c = new THREE.Color();
  if (sigma < 3) return c.setHex(0x4bff88);
  if (sigma < 10) return c.setHex(0xffb84b);
  return c.setHex(0xff4b4b);
}

function sigmaToEmissive(sigma: number): THREE.Color {
  const c = new THREE.Color();
  if (sigma < 3) return c.setHex(0x1f4a2b);
  if (sigma < 10) return c.setHex(0x664b1a);
  return c.setHex(0x661a1a);
}
