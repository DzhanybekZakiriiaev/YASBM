import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { loadPLYPointCloud } from "../three/pointCloud";

/**
 * Cinematic point cloud renderer.
 *
 * Reads a PLY (position + color) and renders it with a custom ShaderMaterial:
 *   - vertex   → size attenuation by camera distance + gentle pulse
 *   - fragment → circular sprite with soft alpha edge (discards corners)
 *
 * Target visual: ~2px at 1m, dust-fine at long range, subtle "breathing"
 * shimmer to keep the scene alive.
 */
interface PointCloudProps {
  url: string;
  /** Base point size in "world" units before attenuation. Default 0.012. */
  baseSize?: number;
  /** How strongly points pulse. 0 = static. Default 0.08. */
  pulse?: number;
}

const vertexShader = /* glsl */ `
  attribute vec3 color;
  varying vec3 vColor;

  uniform float uBaseSize;
  uniform float uPulse;
  uniform float uTime;
  uniform float uPixelRatio;
  uniform float uViewportHeight;

  void main() {
    vColor = color;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float dist = -mvPosition.z;

    // Size attenuation: point size grows as we get closer.
    // The magic constant maps "1 world unit at 1m" ~ 2px.
    float size = uBaseSize * (uViewportHeight * 0.5) / max(dist, 0.0001);

    // Gentle uniform pulse — the whole cloud "breathes" a tiny bit.
    float breathe = 1.0 + uPulse * sin(uTime * 1.3);
    size *= breathe;

    // Clamp so nothing becomes a screen-clearing blob or vanishes.
    size = clamp(size, 1.0, 32.0);

    gl_PointSize = size * uPixelRatio;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = /* glsl */ `
  varying vec3 vColor;

  void main() {
    // Discard corners so square points read as round.
    vec2 uv = gl_PointCoord - vec2(0.5);
    float r = length(uv);
    if (r > 0.5) discard;

    // Soft edge falloff.
    float alpha = smoothstep(0.5, 0.15, r);

    // Slight center hot-spot so bloom picks up bright points.
    float core = smoothstep(0.35, 0.0, r);
    vec3 col = vColor + core * vColor * 0.6;

    gl_FragColor = vec4(col, alpha);
  }
`;

export function PointCloud({
  url,
  baseSize = 0.012,
  pulse = 0.08,
}: PointCloudProps) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);

  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    let cancelled = false;
    loadPLYPointCloud(url, controller.signal)
      .then((geo) => {
        if (!cancelled) setGeometry(geo);
      })
      .catch((err: unknown) => {
        // Surface a helpful console warning; don't crash the viewer.
        if ((err as { name?: string })?.name !== "AbortError") {
          console.warn("[PointCloud] failed to load PLY", err);
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
      // Dispose the previous geometry to free GPU memory.
      setGeometry((prev) => {
        prev?.dispose();
        return null;
      });
    };
  }, [url]);

  const uniforms = useMemo(
    () => ({
      uBaseSize: { value: baseSize },
      uPulse: { value: pulse },
      uTime: { value: 0 },
      uPixelRatio: {
        value: typeof window !== "undefined" ? window.devicePixelRatio : 1,
      },
      uViewportHeight: {
        value: typeof window !== "undefined" ? window.innerHeight : 800,
      },
    }),
    [baseSize, pulse],
  );

  useFrame((state, delta) => {
    if (!materialRef.current) return;
    materialRef.current.uniforms.uTime.value += delta;
    // Track viewport height so size attenuation stays consistent on resize.
    const size = state.size;
    materialRef.current.uniforms.uViewportHeight.value = size.height;
  });

  if (!geometry) return null;

  return (
    <points frustumCulled={false}>
      <primitive object={geometry} attach="geometry" />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.NormalBlending}
      />
    </points>
  );
}
