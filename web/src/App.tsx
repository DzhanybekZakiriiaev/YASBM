import { useAnalysisStore } from "./state/analysis";
import { Uploader } from "./components/Uploader";
import { Player } from "./components/Player";
import { Viewer3D } from "./components/Viewer3D";
import { Timeline } from "./components/Timeline";
import { Verdict } from "./components/Verdict";
import { ObjectList } from "./components/ObjectList";
import { FlythroughButton } from "./components/FlythroughButton";

export default function App() {
  const videoUrl = useAnalysisStore((s) => s.videoUrl);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-neutral-200">
      <header className="flex items-center justify-between border-b border-neutral-900 px-6 py-4">
        <div className="flex items-baseline gap-4">
          <h1 className="font-mono text-2xl font-semibold tracking-[0.2em] text-neutral-100">
            KEPLER
          </h1>
          <div className="text-xs uppercase tracking-widest text-neutral-500">
            physics-based video plausibility auditor
          </div>
        </div>
        <div className="rounded-full border border-neutral-800 bg-neutral-950 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-neutral-400">
          day 1 · scaffold
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 p-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <section className="min-h-[420px]">
            {videoUrl ? <Player /> : <Uploader />}
          </section>
          <section className="flex flex-col gap-4">
            <Viewer3D />
            <FlythroughButton />
          </section>
        </div>
        {/* Full-width timeline strip — scrubbing here drives the video AND
            the 3D scene's animated track markers, so hovering a peak σ
            frame instantly reveals the offending object in 3D. */}
        <Timeline />
        <ObjectList />
        <Verdict />
      </main>
    </div>
  );
}
