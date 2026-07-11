import { useAnalysisStore } from "./state/analysis";
import { Uploader } from "./components/Uploader";
import { Player } from "./components/Player";
import { Viewer3D } from "./components/Viewer3D";
import { ResidualChart } from "./components/ResidualChart";
import { Verdict } from "./components/Verdict";

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

      <main className="grid grid-cols-1 gap-6 p-6 lg:grid-cols-2">
        <section className="min-h-[420px]">
          {videoUrl ? <Player /> : <Uploader />}
        </section>
        <section className="flex flex-col gap-6">
          <Viewer3D />
          <ResidualChart />
          <Verdict />
        </section>
      </main>
    </div>
  );
}
