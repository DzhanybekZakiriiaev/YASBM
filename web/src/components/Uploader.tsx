import { useCallback, useRef, useState } from "react";
import clsx from "clsx";
import { useAnalysisStore } from "../state/analysis";

export function Uploader() {
  const setVideo = useAnalysisStore((s) => s.setVideo);
  const runAnalyze = useAnalysisStore((s) => s.runAnalyze);
  const inputRef = useRef<HTMLInputElement>(null);
  const [hover, setHover] = useState(false);

  const accept = useCallback(
    (file: File) => {
      if (!file.type.startsWith("video/")) return;
      setVideo(file);
      // Fire the analysis immediately — Zustand's set is synchronous,
      // so runAnalyze() will see the just-set videoFile via get().
      void runAnalyze();
    },
    [setVideo, runAnalyze],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setHover(false);
      const file = e.dataTransfer.files?.[0];
      if (file) accept(file);
    },
    [accept],
  );

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) accept(file);
    },
    [accept],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setHover(true);
      }}
      onDragLeave={() => setHover(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={clsx(
        "flex h-full w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-lg border-2 border-dashed p-12 transition-colors",
        hover
          ? "border-neutral-400 bg-neutral-900/60"
          : "border-neutral-700 bg-neutral-950 hover:border-neutral-500 hover:bg-neutral-900/40",
      )}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="h-12 w-12 text-neutral-400"
        aria-hidden
      >
        <path d="M12 3v13" />
        <path d="m7 8 5-5 5 5" />
        <path d="M5 21h14" />
      </svg>
      <div className="text-center">
        <div className="text-sm font-medium tracking-wide text-neutral-200">
          drop a video clip
        </div>
        <div className="mt-1 text-xs text-neutral-500">
          mp4 / mov / webm — 2 to 6 seconds recommended
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={onChange}
      />
    </div>
  );
}
