"""KEPLER inference pipeline package.

Six-stage pipeline that takes a video clip and produces 3D artifacts plus
physics residuals for the browser front-end:

    segment -> track -> depth -> scene -> lift -> physics -> package

Real model integrations (SAM 2.1, CoTracker3, Video Depth Anything,
VGGT / MonST3R) land in subsequent milestones; the current build ships
runnable stubs plus a real back-projection ``lift`` stage so downstream
code has usable shapes to consume.
"""

__version__ = "0.0.1"
