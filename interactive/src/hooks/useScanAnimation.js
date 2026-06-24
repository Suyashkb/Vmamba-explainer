import { useRef, useCallback, useEffect } from "react";
import useExplainerStore from "../store/explainerStore";

/**
 * useScanAnimation — rAF-based stepper for the scan visualisation.
 *
 * Throttled to `fps` (default 12) so React reconciliation never chokes on
 * large grids. Drives store.step so L2 and L3 subscribe to the same counter.
 *
 * @param {number} L    total number of timesteps (196 for L2, 16 for L3)
 * @param {object} opts
 * @param {number} [opts.fps=12]   frames per second cap
 * @param {boolean} [opts.loop=false]  restart from 0 when L is reached
 *
 * Returns { step, playing, play, pause, seek, reset }
 */
function useScanAnimation(L, { fps = 12, loop = false } = {}) {
  const step     = useExplainerStore((s) => s.step);
  const playing  = useExplainerStore((s) => s.playing);
  const setStep  = useExplainerStore((s) => s.setStep);
  const setPlaying = useExplainerStore((s) => s.setPlaying);

  const rafRef   = useRef(null);
  const lastRef  = useRef(0);
  const stepRef  = useRef(step);
  const interval = 1000 / fps;

  // Keep stepRef fresh so the rAF loop never reads a stale closure value.
  stepRef.current = step;

  // Cancel any pending frame on unmount.
  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  // Core loop — defined with useCallback so play() can capture a stable ref.
  const makeLoop = useCallback(() => {
    const loop_ = (ts) => {
      if (ts - lastRef.current >= interval) {
        lastRef.current = ts;
        const next = stepRef.current + 1;
        if (next >= L) {
          if (loop) {
            setStep(0);
          } else {
            setPlaying(false);
            rafRef.current = null;
            return;
          }
        } else {
          setStep(next);
        }
      }
      rafRef.current = requestAnimationFrame(loop_);
    };
    return loop_;
  }, [L, interval, loop, setStep, setPlaying]);

  const play = useCallback(() => {
    if (rafRef.current) return; // already running
    if (stepRef.current >= L - 1) setStep(0); // restart from top
    setPlaying(true);
    lastRef.current = performance.now() - interval; // fire on first tick
    rafRef.current = requestAnimationFrame(makeLoop());
  }, [L, interval, makeLoop, setStep, setPlaying]);

  const pause = useCallback(() => {
    setPlaying(false);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [setPlaying]);

  const seek = useCallback((t) => {
    pause();
    setStep(Math.max(0, Math.min(L - 1, t)));
  }, [pause, setStep, L]);

  const reset = useCallback(() => {
    pause();
    setStep(0);
  }, [pause, setStep]);

  return { step, playing, play, pause, seek, reset };
}

export default useScanAnimation;
