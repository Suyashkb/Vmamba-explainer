import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

/**
 * Renders a LaTeX equation string via KaTeX.
 *
 * Props
 * -----
 * tex        {string}  LaTeX source, e.g. "\\bar{A}_t = \\exp(\\Delta_t A)"
 * display    {boolean} true → display (block) mode; false → inline
 * className  {string}  extra CSS classes
 */
function Equation({ tex, display = false, className = "" }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !tex) return;
    try {
      katex.render(tex, ref.current, {
        displayMode: display,
        throwOnError: false,
        trust: false,
      });
    } catch {
      ref.current.textContent = tex;
    }
  }, [tex, display]);

  return <span ref={ref} className={`equation ${className}`} />;
}

export default Equation;
