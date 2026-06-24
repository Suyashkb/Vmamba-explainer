import { useMemo } from "react";

/**
 * Viridis colour scale — 8 control points, linearly interpolated.
 * Maps t ∈ [0,1] → [r, g, b].
 */
const VIRIDIS = [
  [68,  1,   84],
  [70,  50,  126],
  [54,  92,  141],
  [39,  127, 142],
  [31,  161, 135],
  [74,  194, 109],
  [159, 218,  58],
  [253, 231,  37],
];

/** Returns [r, g, b] integers — for canvas ImageData pixel writes. */
export function viridisRGB(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const n = VIRIDIS.length - 1;
  const i = Math.min(Math.floor(clamped * n), n - 1);
  const f = clamped * n - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[i + 1];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}

export function colorFromT(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const n = VIRIDIS.length - 1;
  const i = Math.min(Math.floor(clamped * n), n - 1);
  const f = clamped * n - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[i + 1];
  return `rgb(${Math.round(a[0] + f * (b[0] - a[0]))},${Math.round(a[1] + f * (b[1] - a[1]))},${Math.round(a[2] + f * (b[2] - a[2]))})`;
}

/**
 * HeatmapCells — renders the grid as <rect> elements with NO wrapper.
 * Embed directly inside an existing <svg> or <g>.
 *
 * @param {{ h, w, values, vmin, vmax }} data
 * @param {number} width   pixel width of the grid
 * @param {number} height  pixel height of the grid
 */
export function HeatmapCells({ data, width, height }) {
  const { h, w, values, vmin, vmax } = data;
  const range = (vmax - vmin) || 1;
  const cw = width  / w;
  const ch = height / h;

  const fills = useMemo(
    () => values.map((v) => colorFromT((v - vmin) / range)),
    [values, vmin, range],
  );

  return (
    <>
      {fills.map((fill, i) => (
        <rect
          key={i}
          x={(i % w) * cw}
          y={Math.floor(i / w) * ch}
          width={cw}
          height={ch}
          fill={fill}
        />
      ))}
    </>
  );
}

/**
 * Heatmap — standalone component (renders its own <svg> wrapper).
 *
 * @param {{ h, w, values, vmin, vmax }} data
 * @param {number} width
 * @param {number} height
 * @param {string} className
 * @param {string} title     optional accessible label
 */
function Heatmap({ data, width = 140, height = 140, className = "", title }) {
  return (
    <svg
      width={width}
      height={height}
      className={`heatmap ${className}`}
      role="img"
      aria-label={title ?? "feature heatmap"}
      style={{ display: "block", borderRadius: 2 }}
    >
      {title && <title>{title}</title>}
      <HeatmapCells data={data} width={width} height={height} />
    </svg>
  );
}

export default Heatmap;
