import useZoomLevel from "../../hooks/useZoomLevel";

const LEVEL_LABELS = {
  L0: "Architecture",
  L1: "VSS Block",
  L2: "SS2D",
  L3: "SSM Cell",
};

/**
 * Top navigation bar showing breadcrumb trail and back button.
 */
function NavBar() {
  const { level, zoomOut, canZoomOut } = useZoomLevel();

  const levels = ["L0", "L1", "L2", "L3"];
  const currentIdx = levels.indexOf(level);
  const breadcrumbs = levels.slice(0, currentIdx + 1);

  return (
    <nav className="navbar">
      <span className="navbar-brand">Vision Mamba Explainer</span>

      <ol className="breadcrumbs" aria-label="Zoom level">
        {breadcrumbs.map((l, i) => (
          <li key={l} className={l === level ? "active" : ""}>
            {i > 0 && <span className="breadcrumb-sep">›</span>}
            <span className="breadcrumb-label">{LEVEL_LABELS[l]}</span>
          </li>
        ))}
      </ol>

      <button
        className="nav-back-btn"
        onClick={zoomOut}
        disabled={!canZoomOut}
        aria-label="Zoom out one level"
      >
        ← Back
      </button>
    </nav>
  );
}

export default NavBar;
