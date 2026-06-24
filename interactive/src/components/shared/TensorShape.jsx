/**
 * Small badge that displays a tensor shape string.
 * e.g. "(B, 56, 56, 96)"
 */
function TensorShape({ shape }) {
  if (!shape) return null;
  return <span className="tensor-shape-badge">{shape}</span>;
}

export default TensorShape;
