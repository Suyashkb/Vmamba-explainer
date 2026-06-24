import { useEffect, useRef } from "react";
import * as d3 from "d3";

/**
 * Generic DAG renderer shared by L0 and L1.
 *
 * Uses precomputed x/y coordinates from architecture.json (deterministic
 * layout — architecture diagrams must be stable, not springy force layouts).
 *
 * @param {React.RefObject} svgRef  — ref to the <svg> element
 * @param {object}          graph   — { nodes: [{id,label,shape,eq,x,y,zoomTo?}],
 *                                      edges: [[sourceId, targetId], ...] }
 * @param {object}          opts
 * @param {function}        opts.onNodeClick  — called with node object on click
 * @param {string|null}     opts.selectedId   — id of currently selected node
 */
function useD3Graph(svgRef, graph, { onNodeClick, selectedId } = {}) {
  const zoomRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !graph) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { nodes, edges } = graph;

    // --- Bounding box ---
    const xs = nodes.map((n) => n.x);
    const ys = nodes.map((n) => n.y);
    const pad = 80;
    const minX = Math.min(...xs) - pad;
    const minY = Math.min(...ys) - pad;
    const maxX = Math.max(...xs) + 160 + pad;
    const maxY = Math.max(...ys) + 60  + pad;

    svg.attr("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`)
       .attr("preserveAspectRatio", "xMidYMid meet")
       .style("width", "100%")
       .style("height", "100%");

    // Arrowhead marker
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 10).attr("refY", 0)
      .attr("markerWidth", 8).attr("markerHeight", 8)
      .attr("orient", "auto")
      .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "var(--color-edge)");

    const g = svg.append("g").attr("class", "graph-root");

    // --- d3 zoom ---
    const zoom = d3.zoom()
      .scaleExtent([0.3, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    // --- Node lookup map ---
    const nodeMap = Object.fromEntries(nodes.map((n) => [n.id, n]));

    // --- Edges ---
    const edgeG = g.append("g").attr("class", "edges");
    edges.forEach(([srcId, tgtId]) => {
      const src = nodeMap[srcId];
      const tgt = nodeMap[tgtId];
      if (!src || !tgt) return;

      // Simple horizontal link with a midpoint
      const x1 = src.x + 120;   // right edge of source node (approx)
      const y1 = src.y + 22;
      const x2 = tgt.x;
      const y2 = tgt.y + 22;

      edgeG.append("path")
        .attr("d", `M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`)
        .attr("fill", "none")
        .attr("stroke", "var(--color-edge)")
        .attr("stroke-width", 1.5)
        .attr("marker-end", "url(#arrow)");
    });

    // --- Nodes ---
    const nodeG = g.append("g").attr("class", "nodes");
    const nodeW = 130;
    const nodeH = 44;

    nodes.forEach((node) => {
      const isSelected  = node.id === selectedId;
      const isClickable = Boolean(node.zoomTo) || Boolean(onNodeClick);

      const group = nodeG.append("g")
        .attr("class", "node")
        .attr("transform", `translate(${node.x}, ${node.y})`)
        .style("cursor", isClickable ? "pointer" : "default")
        .on("click", () => onNodeClick && onNodeClick(node));

      // Box
      group.append("rect")
        .attr("width", nodeW)
        .attr("height", nodeH)
        .attr("rx", 6)
        .attr("ry", 6)
        .attr("fill", isSelected ? "var(--color-node-selected-bg)"
                                 : node.zoomTo ? "var(--color-node-zoomable-bg)"
                                               : "var(--color-node-bg)")
        .attr("stroke", isSelected ? "var(--color-node-selected-border)"
                                   : "var(--color-node-border)")
        .attr("stroke-width", isSelected ? 2 : 1);

      // Label (up to 2 lines via tspan)
      const lines = node.label.split("\n");
      const text = group.append("text")
        .attr("x", nodeW / 2)
        .attr("y", lines.length > 1 ? 14 : 24)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("fill", "var(--color-node-text)")
        .attr("font-size", "11px")
        .attr("font-family", "var(--font-mono)");

      lines.forEach((line, i) => {
        text.append("tspan")
          .attr("x", nodeW / 2)
          .attr("dy", i === 0 ? 0 : "1.2em")
          .text(line);
      });

      // Shape badge (small, below label)
      if (node.shape) {
        group.append("text")
          .attr("x", nodeW / 2)
          .attr("y", nodeH - 8)
          .attr("text-anchor", "middle")
          .attr("fill", "var(--color-shape-text)")
          .attr("font-size", "9px")
          .attr("font-family", "var(--font-mono)")
          .text(node.shape);
      }

      // Zoom indicator arrow for zoomable nodes
      if (node.zoomTo) {
        group.append("text")
          .attr("x", nodeW - 8)
          .attr("y", 14)
          .attr("text-anchor", "end")
          .attr("fill", "var(--color-accent)")
          .attr("font-size", "11px")
          .text("⊕");
      }
    });

    return () => {
      svg.on(".zoom", null);
    };
  }, [svgRef, graph, selectedId, onNodeClick]);

  return { zoomRef };
}

export default useD3Graph;
