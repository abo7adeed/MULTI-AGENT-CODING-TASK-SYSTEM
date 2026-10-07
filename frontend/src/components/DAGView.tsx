import { useMemo, useState } from "react";
import type { Dag, DagNode } from "../api/client";
import { EDGE_CLASS, NODE_TONE, cx, formatDuration, titleCase } from "../lib/ui";
import { StatusBadge } from "./StatusBadge";

interface Props {
  dag: Dag;
  onSelect?: (node: DagNode) => void;
}

const NODE_W = 200;
const NODE_H = 74;
const GAP_X = 28;
const GAP_Y = 34;
const PAD = 24;

/**
 * The execution graph.
 *
 * Laid out by dependency wave (column) rather than by free-form graph
 * coordinates: every task in wave N depends only on waves before it, so a
 * column layout is exactly the temporal structure of the run and it stays
 * readable at any size without a layout engine.
 */
export function DAGView({ dag, onSelect }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  const layout = useMemo(() => {
    const byWave = new Map<number, DagNode[]>();
    for (const node of dag.nodes) {
      const list = byWave.get(node.wave) ?? [];
      list.push(node);
      byWave.set(node.wave, list);
    }
    const waves = [...byWave.entries()].sort((a, b) => a[0] - b[0]);

    const positions = new Map<string, { x: number; y: number }>();
    let maxRows = 1;
    waves.forEach(([, nodes], waveIndex) => {
      maxRows = Math.max(maxRows, nodes.length);
      nodes.forEach((node, rowIndex) => {
        const x = PAD + waveIndex * (NODE_W + GAP_X);
        const y = PAD + rowIndex * (NODE_H + GAP_Y);
        positions.set(node.id, { x, y });
      });
    });

    const width = PAD * 2 + waves.length * NODE_W + Math.max(0, waves.length - 1) * GAP_X;
    const height = PAD * 2 + maxRows * NODE_H + Math.max(0, maxRows - 1) * GAP_Y;
    return { positions, width, height, waveCount: waves.length };
  }, [dag.nodes]);

  const critical = useMemo(() => new Set(dag.critical_path), [dag.critical_path]);

  if (dag.nodes.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-slate-500">
        No tasks yet. Start an orchestration to build the execution graph.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="canvas-grid overflow-x-auto rounded-lg border border-slate-800">
        <svg
          width={layout.width}
          height={layout.height}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          className="min-w-full"
          role="img"
          aria-label="Task dependency graph"
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" opacity="0.7" />
            </marker>
          </defs>

          {dag.edges.map((edge) => {
            const from = layout.positions.get(edge.from);
            const to = layout.positions.get(edge.to);
            if (!from || !to) return null;
            const source = dag.nodes.find((n) => n.id === edge.from);
            const x1 = from.x + NODE_W;
            const y1 = from.y + NODE_H / 2;
            const x2 = to.x;
            const y2 = to.y + NODE_H / 2;
            const mid = (x1 + x2) / 2;
            const active =
              hovered === edge.from || hovered === edge.to || source?.status === "RUNNING";
            return (
              <path
                key={`${edge.from}-${edge.to}`}
                d={`M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`}
                fill="none"
                strokeWidth={active ? 2 : 1.25}
                markerEnd="url(#arrow)"
                className={cx(
                  EDGE_CLASS[source?.status ?? "PENDING"],
                  active && "edge-flow text-sky-400",
                )}
                opacity={hovered && hovered !== edge.from && hovered !== edge.to ? 0.25 : 1}
              />
            );
          })}

          {dag.nodes.map((node) => {
            const pos = layout.positions.get(node.id);
            if (!pos) return null;
            const isCritical = critical.has(node.id);
            const isHovered = hovered === node.id;
            return (
              <g
                key={node.id}
                transform={`translate(${pos.x}, ${pos.y})`}
                className="cursor-pointer"
                onMouseEnter={() => setHovered(node.id)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => onSelect?.(node)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && onSelect?.(node)}
              >
                <rect
                  width={NODE_W}
                  height={NODE_H}
                  rx={10}
                  className={cx(
                    "transition",
                    NODE_TONE[node.status],
                    node.status === "RUNNING" && "running-ring",
                    isHovered && "brightness-125",
                    critical.has(node.id) && !isHovered ? "ring-1 ring-amber-400/40" : "",
                  )}
                  strokeWidth={1.5}
                />
                <text
                  x={12}
                  y={22}
                  className="fill-slate-100 text-[12px] font-semibold"
                  style={{ fontSize: 12 }}
                >
                  {truncate(node.title, 26)}
                </text>
                <text
                  x={12}
                  y={40}
                  className="fill-slate-400"
                  style={{ fontSize: 10.5 }}
                >
                  {titleCase(node.type)}
                  {node.agent ? ` · ${node.agent}` : ""}
                </text>
                <text x={12} y={58} className="fill-slate-500" style={{ fontSize: 10 }}>
                  {node.status}
                  {node.duration_seconds > 0 ? ` · ${formatDuration(node.duration_seconds)}` : ""}
                  {node.retry_count > 0 ? ` · retry ${node.retry_count}` : ""}
                </text>
                {isCritical && (
                  <text
                    x={NODE_W - 12}
                    y={22}
                    textAnchor="end"
                    className="fill-amber-300"
                    style={{ fontSize: 10 }}
                  >
                    critical
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <GraphLegend dag={dag} />
    </div>
  );
}

function GraphLegend({ dag }: { dag: Dag }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-slate-400">
      <span>
        <strong className="text-slate-200">{dag.total_tasks}</strong> tasks
      </span>
      <span>
        <strong className="text-slate-200">{dag.waves.length}</strong> waves
      </span>
      <span>
        peak parallelism{" "}
        <strong className="text-slate-200">{dag.max_parallelism}</strong>
      </span>
      <span>
        critical path{" "}
        <strong className="text-slate-200">{dag.critical_path.length}</strong> tasks
      </span>
      <span className="flex items-center gap-2">
        {Object.entries(dag.status_counts).map(([status, count]) => (
          <span key={status} className="flex items-center gap-1">
            <StatusBadge status={status} />
            {count}
          </span>
        ))}
      </span>
    </div>
  );
}

export function NodeDetail({ node }: { node: DagNode }) {
  return (
    <div className="space-y-3 rounded-lg border border-slate-700 bg-slate-950/70 p-4 text-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-semibold text-slate-100">{node.title}</p>
          <p className="text-xs text-slate-500 mono">{node.id}</p>
        </div>
        <StatusBadge status={node.status} pulse />
      </div>
      <dl className="grid grid-cols-2 gap-2 text-xs">
        <Field label="Type" value={titleCase(node.type)} />
        <Field label="Agent" value={node.agent ?? "-"} />
        <Field label="Wave" value={String(node.wave)} />
        <Field label="Duration" value={formatDuration(node.duration_seconds)} />
        <Field label="Retries" value={String(node.retry_count)} />
        <Field label="Commit" value={node.commit ? node.commit.slice(0, 10) : "-"} />
      </dl>
      {node.blocked_reason && (
        <p className="rounded bg-amber-500/10 px-2 py-1 text-xs text-amber-200">
          {node.blocked_reason}
        </p>
      )}
      {node.errors.length > 0 && (
        <ul className="space-y-1 text-xs text-red-300">
          {node.errors.map((error, i) => (
            <li key={i} className="rounded bg-red-500/10 px-2 py-1">
              {error}
            </li>
          ))}
        </ul>
      )}
      {node.files_changed.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-semibold text-slate-300">
            Files changed ({node.files_changed.length})
          </p>
          <ul className="max-h-40 space-y-0.5 overflow-auto text-xs text-slate-400 mono">
            {node.files_changed.map((file) => (
              <li key={file} className="truncate">
                {file}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-slate-200">{value}</dd>
    </div>
  );
}

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}
