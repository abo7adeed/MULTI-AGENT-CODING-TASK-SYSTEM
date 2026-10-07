import { useState } from "react";
import { api, type DagNode, type Health } from "../api/client";
import { AgentTable } from "../components/AgentTable";
import { DAGView, NodeDetail } from "../components/DAGView";
import { EmptyState, ErrorBanner, PageHeader, Panel, Spinner } from "../components/Layout";
import { StatusBadge } from "../components/StatusBadge";
import { useLiveRun } from "../lib/useLiveRun";
import { BTN_GHOST, FINAL_CLASS, formatDuration, titleCase } from "../lib/ui";

interface Props {
  orchestrationId: string | null;
  health: Health | null;
  onNavigate: (page: string, id?: string) => void;
}

export function ExecutionPage({ orchestrationId, health, onNavigate }: Props) {
  const run = useLiveRun(orchestrationId, health);
  const [selected, setSelected] = useState<DagNode | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  if (!orchestrationId) {
    return (
      <EmptyState
        title="No orchestration selected"
        hint="Start one from the New Task page, or pick a run from the dashboard."
      />
    );
  }

  if (run.loading && !run.orchestration) {
    return <Spinner label="Loading execution…" />;
  }

  if (!run.orchestration) {
    return <EmptyState title="Orchestration not found" />;
  }

  const o = run.orchestration;
  const analysis = o.task_analysis as Record<string, unknown>;

  const control = async (action: "pause" | "resume" | "cancel") => {
    setBusy(action);
    setActionError(null);
    try {
      if (action === "pause") await api.pause(orchestrationId);
      if (action === "resume") await api.resume(orchestrationId);
      if (action === "cancel") await api.cancel(orchestrationId);
      await run.refresh();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : `Failed to ${action}`);
    } finally {
      setBusy(null);
    }
  };

  const retry = async (taskId: string) => {
    setBusy(taskId);
    setActionError(null);
    try {
      await api.retryAgent(taskId, orchestrationId);
      await run.refresh();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "Retry failed");
    } finally {
      setBusy(null);
    }
  };

  const running = o.status === "RUNNING";

  return (
    <div className="space-y-5">
      <PageHeader
        title={o.name || "Orchestration"}
        subtitle={o.original_task}
        actions={
          <div className="flex items-center gap-2">
            {run.connected && running && (
              <span className="flex items-center gap-1.5 text-[11px] text-emerald-400">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                live
              </span>
            )}
            {running && o.status !== "PAUSED" && (
              <button
                className={BTN_GHOST}
                onClick={() => control("pause")}
                disabled={busy !== null}
              >
                {busy === "pause" ? "Pausing…" : "Pause"}
              </button>
            )}
            {o.status === "PAUSED" && (
              <button
                className={BTN_GHOST}
                onClick={() => control("resume")}
                disabled={busy !== null}
              >
                {busy === "resume" ? "Resuming…" : "Resume"}
              </button>
            )}
            {(running || o.status === "PAUSED") && (
              <button
                className="rounded-lg border border-red-500/40 px-3 py-1.5 text-sm text-red-300 transition hover:bg-red-500/10"
                onClick={() => control("cancel")}
                disabled={busy !== null}
              >
                {busy === "cancel" ? "Cancelling…" : "Cancel"}
              </button>
            )}
            <button className={BTN_GHOST} onClick={() => run.refresh()}>
              Refresh
            </button>
          </div>
        }
      />

      <ErrorBanner error={run.error ?? actionError ?? ""} onDismiss={() => setActionError(null)} />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Metric label="Status" value={<StatusBadge status={o.status} pulse />} />
        <Metric label="Final" value={<span className={FINAL_CLASS[o.final_status]}>{titleCase(o.final_status)}</span>} />
        <Metric label="Phase" value={titleCase(o.current_phase)} />
        <Metric
          label="Progress"
          value={`${Math.round(o.progress * 100)}% (${o.completed_tasks}/${o.total_tasks})`}
        />
        <Metric label="Duration" value={formatDuration(o.duration_seconds)} />
      </div>

      {o.errors.length > 0 && (
        <Panel title="Errors">
          <ul className="space-y-1 text-sm text-red-300">
            {o.errors.map((error, i) => (
              <li key={i} className="rounded bg-red-500/10 px-3 py-1.5">
                {error}
              </li>
            ))}
          </ul>
        </Panel>
      )}

      <Panel
        title="Execution graph"
        actions={
          run.dag ? (
            <span className="text-[11px] text-slate-500">
              peak parallelism {run.dag.max_parallelism}
            </span>
          ) : null
        }
      >
        {run.dag ? (
          <DAGView dag={run.dag} onSelect={setSelected} />
        ) : (
          <Spinner label="Building the graph…" />
        )}
      </Panel>

      {selected && <NodeDetail node={selected} />}

      <Panel title="Agent activity">
        <AgentTable agents={run.agents} onRetry={retry} busyTaskId={busy} />
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Plan">
          <dl className="space-y-2 text-sm">
            <Row label="Detected types">
              {((analysis.task_types as string[]) ?? []).map(titleCase).join(", ") || "-"}
            </Row>
            <Row label="Deliverables">
              {((analysis.deliverables as string[]) ?? []).join(", ") || "-"}
            </Row>
            <Row label="Complexity">{titleCase(String(analysis.complexity ?? "-"))}</Row>
            <Row label="Greenfield">{analysis.is_greenfield ? "yes" : "no"}</Row>
            {Array.isArray(analysis.ambiguities) && (analysis.ambiguities as string[]).length > 0 && (
              <Row label="Ambiguities">
                <ul className="space-y-0.5 text-xs text-amber-300">
                  {(analysis.ambiguities as string[]).map((a, i) => (
                    <li key={i}>{a}</li>
                  ))}
                </ul>
              </Row>
            )}
            {Array.isArray(analysis.acceptance_criteria) &&
              (analysis.acceptance_criteria as string[]).length > 0 && (
                <Row label="Acceptance">
                  <ul className="space-y-0.5 text-xs text-slate-400">
                    {(analysis.acceptance_criteria as string[]).map((a, i) => (
                      <li key={i}>· {a}</li>
                    ))}
                  </ul>
                </Row>
              )}
          </dl>
        </Panel>

        <Panel
          title="Repository"
          actions={
            <button
              className="text-[11px] text-sky-400 hover:text-sky-300"
              onClick={() => onNavigate("changes", orchestrationId)}
            >
              view changes
            </button>
          }
        >
          <p className="mono mb-3 truncate text-xs text-slate-500">{o.repository}</p>
          <RepositorySummary analysis={o.repository_analysis} />
        </Panel>
      </div>
    </div>
  );
}

function RepositorySummary({ analysis }: { analysis: Record<string, unknown> }) {
  const languages = (analysis.languages as Record<string, number>) ?? {};
  const frameworks = (analysis.frameworks as string[]) ?? [];
  const total = languages ? Object.values(languages).reduce((a, b) => a + b, 0) : 0;
  return (
    <div className="space-y-3 text-sm">
      {total > 0 && (
        <div>
          <p className="mb-1 text-[11px] uppercase tracking-wide text-slate-500">Languages</p>
          <div className="flex h-2 overflow-hidden rounded-full bg-slate-800">
            {Object.entries(languages)
              .sort((a, b) => b[1] - a[1])
              .map(([lang, count], i) => (
                <div
                  key={lang}
                  className={COLORS[i % COLORS.length]}
                  style={{ width: `${(count / total) * 100}%` }}
                  title={`${lang}: ${count}`}
                />
              ))}
          </div>
          <p className="mt-1 text-[11px] text-slate-500">
            {Object.entries(languages)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 4)
              .map(([l, c]) => `${l} ${c}`)
              .join(" · ")}
          </p>
        </div>
      )}
      {frameworks.length > 0 && (
        <div>
          <p className="mb-1 text-[11px] uppercase tracking-wide text-slate-500">Frameworks</p>
          <div className="flex flex-wrap gap-1">
            {frameworks.map((f) => (
              <span
                key={f}
                className="rounded border border-slate-700 px-1.5 py-0.5 text-[11px] text-slate-400"
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
      {typeof analysis.summary === "string" && analysis.summary && (
        <p className="text-xs text-slate-400">{analysis.summary}</p>
      )}
    </div>
  );
}

const COLORS = [
  "bg-sky-500",
  "bg-violet-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-cyan-500",
];

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-0.5 text-sm font-semibold text-slate-100">{value}</p>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <dt className="w-28 shrink-0 text-[11px] uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="min-w-0 flex-1 text-slate-200">{children}</dd>
    </div>
  );
}
