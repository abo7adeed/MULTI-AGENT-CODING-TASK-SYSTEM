import { useEffect, useState } from "react";
import { api, type OrchestrationSummary, type Project } from "../api/client";
import { EmptyState, ErrorBanner, PageHeader, Panel, Spinner } from "../components/Layout";
import { StatusBadge } from "../components/StatusBadge";
import { BTN_GHOST, FINAL_CLASS, formatWhen } from "../lib/ui";

interface Props {
  onNavigate: (page: string, id?: string) => void;
}

export function DashboardPage({ onNavigate }: Props) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [runs, setRuns] = useState<OrchestrationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    Promise.all([api.listProjects(), api.listOrchestrations()])
      .then(([p, r]) => {
        setProjects(p);
        setRuns(r);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const active = runs.filter((r) => r.status === "RUNNING" || r.status === "PAUSED").length;
  const succeeded = runs.filter((r) => r.final_status === "SUCCESS").length;
  const failed = runs.filter((r) => r.final_status === "FAILED").length;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Dashboard"
        subtitle="Projects and orchestration runs"
        actions={
          <button className={BTN_GHOST} onClick={load} disabled={loading}>
            Refresh
          </button>
        }
      />

      <ErrorBanner error={error ?? ""} onDismiss={() => setError(null)} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Projects" value={projects.length} tone="text-slate-100" />
        <Stat label="Total runs" value={runs.length} tone="text-slate-100" />
        <Stat label="Active" value={active} tone="text-violet-300" />
        <Stat
          label="Succeeded / Failed"
          value={`${succeeded} / ${failed}`}
          tone={failed > 0 ? "text-amber-300" : "text-emerald-300"}
        />
      </div>

      <Panel title="Projects">
        {loading ? (
          <Spinner label="Loading projects…" />
        ) : projects.length === 0 ? (
          <EmptyState
            title="No projects yet"
            hint="Create one from the New Task page to point the orchestrator at a repository."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-[11px] uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-3 font-semibold">Name</th>
                  <th className="py-2 pr-3 font-semibold">Path</th>
                  <th className="py-2 pr-3 font-semibold">Branch</th>
                  <th className="py-2 pr-3 text-right font-semibold">Runs</th>
                  <th className="py-2 font-semibold" />
                </tr>
              </thead>
              <tbody>
                {projects.map((project) => (
                  <tr
                    key={project.id}
                    className="border-b border-slate-800/60 hover:bg-slate-800/30"
                  >
                    <td className="py-2.5 pr-3 font-medium text-slate-200">{project.name}</td>
                    <td className="mono max-w-xs truncate py-2.5 pr-3 text-xs text-slate-500">
                      {project.local_path}
                    </td>
                    <td className="py-2.5 pr-3 text-xs text-slate-400">{project.base_branch}</td>
                    <td className="py-2.5 pr-3 text-right tabular-nums text-slate-300">
                      {project.orchestration_count}
                      {project.running_count > 0 && (
                        <span className="ml-1 text-violet-400">({project.running_count} live)</span>
                      )}
                    </td>
                    <td className="py-2.5 text-right">
                      <button
                        className="rounded border border-slate-700 px-2 py-1 text-[11px] text-slate-300 hover:border-sky-500 hover:text-sky-300"
                        onClick={() => onNavigate("new-task", project.id)}
                      >
                        New task
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel title="Recent runs">
        {runs.length === 0 ? (
          <EmptyState title="No orchestrations yet" hint="Start one to see the execution graph." />
        ) : (
          <div className="space-y-2">
            {runs.slice(0, 12).map((run) => (
              <button
                key={run.id}
                onClick={() => onNavigate("orchestration", run.id)}
                className="flex w-full items-center gap-4 rounded-lg border border-slate-800 px-3 py-2.5 text-left transition hover:border-sky-500/50 hover:bg-slate-800/40"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-slate-200">{run.name}</p>
                  <p className="text-[11px] text-slate-500">
                    {run.completed_tasks}/{run.total_tasks} tasks ·{" "}
                    {formatWhen(run.created_at)}
                  </p>
                </div>
                <div className="hidden w-32 sm:block">
                  <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-sky-500 transition-all"
                      style={{ width: `${Math.round(run.progress * 100)}%` }}
                    />
                  </div>
                </div>
                <span className="text-xs tabular-nums text-slate-500">
                  {Math.round(run.progress * 100)}%
                </span>
                <StatusBadge status={run.status} pulse />
                <span className={`text-xs ${FINAL_CLASS[run.final_status]}`}>
                  {run.final_status.replace(/_/g, " ").toLowerCase()}
                </span>
              </button>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string | number; tone: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3">
      <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-1 text-2xl font-bold tabular-nums ${tone}`}>{value}</p>
    </div>
  );
}
