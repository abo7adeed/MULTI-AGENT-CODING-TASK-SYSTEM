import type { AgentActivity } from "../api/client";
import { cx, formatDuration, shortId } from "../lib/ui";
import { StatusBadge } from "./StatusBadge";

interface Props {
  agents: AgentActivity[];
  onRetry?: (taskId: string) => void;
  busyTaskId?: string | null;
}

const ROLE_LABEL: Record<string, string> = {
  repository_analyst: "Repo Analyst",
  planner: "Planner",
  architect: "Architect",
  backend: "Backend",
  frontend: "Frontend",
  database: "Database",
  ai_ml: "AI / ML",
  testing: "Testing",
  devops: "DevOps",
  debugging: "Debugging",
  code_review: "Review",
  documentation: "Docs",
  refactor: "Refactor",
  security: "Security",
  integration: "Integration",
  generic: "General",
};

export function AgentTable({ agents, onRetry, busyTaskId }: Props) {
  if (agents.length === 0) {
    return (
      <p className="rounded-lg border border-slate-800 px-4 py-8 text-center text-sm text-slate-500">
        No agent activity yet.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-slate-800 text-[11px] uppercase tracking-wide text-slate-500">
            <th className="py-2 pr-3 font-semibold">Agent</th>
            <th className="py-2 pr-3 font-semibold">Task</th>
            <th className="py-2 pr-3 font-semibold">Status</th>
            <th className="py-2 pr-3 text-right font-semibold">Files</th>
            <th className="py-2 pr-3 text-right font-semibold">Tests</th>
            <th className="py-2 pr-3 text-right font-semibold">Time</th>
            <th className="py-2 font-semibold" />
          </tr>
        </thead>
        <tbody>
          {agents.map((agent) => (
            <tr
              key={agent.task_id}
              className="border-b border-slate-800/60 transition hover:bg-slate-800/30"
            >
              <td className="py-2.5 pr-3">
                <div className="font-medium text-slate-200">
                  {ROLE_LABEL[agent.role ?? "generic"] ?? agent.role ?? "Unassigned"}
                </div>
                <div className="text-[11px] text-slate-500 mono">{shortId(agent.task_id, 6)}</div>
              </td>
              <td className="max-w-xs py-2.5 pr-3">
                <div className="truncate text-slate-200" title={agent.title}>
                  {agent.title}
                </div>
                {agent.summary && (
                  <div className="truncate text-[11px] text-slate-500" title={agent.summary}>
                    {agent.summary}
                  </div>
                )}
              </td>
              <td className="py-2.5 pr-3">
                <StatusBadge status={agent.status} pulse />
                {agent.attempt > 1 && (
                  <span className="ml-1.5 text-[11px] text-amber-400">
                    attempt {agent.attempt}
                  </span>
                )}
              </td>
              <td className="py-2.5 pr-3 text-right tabular-nums text-slate-300">
                {agent.files_changed.length || "-"}
              </td>
              <td className="py-2.5 pr-3 text-right text-xs tabular-nums">
                {agent.tests_passed + agent.tests_failed > 0 ? (
                  <>
                    <span className="text-emerald-400">{agent.tests_passed}</span>
                    <span className="text-slate-600">/</span>
                    <span className="text-red-400">{agent.tests_failed}</span>
                  </>
                ) : (
                  <span className="text-slate-600">-</span>
                )}
              </td>
              <td className="py-2.5 pr-3 text-right tabular-nums text-slate-400">
                {formatDuration(agent.duration_seconds)}
              </td>
              <td className="py-2.5 text-right">
                {onRetry && (agent.status === "FAILED" || agent.status === "BLOCKED") && (
                  <button
                    className={cx(
                      "rounded border border-slate-700 px-2 py-1 text-[11px] text-slate-300",
                      "transition hover:border-sky-500 hover:text-sky-300",
                      "disabled:cursor-not-allowed disabled:opacity-40",
                    )}
                    disabled={busyTaskId === agent.task_id}
                    onClick={() => onRetry(agent.task_id)}
                  >
                    {busyTaskId === agent.task_id ? "Retrying…" : "Retry"}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
