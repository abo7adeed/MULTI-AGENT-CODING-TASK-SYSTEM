import { useEffect, useMemo, useState } from "react";
import { api, type GitChange, type Health, type LogEntry, type TestRun } from "../api/client";
import { EmptyState, ErrorBanner, PageHeader, Panel, Spinner } from "../components/Layout";
import { StatusBadge } from "../components/StatusBadge";
import { useLiveRun } from "../lib/useLiveRun";
import { BTN_GHOST, formatDuration, formatTime, shortId, toneFor } from "../lib/ui";

interface Props {
  orchestrationId: string | null;
  health: Health | null;
}

// ── logs ────────────────────────────────────────────────────────────────────

const LOG_TONE: Record<string, string> = {
  "task.failed": "text-red-300",
  "task.blocked": "text-amber-300",
  "task.retrying": "text-amber-300",
  "task.succeeded": "text-emerald-300",
  "task.started": "text-sky-300",
  "agent.started": "text-violet-300",
  "agent.finished": "text-violet-200",
  "orchestration.finished": "text-emerald-200",
  "orchestration.started": "text-emerald-200",
  "orchestration.paused": "text-amber-300",
  "orchestration.cancelled": "text-slate-400",
  "integration.started": "text-sky-300",
  "integration.finished": "text-sky-200",
  "conflict.detected": "text-amber-300",
  "conflict.resolved": "text-emerald-300",
  "tests.finished": "text-sky-200",
};

export function LogsPage({ orchestrationId, health }: Props) {
  const run = useLiveRun(orchestrationId, health);
  const [filter, setFilter] = useState("");

  const events = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return run.events;
    return run.events.filter(
      (e) =>
        e.type.toLowerCase().includes(needle) ||
        JSON.stringify(e.data).toLowerCase().includes(needle),
    );
  }, [run.events, filter]);

  if (!orchestrationId) return <EmptyState title="No orchestration selected" />;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Event log"
        subtitle={`${run.events.length} events`}
        actions={
          <input
            className="w-48 rounded-lg border border-slate-700 bg-slate-950/70 px-3 py-1.5 text-sm outline-none focus:border-sky-500"
            placeholder="Filter events…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        }
      />
      <ErrorBanner error={run.error ?? ""} onDismiss={run.clearError} />
      <Panel>
        {events.length === 0 ? (
          <EmptyState title="No events yet" hint="Events appear as the run progresses." />
        ) : (
          <div className="max-h-[70vh] space-y-0.5 overflow-auto">
            {events.map((event) => (
              <LogRow key={`${event.sequence}-${event.type}`} event={event} />
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

function LogRow({ event }: { event: LogEntry }) {
  const tone = LOG_TONE[event.type] ?? "text-slate-400";
  const summary = summarise(event);
  return (
    <div className="flex gap-3 rounded px-2 py-1 text-xs hover:bg-slate-800/40">
      <span className="shrink-0 tabular-nums text-slate-600">{formatTime(event.timestamp)}</span>
      <span className={`w-48 shrink-0 truncate font-medium ${tone}`}>{event.type}</span>
      <span className="min-w-0 flex-1 truncate text-slate-400">{summary}</span>
    </div>
  );
}

function summarise(event: LogEntry): string {
  const d = event.data;
  const bits: string[] = [];
  for (const [key, value] of Object.entries(d)) {
    if (key === "orchestration_id" || value === null || value === undefined) continue;
    if (Array.isArray(value)) {
      if (value.length) bits.push(`${key}=${value.slice(0, 3).join(", ")}${value.length > 3 ? "…" : ""}`);
    } else if (typeof value === "object") {
      bits.push(`${key}=${JSON.stringify(value).slice(0, 80)}`);
    } else {
      bits.push(`${key}=${String(value).slice(0, 80)}`);
    }
  }
  return bits.join("  ");
}

// ── git changes ─────────────────────────────────────────────────────────────

export function ChangesPage({ orchestrationId }: Props) {
  const [changes, setChanges] = useState<GitChange[]>([]);
  const [diff, setDiff] = useState<Awaited<ReturnType<typeof api.diff>> | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!orchestrationId) return;
    setLoading(true);
    Promise.all([api.changes(orchestrationId), api.diff(orchestrationId)])
      .then(([c, d]) => {
        setChanges(c);
        setDiff(d);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load changes"))
      .finally(() => setLoading(false));
  }, [orchestrationId]);

  useEffect(() => {
    if (!orchestrationId || !selected) return;
    api
      .diff(orchestrationId, selected)
      .then(setDiff)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load diff"));
  }, [orchestrationId, selected]);

  if (!orchestrationId) return <EmptyState title="No orchestration selected" />;

  return (
    <div className="space-y-5">
      <PageHeader title="Git changes" subtitle="What each agent contributed" />
      <ErrorBanner error={error ?? ""} onDismiss={() => setError(null)} />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,360px)_1fr]">
        <Panel title="Agent changes">
          {loading ? (
            <Spinner />
          ) : changes.length === 0 ? (
            <EmptyState title="No changes recorded" />
          ) : (
            <div className="space-y-1.5">
              {changes.map((change) => (
                <button
                  key={change.task_id}
                  onClick={() => setSelected(change.branch ?? undefined as unknown as string)}
                  className={`w-full rounded-lg border px-3 py-2 text-left text-sm transition ${
                    selected === change.branch
                      ? "border-sky-500 bg-sky-500/10"
                      : "border-slate-800 hover:border-slate-600"
                  }`}
                >
                  <p className="truncate font-medium text-slate-200">{change.title}</p>
                  <p className="mono mt-0.5 truncate text-[11px] text-slate-500">
                    {change.branch ?? "no branch"}
                    {change.commit ? ` · ${change.commit.slice(0, 8)}` : ""}
                  </p>
                  <p className="mt-0.5 text-[11px] text-slate-500">
                    {change.files_changed.length} file(s) changed
                  </p>
                </button>
              ))}
            </div>
          )}
        </Panel>

        <Panel
          title={selected ? `Diff: ${selected}` : "Integrated diff"}
          actions={
            diff?.insertions !== undefined && (
              <span className="text-[11px]">
                <span className="text-emerald-400">+{diff.insertions}</span>{" "}
                <span className="text-red-400">-{diff.deletions}</span>
              </span>
            )
          }
        >
          {diff ? (
            <div className="space-y-3">
              {diff.files && diff.files.length > 0 && (
                <div>
                  <p className="mb-1 text-[11px] uppercase tracking-wide text-slate-500">
                    Changed files
                  </p>
                  <ul className="mono max-h-40 space-y-0.5 overflow-auto text-xs">
                    {diff.files.map((f) => (
                      <li key={f.path} className="flex items-center gap-2">
                        <span
                          className={
                            f.unmerged
                              ? "text-amber-400"
                              : f.status === "??"
                                ? "text-emerald-400"
                                : "text-slate-400"
                          }
                        >
                          {f.status}
                        </span>
                        <span className="truncate text-slate-300">{f.path}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {diff.patch ? (
                <pre className="mono max-h-[60vh] overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-300">
                  {diff.patch}
                </pre>
              ) : (
                <p className="text-sm text-slate-500">
                  No textual diff. The change may be binary, or already merged.
                </p>
              )}
            </div>
          ) : (
            <Spinner />
          )}
        </Panel>
      </div>
    </div>
  );
}

// ── test results ────────────────────────────────────────────────────────────

export function TestsPage({ orchestrationId, health }: Props) {
  const run = useLiveRun(orchestrationId, health);
  const [results, setResults] = useState<{
    summary: Record<string, unknown>;
    runs: TestRun[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!orchestrationId) return;
    api
      .testResults(orchestrationId)
      .then(setResults)
      .catch((e) => setError(e instanceof Error ? e.message : "No test results yet"));
  }, [orchestrationId, run.orchestration?.updated_at]);

  if (!orchestrationId) return <EmptyState title="No orchestration selected" />;

  return (
    <div className="space-y-5">
      <PageHeader title="Test results" subtitle="Baseline versus after integration" />
      <ErrorBanner error={error ?? ""} onDismiss={() => setError(null)} />

      {!results || results.runs.length === 0 ? (
        <EmptyState
          title="No test run recorded"
          hint="Tests run automatically during integration when the project has a detectable test command."
        />
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            {results.runs.map((testRun, i) => (
              <div key={i} className="rounded-lg border border-slate-800 bg-slate-900/50 p-3">
                <p className="text-[11px] uppercase tracking-wide text-slate-500">
                  {i === 0 ? "Before merge" : "After merge"}
                </p>
                <div className="mt-1 flex items-center gap-2">
                  <StatusBadge status={testRun.passed ? "SUCCESS" : "FAILED"} />
                  {testRun.skipped && <span className="text-xs text-slate-500">skipped</span>}
                </div>
                <p className="mt-1.5 text-xs text-slate-400">
                  {testRun.passed_count} passed · {testRun.failed_count} failed ·{" "}
                  {formatDuration(testRun.duration_seconds)}
                </p>
              </div>
            ))}
          </div>

          {results.runs.map((testRun, i) => (
            <Panel key={i} title={testRun.command || "(no command)"}>
              {testRun.skipped ? (
                <p className="text-sm text-slate-400">{testRun.stdout}</p>
              ) : (
                <pre className="mono max-h-96 overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-300">
                  {(testRun.stdout || testRun.stderr || "(no output)").slice(-8000)}
                </pre>
              )}
            </Panel>
          ))}
        </>
      )}
    </div>
  );
}

// ── final report ────────────────────────────────────────────────────────────

export function ReportPage({ orchestrationId, health }: Props) {
  const run = useLiveRun(orchestrationId, health);
  const [report, setReport] = useState<Awaited<ReturnType<typeof api.report>> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!orchestrationId) return;
    api
      .report(orchestrationId)
      .then(setReport)
      .catch((e) => setError(e instanceof Error ? e.message : "No report yet"));
  }, [orchestrationId, run.orchestration?.updated_at]);

  if (!orchestrationId) return <EmptyState title="No orchestration selected" />;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Integration report"
        actions={
          report && (
            <button
              className={BTN_GHOST}
              onClick={() => navigator.clipboard?.writeText(report.markdown)}
            >
              Copy markdown
            </button>
          )
        }
      />
      <ErrorBanner error={error ?? run.error ?? ""} onDismiss={() => setError(null)} />

      {report ? (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <StatusBadge status={report.success ? "SUCCESS" : "FAILED"} />
            <span className="text-sm text-slate-400">
              {report.merged_branches.length} branch(es) merged
            </span>
            {report.conflicts.length > 0 && (
              <span className="text-sm text-amber-300">{report.conflicts.length} conflict(s)</span>
            )}
            {report.regressions.length > 0 && (
              <span className="text-sm text-red-300">{report.regressions.length} regression(s)</span>
            )}
          </div>
          {report.regressions.length > 0 && (
            <Panel title="Regressions">
              <ul className="space-y-1 text-sm text-red-300">
                {report.regressions.map((r, i) => (
                  <li key={i}>· {r}</li>
                ))}
              </ul>
            </Panel>
          )}
          {report.review_findings.length > 0 && (
            <Panel title="Review findings">
              <ul className="space-y-2 text-sm">
                {report.review_findings.map((f, i) => (
                  <li key={i} className="rounded bg-slate-950/60 p-2">
                    <span
                      className={`mr-2 text-[11px] font-semibold uppercase ${
                        f.severity === "critical"
                          ? "text-red-400"
                          : f.severity === "major"
                            ? "text-amber-400"
                            : "text-sky-400"
                      }`}
                    >
                      {String(f.severity ?? "info")}
                    </span>
                    <span className="text-slate-200">{String(f.issue ?? "")}</span>
                  </li>
                ))}
              </ul>
            </Panel>
          )}
          <Panel title="Report">
            <pre className="mono max-h-[70vh] overflow-auto whitespace-pre-wrap text-xs leading-relaxed text-slate-300">
              {report.markdown}
            </pre>
          </Panel>
        </>
      ) : (
        <EmptyState
          title="No report yet"
          hint="The report is written once integration finishes. If the run produced no mergeable branches, the report explains why."
        />
      )}
    </div>
  );
}

export { shortId, toneFor };
