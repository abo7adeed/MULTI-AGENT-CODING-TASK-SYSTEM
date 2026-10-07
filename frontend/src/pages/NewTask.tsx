import { useEffect, useState } from "react";
import { api, type Project } from "../api/client";
import { ErrorBanner, PageHeader, Panel } from "../components/Layout";
import { BTN_GHOST, BTN_PRIMARY, INPUT } from "../lib/ui";

interface Props {
  presetProjectId?: string;
  onStarted: (orchestrationId: string) => void;
  onNavigate?: (page: string) => void;
}

const PRESETS: { label: string; request: string }[] = [
  {
    label: "Full-stack RAG app",
    request:
      "Build a FastAPI RAG application with authentication, vector search, tests, Docker and a React frontend.",
  },
  {
    label: "API + database",
    request:
      "Build a REST API with PostgreSQL, JWT authentication, reversible migrations, and a CI pipeline.",
  },
  {
    label: "Small backend change",
    request: "Add rate limiting to the existing API and cover it with tests.",
  },
  {
    label: "Docs only",
    request: "Document the authentication module and add runnable examples to the README.",
  },
];

/**
 * What the Repository panel is pointed at.
 *
 * `undecided` exists so that "no choice yet" can never be confused with "the
 * user asked for a new project": an empty selection is a real answer, and
 * treating it as a missing one is what silently re-selected an existing
 * project and made the "Create a new project" option unclickable.
 */
type Target =
  | { kind: "undecided" }
  | { kind: "new" }
  | { kind: "existing"; id: string };

export function NewTaskPage({ presetProjectId, onStarted, onNavigate }: Props) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [target, setTarget] = useState<Target>(
    presetProjectId ? { kind: "existing", id: presetProjectId } : { kind: "undecided" },
  );
  const [projectName, setProjectName] = useState("");
  const [localPath, setLocalPath] = useState("");
  const [request, setRequest] = useState(PRESETS[0].request);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sysInfo, setSysInfo] = useState<Awaited<ReturnType<typeof api.systemInfo>> | null>(null);

  useEffect(() => {
    api.systemInfo().then(setSysInfo).catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .listProjects()
      .then((list) => {
        if (cancelled) return;
        setProjects(list);
        // Default to the most recent project, but only for someone who has not
        // asked for anything yet -- and only once. Re-deciding later would
        // overrule the user, and the user is the one who knows.
        setTarget((current) =>
          current.kind === "undecided"
            ? list.length > 0
              ? { kind: "existing", id: list[0].id }
              : { kind: "new" }
            : current,
        );
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "Failed to load projects"));
    return () => {
      cancelled = true;
    };
  }, []);

  const needsNewProject = target.kind === "new";

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      let targetId = target.kind === "existing" ? target.id : "";
      if (needsNewProject) {
        if (!localPath.trim()) throw new Error("A repository path is required.");
        const project = await api.createProject({
          name: projectName.trim() || "Untitled project",
          description: request.slice(0, 200),
          local_path: localPath.trim(),
        });
        targetId = project.id;
      }
      const orchestration = await api.startOrchestration({
        project_id: targetId,
        user_request: request.trim(),
      });
      onStarted(orchestration.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start orchestration");
    } finally {
      setBusy(false);
    }
  };

  const canSubmit =
    request.trim().length >= 3 &&
    !busy &&
    target.kind !== "undecided" &&
    (!needsNewProject || !!localPath.trim());

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <PageHeader
        title="New Coding Task"
        subtitle="Describe the work. The orchestrator will inspect the repository, plan a DAG and dispatch agents."
      />

      <ErrorBanner error={error ?? ""} onDismiss={() => setError(null)} />

      {/* AI Model Status Banner */}
      {sysInfo && (
        sysInfo.llm.active_provider === "mock" ? (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-xl border border-amber-500/40 bg-amber-500/10 p-3.5 text-xs text-amber-200">
            <div>
              <span className="font-semibold">⚠️ Running in Mock Mode (Static Canned Data).</span>
              <p className="mt-0.5 text-amber-300/80">Agents will use pre-set demo stubs rather than a live LLM model.</p>
            </div>
            {onNavigate && (
              <button
                type="button"
                onClick={() => onNavigate("settings")}
                className="shrink-0 rounded-lg bg-amber-500/20 px-3 py-1.5 font-medium text-amber-100 hover:bg-amber-500/30 transition border border-amber-500/30"
              >
                Connect Gemini or OpenAI →
              </button>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-2.5 text-xs text-slate-300">
            <div className="flex items-center gap-2.5">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-slate-400">Live AI Provider:</span>
              <span className="font-semibold text-slate-100 capitalize">{sysInfo.llm.active_provider}</span>
              <span className="mono rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-sky-300">
                {sysInfo.llm.active_model}
              </span>
            </div>
            {onNavigate && (
              <button
                type="button"
                onClick={() => onNavigate("settings")}
                className="text-[11px] text-sky-400 hover:text-sky-300 hover:underline"
              >
                Change Provider
              </button>
            )}
          </div>
        )
      )}

      <Panel title="Repository">
        <div className="space-y-4">
          {projects.length > 0 && (
            <div>
              <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                Existing project
              </label>
              <div className="space-y-1.5">
                <label className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-slate-800 px-3 py-2 text-sm">
                  <input
                    type="radio"
                    checked={needsNewProject}
                    onChange={() => setTarget({ kind: "new" })}
                    className="accent-sky-500"
                  />
                  <span className="text-slate-300">Create a new project</span>
                </label>
                {projects.map((project) => (
                  <label
                    key={project.id}
                    className="flex cursor-pointer items-start gap-2.5 rounded-lg border border-slate-800 px-3 py-2 text-sm transition hover:border-slate-600"
                  >
                    <input
                      type="radio"
                      checked={target.kind === "existing" && target.id === project.id}
                      onChange={() => setTarget({ kind: "existing", id: project.id })}
                      className="mt-1 accent-sky-500"
                    />
                    <span className="min-w-0">
                      <span className="block font-medium text-slate-200">{project.name}</span>
                      <span className="mono block truncate text-[11px] text-slate-500">
                        {project.local_path}
                      </span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {needsNewProject && (
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Project name
                </label>
                <input
                  className={INPUT}
                  placeholder="my-service"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                />
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Repository path
                </label>
                <input
                  className={`${INPUT} mono`}
                  placeholder="/path/to/repo"
                  value={localPath}
                  onChange={(e) => setLocalPath(e.target.value)}
                />
                <p className="mt-1 text-[11px] text-slate-500">
                  Initialised as a git repository if it is not one already.
                </p>
              </div>
            </div>
          )}
        </div>
      </Panel>

      <Panel title="Coding task">
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => setRequest(preset.request)}
                className="rounded-full border border-slate-700 px-2.5 py-1 text-[11px] text-slate-400 transition hover:border-sky-500 hover:text-sky-300"
              >
                {preset.label}
              </button>
            ))}
          </div>
          <textarea
            className={`${INPUT} min-h-[140px] resize-y font-normal leading-relaxed`}
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            placeholder="Describe the feature, fix or refactor you want…"
          />
          <p className="text-[11px] text-slate-500">
            Be specific about deliverables (API, UI, tests, Docker, CI). More detail produces a
            better decomposition.
          </p>
        </div>
      </Panel>

      <div className="flex items-center gap-3">
        <button className={BTN_PRIMARY} onClick={submit} disabled={!canSubmit}>
          {busy ? "Planning…" : "Start orchestration"}
        </button>
        <button className={BTN_GHOST} onClick={() => setRequest("")} disabled={busy}>
          Clear
        </button>
        {busy && (
          <span className="text-xs text-slate-500">
            Analysing the repository and building the task graph…
          </span>
        )}
      </div>
    </div>
  );
}
