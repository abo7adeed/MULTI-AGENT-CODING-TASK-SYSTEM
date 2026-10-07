import { useCallback, useEffect, useState } from "react";
import { api, type Health } from "./api/client";
import { Layout } from "./components/Layout";
import { AgentsPage, SettingsPage } from "./pages/Agents";
import { DashboardPage } from "./pages/Dashboard";
import { ExecutionPage } from "./pages/Execution";
import { ChangesPage, LogsPage, ReportPage, TestsPage } from "./pages/Inspect";
import { NewTaskPage } from "./pages/NewTask";

type Page =
  | "dashboard"
  | "new-task"
  | "orchestration"
  | "agents"
  | "logs"
  | "changes"
  | "tests"
  | "report"
  | "settings";

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [orchestrationId, setOrchestrationId] = useState<string | null>(
    () => localStorage.getItem("lastOrchestration"),
  );
  const [presetProjectId, setPresetProjectId] = useState<string | undefined>();
  const [health, setHealth] = useState<Health | null>(null);
  const [backendUp, setBackendUp] = useState(true);

  // Health poll: also the heartbeat that tells us the API is reachable.
  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      api
        .health()
        .then((h) => {
          if (cancelled) return;
          setHealth(h);
          setBackendUp(true);
        })
        .catch(() => !cancelled && setBackendUp(false));
    };
    tick();
    const timer = setInterval(tick, 10_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const navigate = useCallback((next: string, id?: string) => {
    if (id) {
      setOrchestrationId(id);
      localStorage.setItem("lastOrchestration", id);
    }
    if (next === "new-task") {
      setPresetProjectId(id);
      setPage("new-task");
      return;
    }
    setPage(next as Page);
  }, []);

  const started = useCallback((id: string) => {
    setOrchestrationId(id);
    localStorage.setItem("lastOrchestration", id);
    setPage("orchestration");
  }, []);

  return (
    <Layout
      page={page}
      onNavigate={navigate}
      hasRun={!!orchestrationId}
      health={health}
      backendUp={backendUp}
      live={false}
    >
      {!backendUp && (
        <div className="mb-4 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          Cannot reach the API at <code className="mono">{api.baseUrl}</code>. Start the backend with{" "}
          <code className="mono">uvicorn app.api.main:app --reload</code>.
        </div>
      )}

      {page === "dashboard" && <DashboardPage onNavigate={navigate} />}
      {page === "new-task" && (
        <NewTaskPage
          presetProjectId={presetProjectId}
          onStarted={started}
          onNavigate={navigate}
        />
      )}
      {page === "orchestration" && (
        <ExecutionPage
          orchestrationId={orchestrationId}
          health={health}
          onNavigate={navigate}
        />
      )}
      {page === "agents" && (
        <AgentsPage orchestrationId={orchestrationId} health={health} onNavigate={navigate} />
      )}
      {page === "logs" && <LogsPage orchestrationId={orchestrationId} health={health} />}
      {page === "changes" && <ChangesPage orchestrationId={orchestrationId} health={health} />}
      {page === "tests" && <TestsPage orchestrationId={orchestrationId} health={health} />}
      {page === "report" && <ReportPage orchestrationId={orchestrationId} health={health} />}
      {page === "settings" && <SettingsPage />}
    </Layout>
  );
}
