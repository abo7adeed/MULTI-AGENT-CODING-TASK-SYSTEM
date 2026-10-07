// Typed API client. One place that knows the wire format and the base URL.

const BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

// ── wire types ──────────────────────────────────────────────────────────────

export interface Project {
  id: string;
  name: string;
  description: string;
  local_path: string;
  repository_url: string | null;
  base_branch: string;
  created_at: number;
  orchestration_count: number;
  running_count: number;
}

export interface Task {
  id: string;
  title: string;
  description: string;
  type: string;
  status: TaskStatus;
  priority: number;
  dependencies: string[];
  assigned_agent: string | null;
  branch: string | null;
  workspace: string | null;
  retry_count: number;
  max_retries: number;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  duration_seconds: number;
  blocked_reason: string | null;
  errors: string[];
  tests: string[];
  output: Record<string, unknown>;
}

export interface AgentResult {
  task_id: string;
  status: TaskStatus;
  summary: string;
  agent_role: string | null;
  files_changed: string[];
  commit: string | null;
  branch: string | null;
  duration_seconds: number;
  attempt: number;
  errors: string[];
  recommendations: string[];
  tests_passed: number;
  tests_failed: number;
}

export type TestRun = {
  command: string;
  passed: boolean;
  exit_code: number;
  stdout: string;
  stderr: string;
  passed_count: number;
  failed_count: number;
  duration_seconds: number;
  skipped: boolean;
  summary: string;
};

export interface IntegrationReport {
  merged_branches: string[];
  skipped_branches: string[];
  skipped_details: { task_id: string; title: string; skip_reason: string }[];
  conflicts: Record<string, unknown>[];
  conflict_resolutions: Record<string, unknown>[];
  test_runs: TestRun[];
  regressions: string[];
  review_findings: Record<string, unknown>[];
  report: string;
  success: boolean;
}

export interface OrchestrationSummary {
  id: string;
  name: string;
  status: RunStatus;
  final_status: FinalStatus;
  current_phase: string;
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  progress: number;
  duration_seconds: number;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface Orchestration extends OrchestrationSummary {
  project_id: string;
  original_task: string;
  repository: string;
  tasks: Record<string, Task>;
  agent_results: Record<string, AgentResult>;
  errors: string[];
  test_results: Record<string, unknown>;
  repository_analysis: Record<string, unknown>;
  task_analysis: Record<string, unknown>;
  integration_report: IntegrationReport | null;
  updated_at: number;
}

export type TaskStatus =
  | "PENDING"
  | "READY"
  | "RUNNING"
  | "BLOCKED"
  | "SUCCESS"
  | "FAILED"
  | "RETRYING"
  | "CANCELLED";

export type RunStatus = "PENDING" | "RUNNING" | "PAUSED" | "COMPLETED" | "FAILED" | "CANCELLED";
export type FinalStatus =
  | "PENDING"
  | "RUNNING"
  | "SUCCESS"
  | "PARTIAL_SUCCESS"
  | "FAILED"
  | "CANCELLED";

export interface DagNode {
  id: string;
  title: string;
  type: string;
  status: TaskStatus;
  priority: number;
  agent: string | null;
  dependencies: string[];
  wave: number;
  depth: number;
  duration_seconds: number;
  retry_count: number;
  files_changed: string[];
  tests: string[];
  errors: string[];
  blocked_reason: string | null;
  commit: string | null;
}

export interface Dag {
  orchestration_id: string;
  total_tasks: number;
  waves: string[][];
  critical_path: string[];
  max_parallelism: number;
  status_counts: Record<string, number>;
  nodes: DagNode[];
  edges: { from: string; to: string }[];
}

export interface AgentInfo {
  id: string | null;
  name: string;
  role: string;
  description: string;
  capabilities: string[];
  max_concurrency: number;
  implementation: string | null;
}

export interface AgentActivity {
  task_id: string;
  title: string;
  role: string | null;
  status: TaskStatus;
  summary: string;
  files_changed: string[];
  commit: string | null;
  branch: string | null;
  duration_seconds: number;
  attempt: number;
  errors: string[];
  recommendations: string[];
  tests_passed: number;
  tests_failed: number;
}

export interface LogEntry {
  sequence: number;
  type: string;
  timestamp: number;
  data: Record<string, unknown>;
}

export interface Health {
  status: string;
  version: string;
  provider: string;
  model: string;
  agents_registered: number;
  sandbox: string;
  database: string;
  uptime_seconds: number;
  // From the probe taken at startup. null means the provider cannot be probed.
  provider_reachable: boolean | null;
  provider_model_available: boolean | null;
  provider_detail: string;
}

export interface SystemInfo {
  llm: {
    active_provider: string;
    active_model: string;
    available_providers: { id: string; label: string }[];
    stats: Record<string, number | string>;
    config?: {
      llm_provider: string;
      llm_model: string;
      gemini_model: string;
      gemini_has_key: boolean;
      api_model: string;
      api_base_url: string;
      api_has_key: boolean;
      ollama_model: string;
      ollama_base_url: string;
      ollama_has_key: boolean;
      ollama_num_predict: number;
      opencode_model: string;
    };
    probe?: {
      provider: string;
      base_url?: string;
      model: string;
      cloud?: boolean;
      has_key?: boolean;
      reachable: boolean;
      model_available: boolean | null;
      models: string[];
      error: string | null;
    } | null;
  };
  scheduling: Record<string, number>;
  sandbox: Record<string, unknown>;
  storage: Record<string, string>;
  roster: AgentInfo[];
}

export interface SystemConfigUpdate {
  llm_provider: string;
  llm_model?: string;
  api_key?: string;
  api_base_url?: string;
  persist_to_env?: boolean;
}

export interface LLMTestRequest {
  provider: string;
  model?: string;
  api_key?: string;
  base_url?: string;
  prompt?: string;
}

export interface LLMTestResponse {
  success: boolean;
  provider: string;
  model: string;
  latency_ms: number;
  reply: string;
  error?: string | null;
}


export interface GitChange {
  branch: string | null;
  commit: string | null;
  task_id: string;
  title: string;
  files_changed: string[];
}

export interface OrchestrationEvent {
  type: string;
  orchestration_id: string;
  timestamp: number;
  sequence: number;
  data: Record<string, unknown>;
}

// ── error handling ──────────────────────────────────────────────────────────

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      `Cannot reach the API at ${BASE}. Is the backend running?`,
      0,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(
      typeof detail === "string" ? detail : JSON.stringify(detail),
      response.status,
    );
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ── client ──────────────────────────────────────────────────────────────────

export const api = {
  baseUrl: BASE,

  health: () => request<Health>("/health"),
  systemInfo: () => request<SystemInfo>("/system/info"),
  models: () =>
    request<{
      provider: string;
      model: string;
      cloud: boolean;
      note: string | null;
      available: string[];
      error: string | null;
    }>("/system/models"),

  listProjects: () => request<Project[]>("/projects"),
  listAgents: () => request<AgentInfo[]>("/agents"),
  createProject: (data: {
    name: string;
    description: string;
    local_path: string;
    repository_url?: string | null;
    base_branch?: string;
  }) => request<Project>("/projects", { method: "POST", body: JSON.stringify(data) }),
  getProject: (id: string) => request<Project>(`/projects/${id}`),
  deleteProject: (id: string) => request<{ message: string }>(`/projects/${id}`, { method: "DELETE" }),

  listOrchestrations: (projectId?: string) =>
    request<OrchestrationSummary[]>(
      `/orchestrations${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`,
    ),
  startOrchestration: (data: {
    project_id: string;
    user_request: string;
    name?: string;
    llm_provider?: string;
    llm_model?: string;
    api_key?: string;
    api_base_url?: string;
    wait?: boolean;
  }) =>
    request<Orchestration>("/orchestrations", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  updateSystemConfig: (data: SystemConfigUpdate) =>
    request<{ message: string; provider: string; model: string }>("/system/config", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  testLLM: (data: LLMTestRequest) =>
    request<LLMTestResponse>("/system/test-llm", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  getOrchestration: (id: string) => request<Orchestration>(`/orchestrations/${id}`),
  deleteOrchestration: (id: string) =>
    request<{ message: string }>(`/orchestrations/${id}`, { method: "DELETE" }),
  pause: (id: string) =>
    request<{ message: string }>(`/orchestrations/${id}/pause`, { method: "POST" }),
  resume: (id: string) =>
    request<{ message: string }>(`/orchestrations/${id}/resume`, { method: "POST" }),
  cancel: (id: string) =>
    request<{ message: string }>(`/orchestrations/${id}/cancel`, { method: "POST" }),

  dag: (id: string) => request<Dag>(`/orchestrations/${id}/dag`),
  agents: (id: string) => request<AgentActivity[]>(`/orchestrations/${id}/agents`),
  logs: (id: string, after = 0) => request<LogEntry[]>(`/orchestrations/${id}/logs?after=${after}`),
  changes: (id: string) => request<GitChange[]>(`/orchestrations/${id}/changes`),
  diff: (id: string, ref?: string) =>
    request<{
      branch?: string;
      base?: string;
      files: { path: string; status: string; unmerged: boolean }[];
      insertions?: number;
      deletions?: number;
      patch: string;
    }>(`/orchestrations/${id}/diff${ref ? `?ref=${encodeURIComponent(ref)}` : ""}`),
  testResults: (id: string) =>
    request<{ summary: Record<string, unknown>; runs: TestRun[]; report: string }>(
      `/orchestrations/${id}/test-results`,
    ),
  report: (id: string) =>
    request<{
      markdown: string;
      success: boolean;
      merged_branches: string[];
      skipped: { title: string; skip_reason: string }[];
      conflicts: Record<string, unknown>[];
      regressions: string[];
      review_findings: Record<string, unknown>[];
    }>(`/orchestrations/${id}/report`),

  retryAgent: (taskId: string, orchestrationId?: string, reason = "") =>
    request<{ message: string }>(
      `/agents/${taskId}/retry${orchestrationId ? `?orchestration_id=${orchestrationId}` : ""}`,
      { method: "POST", body: JSON.stringify({ reason, reset_dependencies: true }) },
    ),
};

/**
 * Subscribe to a run's event stream.
 *
 * Returns an unsubscribe function. The caller is responsible for re-fetching
 * state on any event; the stream itself only carries notifications, which
 * keeps the connection cheap even for long runs.
 */
export function streamOrchestration(
  id: string,
  onEvent: (event: OrchestrationEvent) => void,
  onError?: (error: Event) => void,
): () => void {
  const source = new EventSource(`${BASE}/orchestrations/${id}/stream`);
  const handler = (raw: MessageEvent) => {
    try {
      onEvent(JSON.parse(raw.data) as OrchestrationEvent);
    } catch {
      /* ignore malformed frame */
    }
  };
  source.addEventListener("message", handler);
  for (const type of [
    "orchestration.started",
    "orchestration.phase",
    "orchestration.finished",
    "orchestration.paused",
    "orchestration.resumed",
    "orchestration.cancelled",
    "dag.created",
    "dag.updated",
    "task.started",
    "task.succeeded",
    "task.failed",
    "task.retrying",
    "task.blocked",
    "task.cancelled",
    "agent.started",
    "agent.finished",
    "integration.started",
    "integration.finished",
    "conflict.detected",
    "conflict.resolved",
    "tests.started",
    "tests.finished",
  ]) {
    source.addEventListener(type, handler as EventListener);
  }
  source.onerror = (event) => onError?.(event);
  return () => source.close();
}
