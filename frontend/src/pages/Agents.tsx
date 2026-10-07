import { useEffect, useState } from "react";
import { api, type AgentInfo, type Health } from "../api/client";
import { AgentTable } from "../components/AgentTable";
import { EmptyState, ErrorBanner, PageHeader, Panel, Spinner } from "../components/Layout";
import { StatusBadge } from "../components/StatusBadge";
import { useLiveRun } from "../lib/useLiveRun";
import { BTN_GHOST, BTN_PRIMARY, INPUT } from "../lib/ui";

interface Props {
  orchestrationId: string | null;
  health: Health | null;
  onNavigate: (page: string, id?: string) => void;
}

export function AgentsPage({ orchestrationId, health }: Props) {
  const [roster, setRoster] = useState<AgentInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const run = useLiveRun(orchestrationId, health);

  useEffect(() => {
    api
      .listAgents()
      .then((list) => {
        setRoster(list);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load agents"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Agents"
        subtitle={`${roster.length} specialised roles available for routing`}
      />
      <ErrorBanner error={error ?? ""} onDismiss={() => setError(null)} />

      <Panel title="Roster">
        {loading ? (
          <Spinner label="Loading roster…" />
        ) : (
          <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
            {roster.map((agent) => (
              <div
                key={agent.role}
                className="rounded-lg border border-slate-800 bg-slate-950/50 p-3 transition hover:border-slate-600"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-semibold text-slate-100">{agent.name}</p>
                  <span className="mono shrink-0 text-[10px] text-slate-500">{agent.role}</span>
                </div>
                <p className="mt-1 text-xs leading-relaxed text-slate-400">{agent.description}</p>
                <div className="mt-2 flex flex-wrap gap-1">
                  {agent.capabilities.slice(0, 5).map((c) => (
                    <span
                      key={c}
                      className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400"
                    >
                      {c}
                    </span>
                  ))}
                </div>
                {agent.implementation && (
                  <p className="mt-2 text-[10px] text-slate-600">
                    implemented by {agent.implementation}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </Panel>

      <Panel title="Activity in the current run">
        {orchestrationId ? (
          <>
            <AgentTable agents={run.agents} />
            {run.orchestration && (
              <p className="mt-3 text-xs text-slate-500">
                Run status <StatusBadge status={run.orchestration.status} /> ·{" "}
                {run.orchestration.completed_tasks}/{run.orchestration.total_tasks} tasks done
              </p>
            )}
          </>
        ) : (
          <EmptyState
            title="No run selected"
            hint="Start an orchestration to see which agent took which task."
          />
        )}
      </Panel>
    </div>
  );
}

export function SettingsPage() {
  const [info, setInfo] = useState<Awaited<ReturnType<typeof api.systemInfo>> | null>(null);
  const [models, setModels] = useState<{
    provider: string;
    cloud?: boolean;
    note?: string | null;
    available: string[];
    error: string | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form state
  const [selectedProvider, setSelectedProvider] = useState<string>("gemini");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [showKey, setShowKey] = useState(false);

  // Testing & Saving state
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    provider: string;
    model: string;
    latency_ms: number;
    reply: string;
    error?: string | null;
  } | null>(null);
  const [saving, setSaving] = useState(false);

  const load = () => {
    api.systemInfo()
      .then((data) => {
        setInfo(data);
        const active = data.llm.active_provider;
        setSelectedProvider(active);
        const cfg = data.llm.config;
        if (cfg) {
          if (active === "gemini") {
            setModel(cfg.gemini_model || "gemini-2.5-flash");
          } else if (active === "api") {
            setModel(cfg.api_model || "gpt-4o-mini");
            setBaseUrl(cfg.api_base_url || "https://api.openai.com/v1");
          } else if (active === "ollama") {
            setModel(
              cfg.ollama_model ||
                (cfg.ollama_base_url?.includes("ollama.com") ? "gpt-oss:120b" : "qwen2.5-coder:7b"),
            );
            setBaseUrl(cfg.ollama_base_url || "http://localhost:11434");
          } else if (active === "opencode") {
            setModel(cfg.opencode_model || "nemotron-3.5-lightning-free");
          }
        }
      })
      .catch((e) => setError(e.message));

    api.models().then(setModels).catch(() => undefined);
  };

  useEffect(() => {
    load();
  }, []);

  const handleProviderSelect = (prov: string) => {
    setSelectedProvider(prov);
    setTestResult(null);
    setSuccess(null);
    if (prov === "gemini") {
      setModel(info?.llm.config?.gemini_model || "gemini-2.5-flash");
      setBaseUrl("https://generativelanguage.googleapis.com/v1beta/openai");
    } else if (prov === "api") {
      setModel(info?.llm.config?.api_model || "gpt-4o-mini");
      setBaseUrl(info?.llm.config?.api_base_url || "https://api.openai.com/v1");
    } else if (prov === "ollama") {
      const url = info?.llm.config?.ollama_base_url || "http://localhost:11434";
      setModel(
        info?.llm.config?.ollama_model ||
          (url.includes("ollama.com") ? "gpt-oss:120b" : "qwen2.5-coder:7b"),
      );
      setBaseUrl(url);
    } else if (prov === "opencode") {
      setModel(info?.llm.config?.opencode_model || "nemotron-3.5-lightning-free");
    } else if (prov === "mock") {
      setModel("mock-model");
    }
  };

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    setError(null);
    try {
      const res = await api.testLLM({
        provider: selectedProvider,
        model: model.trim() || undefined,
        api_key: apiKey.trim() || undefined,
        base_url: baseUrl.trim() || undefined,
      });
      setTestResult(res);
    } catch (e) {
      setTestResult({
        success: false,
        provider: selectedProvider,
        model: model || "unknown",
        latency_ms: 0,
        reply: "",
        error: e instanceof Error ? e.message : "Test failed",
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const res = await api.updateSystemConfig({
        llm_provider: selectedProvider,
        llm_model: model.trim() || undefined,
        api_key: apiKey.trim() || undefined,
        api_base_url: baseUrl.trim() || undefined,
        persist_to_env: true,
      });
      setSuccess(`${res.message}. The system is now actively running with live dynamic models.`);
      setApiKey("");
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update configuration");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <PageHeader
        title="Settings & LLM Configuration"
        subtitle="Configure live AI providers (Google Gemini, OpenAI, Ollama) to run real agents instead of static mock data."
      />

      <ErrorBanner error={error ?? ""} onDismiss={() => setError(null)} />

      {success && (
        <div className="flex items-center justify-between rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
          <span>✓ {success}</span>
          <button onClick={() => setSuccess(null)} className="text-emerald-400 hover:text-emerald-200">
            ✕
          </button>
        </div>
      )}

      {/* Primary Configuration Panel */}
      <Panel title="Configure Active AI Provider">
        <div className="space-y-5">
          {/* Provider Selection Tabs */}
          <div>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-slate-400">
              Select Provider
            </label>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
              {[
                { id: "gemini", name: "Google Gemini", badge: "Recommended" },
                { id: "api", name: "OpenAI / API", badge: "GPT-4o, Groq" },
                {
                  id: "ollama",
                  name: "Ollama",
                  // The same provider serves a local server and the cloud, so
                  // the badge reports which one is actually configured.
                  badge: info?.llm.config?.ollama_base_url?.includes("ollama.com")
                    ? "Cloud"
                    : "Local Server",
                },
                { id: "opencode", name: "OpenCode CLI", badge: "CLI" },
                { id: "mock", name: "Mock", badge: "Static / Offline" },
              ].map((p) => {
                const isActive = selectedProvider === p.id;
                const isCurrentLive = info?.llm.active_provider === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => handleProviderSelect(p.id)}
                    className={`relative flex flex-col items-start rounded-xl border p-3 text-left transition ${
                      isActive
                        ? "border-sky-500 bg-sky-500/10 text-sky-100 ring-1 ring-sky-500/50"
                        : "border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex w-full items-center justify-between">
                      <span className="text-sm font-semibold">{p.name}</span>
                      {isCurrentLive && (
                        <span className="rounded bg-emerald-500/20 px-1.5 py-0.2 text-[10px] font-medium text-emerald-300">
                          Active
                        </span>
                      )}
                    </div>
                    <span className="mt-1 text-[11px] text-slate-400">{p.badge}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Provider Specific Inputs */}
          <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-4 space-y-4">
            {selectedProvider === "gemini" && (
              <>
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Gemini API Key
                    </label>
                    <a
                      href="https://aistudio.google.com/app/apikey"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-sky-400 hover:underline"
                    >
                      Get free Gemini API Key ↗
                    </a>
                  </div>
                  <div className="relative">
                    <input
                      type={showKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={
                        info?.llm.config?.gemini_has_key
                          ? "•••••••••••••••• (API key is already configured on server)"
                          : "AIzaSy..."
                      }
                      className={INPUT}
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="absolute right-3 top-2.5 text-xs text-slate-400 hover:text-slate-200"
                    >
                      {showKey ? "Hide" : "Show"}
                    </button>
                  </div>
                  <p className="mt-1 text-[11px] text-slate-500">
                    Your key is securely transmitted to the local API and stored in your project's .env file.
                  </p>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Model Selection
                  </label>
                  <input
                    type="text"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder="gemini-2.5-flash"
                    className={INPUT}
                  />
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"].map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setModel(m)}
                        className={`rounded-md border px-2 py-1 text-xs transition ${
                          model === m
                            ? "border-sky-500 bg-sky-500/20 text-sky-200"
                            : "border-slate-800 bg-slate-900 text-slate-400 hover:border-slate-700"
                        }`}
                      >
                        {m} {m === "gemini-2.5-flash" && "★ Fast"}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}

            {selectedProvider === "api" && (
              <>
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      API Key (Bearer Token)
                    </label>
                    <span className="text-xs text-slate-500">OpenAI, Groq, OpenRouter, DeepSeek</span>
                  </div>
                  <div className="relative">
                    <input
                      type={showKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={
                        info?.llm.config?.api_has_key
                          ? "•••••••••••••••• (API key is already configured on server)"
                          : "sk-..."
                      }
                      className={INPUT}
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="absolute right-3 top-2.5 text-xs text-slate-400 hover:text-slate-200"
                    >
                      {showKey ? "Hide" : "Show"}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
                    API Base URL
                  </label>
                  <input
                    type="text"
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                    placeholder="https://api.openai.com/v1"
                    className={INPUT}
                  />
                  <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
                    {[
                      { name: "OpenAI", url: "https://api.openai.com/v1", model: "gpt-4o-mini" },
                      { name: "Groq", url: "https://api.groq.com/openai/v1", model: "llama-3.3-70b-versatile" },
                      { name: "OpenRouter", url: "https://openrouter.ai/api/v1", model: "meta-llama/llama-3.3-70b-instruct" },
                      { name: "DeepSeek", url: "https://api.deepseek.com/v1", model: "deepseek-chat" },
                    ].map((preset) => (
                      <button
                        key={preset.name}
                        type="button"
                        onClick={() => {
                          setBaseUrl(preset.url);
                          setModel(preset.model);
                        }}
                        className="rounded border border-slate-800 bg-slate-900 px-2 py-0.5 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                      >
                        Preset: {preset.name}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Model Name
                  </label>
                  <input
                    type="text"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder="gpt-4o-mini"
                    className={INPUT}
                  />
                </div>
              </>
            )}

            {selectedProvider === "ollama" && (
              <>
                <div>
                  <div className="mb-1.5 flex items-center justify-between">
                    <label className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Ollama Endpoint
                    </label>
                    <span className="text-xs text-slate-500">
                      Local server or Ollama Cloud
                    </span>
                  </div>
                  <input
                    type="text"
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                    placeholder="http://localhost:11434"
                    className={INPUT}
                  />
                  <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
                    <button
                      type="button"
                      onClick={() => {
                        setBaseUrl("http://localhost:11434");
                        setModel("qwen2.5-coder:7b");
                      }}
                      className="rounded border border-slate-800 bg-slate-900 px-2 py-0.5 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                    >
                      Preset: Local server
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setBaseUrl("https://ollama.com");
                        setModel("gpt-oss:120b");
                      }}
                      className="rounded border border-slate-800 bg-slate-900 px-2 py-0.5 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                    >
                      Preset: Ollama Cloud
                    </button>
                  </div>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
                    API Key
                  </label>
                  <div className="relative">
                    <input
                      type={showKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={
                        info?.llm.config?.ollama_has_key
                          ? "•••••••••••••••• (API key is already configured on server)"
                          : "Required by ollama.com; leave empty for a local server"
                      }
                      className={INPUT}
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="absolute right-3 top-2.5 text-xs text-slate-400 hover:text-slate-200"
                    >
                      {showKey ? "Hide" : "Show"}
                    </button>
                  </div>
                  <p className="mt-1 text-[11px] text-slate-500">
                    A local server ignores this. Keys are stored in your project's .env file, never
                    in the browser.
                  </p>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Model Name
                  </label>
                  <input
                    type="text"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder={
                      baseUrl.includes("ollama.com") ? "gpt-oss:120b" : "qwen2.5-coder:7b"
                    }
                    className={INPUT}
                  />
                  {baseUrl.includes("ollama.com") && (
                    <p className="mt-1 text-[11px] text-slate-500">
                      Use the API name from the list below ({models?.available[0] ?? "gpt-oss:120b"});
                      the app and CLI add a <span className="mono">:cloud</span> suffix that the API
                      rejects.
                    </p>
                  )}
                </div>
              </>
            )}

            {selectedProvider === "opencode" && (
              <div>
                <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
                  OpenCode Model
                </label>
                <input
                  type="text"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="nemotron-3.5-lightning-free"
                  className={INPUT}
                />
              </div>
            )}

            {selectedProvider === "mock" && (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200 leading-relaxed">
                ⚠️ <strong>Mock mode (Static Data):</strong> In this mode, no real AI model is called.
                Responses are deterministic canned stubs designed for offline automated tests.
                Select <strong>Google Gemini</strong> or <strong>OpenAI</strong> above to run real code generation!
              </div>
            )}

            {/* Test Connection Results Preview */}
            {testResult && (
              <div
                className={`rounded-lg border p-3 text-xs space-y-1 ${
                  testResult.success
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                    : "border-red-500/40 bg-red-500/10 text-red-200"
                }`}
              >
                <div className="flex items-center justify-between font-semibold">
                  <span>{testResult.success ? `✓ Connection Verified (${testResult.latency_ms}ms)` : "✕ Connection Failed"}</span>
                  <span className="mono text-[11px] opacity-75">{testResult.provider} / {testResult.model}</span>
                </div>
                {testResult.success && testResult.reply && (
                  <p className="mt-1 italic text-slate-300">"{testResult.reply}"</p>
                )}
                {testResult.error && (
                  <p className="mt-1 text-red-300 font-mono text-[11px]">{testResult.error}</p>
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
              <button
                type="button"
                onClick={handleTestConnection}
                disabled={testing}
                className={BTN_GHOST}
              >
                {testing ? "Testing Connectivity…" : "Test Connection"}
              </button>

              <button
                type="button"
                onClick={handleSave}
                disabled={saving}
                className={BTN_PRIMARY}
              >
                {saving ? "Saving & Activating…" : "Save & Activate Provider"}
              </button>
            </div>
          </div>
        </div>
      </Panel>

      {/* Active Diagnostics & Details */}
      <Panel title="Active System Status">
        {info ? (
          <dl className="space-y-2 text-sm">
            <Row label="Live Provider">
              <span className="font-semibold text-slate-100">{info.llm.active_provider}</span>
              <span className="mx-2 text-slate-600">/</span>
              <span className="mono text-sky-400">{info.llm.active_model}</span>
            </Row>
            <Row label="Total Calls">{String(info.llm.stats.calls ?? 0)}</Row>
            <Row label="Total Tokens">{String(info.llm.stats.tokens ?? 0)}</Row>
            {info.llm.probe && (
              <Row label="Provider Probe">
                {info.llm.probe.reachable && info.llm.probe.model_available ? (
                  <span className="text-emerald-300">
                    reachable, {info.llm.probe.model} is available
                  </span>
                ) : (
                  <span className="text-amber-300">
                    {info.llm.probe.error ||
                      `${info.llm.probe.model} could not be confirmed at ${info.llm.probe.base_url}`}
                  </span>
                )}
              </Row>
            )}
            {models && (
              <Row label="Models Reachable">
                {models.error ? (
                  <span className="text-amber-300">{models.error}</span>
                ) : models.available.length ? (
                  <div className="flex flex-wrap gap-1">
                    {models.available.slice(0, 10).map((m) => (
                      <span
                        key={m}
                        className="mono rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-300"
                      >
                        {m}
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-slate-500">
                    none reported by {models.provider}
                  </span>
                )}
                {models.note && (
                  <p className="mt-2 text-[11px] leading-relaxed text-slate-500">{models.note}</p>
                )}
              </Row>
            )}
          </dl>
        ) : (
          <Spinner />
        )}
      </Panel>

      {/* Scheduling & Sandbox Panels */}
      <div className="grid gap-5 md:grid-cols-2">
        <Panel title="Scheduling Policy">
          {info ? (
            <dl className="space-y-2 text-sm">
              <Row label="Parallel Tasks">{info.scheduling.max_parallel_tasks} agents</Row>
              <Row label="Max Retries">{info.scheduling.max_task_retries} per task</Row>
              <Row label="Task Timeout">{info.scheduling.task_timeout_seconds}s</Row>
            </dl>
          ) : (
            <Spinner />
          )}
        </Panel>

        <Panel title="Sandbox Isolation">
          {info ? (
            <div className="space-y-2 text-sm">
              <Row label="Mode">
                {info.sandbox.policy ? "Docker isolation" : "Host process (Isolated worktrees)"}
              </Row>
              <p className="text-xs text-slate-400">
                Each agent writes to an isolated Git worktree branch, ensuring concurrency safety and git three-way merge review.
              </p>
            </div>
          ) : (
            <Spinner />
          )}
        </Panel>
      </div>
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

