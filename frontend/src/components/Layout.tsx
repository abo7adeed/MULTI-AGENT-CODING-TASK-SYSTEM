import type { ReactNode } from "react";
import { cx } from "../lib/ui";

export interface NavItem {
  key: string;
  label: string;
  icon: string;
}

export const NAV: NavItem[] = [
  { key: "dashboard", label: "Dashboard", icon: "▤" },
  { key: "new-task", label: "New Task", icon: "＋" },
  { key: "orchestration", label: "Execution", icon: "◇" },
  { key: "agents", label: "Agents", icon: "◈" },
  { key: "logs", label: "Logs", icon: "≡" },
  { key: "changes", label: "Git Changes", icon: "⑂" },
  { key: "tests", label: "Test Results", icon: "✓" },
  { key: "report", label: "Report", icon: "✦" },
  { key: "settings", label: "Settings", icon: "⚙" },
];

interface Props {
  page: string;
  onNavigate: (page: string, id?: string) => void;
  hasRun: boolean;
  health: { provider: string; model: string; agents_registered: number } | null;
  backendUp: boolean;
  live: boolean;
  children: ReactNode;
}

export function Layout({
  page,
  onNavigate,
  hasRun,
  health,
  backendUp,
  live,
  children,
}: Props) {
  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">
      <aside className="hidden w-56 shrink-0 flex-col border-r border-slate-800 bg-slate-900/40 md:flex">
        <div className="border-b border-slate-800 px-4 py-4">
          <p className="text-sm font-bold tracking-tight">Multi-Agent</p>
          <p className="text-[11px] text-slate-500">Coding Orchestrator</p>
        </div>
        <nav className="flex-1 space-y-0.5 p-2">
          {NAV.map((item) => {
            const disabled = item.key === "orchestration" && !hasRun;
            return (
              <button
                key={item.key}
                disabled={disabled}
                onClick={() => onNavigate(item.key)}
                className={cx(
                  "flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm transition",
                  page === item.key
                    ? "bg-sky-500/15 text-sky-200 ring-1 ring-sky-500/30"
                    : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200",
                  disabled && "cursor-not-allowed opacity-35 hover:bg-transparent",
                )}
              >
                <span className="w-4 text-center text-xs">{item.icon}</span>
                {item.label}
                {item.key === "orchestration" && live && (
                  <span className="ml-auto h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                )}
              </button>
            );
          })}
        </nav>
        <div className="border-t border-slate-800 px-4 py-3 text-[11px]">
          {backendUp && health ? (
            <>
              <p className="flex items-center gap-1.5 text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                API online
              </p>
              <p className="mt-1 text-slate-500">
                {health.provider}/{health.model}
              </p>
              <p className="text-slate-500">{health.agents_registered} agents</p>
            </>
          ) : (
            <p className="flex items-center gap-1.5 text-red-400">
              <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
              API offline
            </p>
          )}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-slate-800 px-4 py-3 md:hidden">
          <span className="text-sm font-bold">Multi-Agent Orchestrator</span>
          <div className="ml-auto flex gap-1 overflow-x-auto">
            {NAV.map((item) => (
              <button
                key={item.key}
                onClick={() => onNavigate(item.key)}
                className={cx(
                  "rounded px-2 py-1 text-xs",
                  page === item.key ? "bg-sky-500/20 text-sky-200" : "text-slate-400",
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        </header>
        <main className="min-w-0 flex-1 overflow-auto p-4 md:p-6">{children}</main>
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-xl font-bold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-slate-400">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Panel({
  title,
  children,
  actions,
  className,
}: {
  title?: string;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx("rounded-xl border border-slate-800 bg-slate-900/50", className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-slate-800 px-4 py-2.5">
          {title && <h2 className="text-sm font-semibold text-slate-200">{title}</h2>}
          {actions}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-800 px-6 py-14 text-center">
      <p className="text-sm font-medium text-slate-300">{title}</p>
      {hint && <p className="mt-1 max-w-md text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

export function ErrorBanner({ error, onDismiss }: { error: string; onDismiss?: () => void }) {
  if (!error) return null;
  return (
    <div className="mb-4 flex items-start gap-3 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
      <span className="flex-1">{error}</span>
      {onDismiss && (
        <button onClick={onDismiss} className="text-red-300 hover:text-white" aria-label="Dismiss">
          ✕
        </button>
      )}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-400">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-600 border-t-sky-400" />
      {label}
    </span>
  );
}
