import { cx, statusClass, toneFor } from "../lib/ui";

interface Props {
  status: string;
  pulse?: boolean;
  className?: string;
}

/** Compact status pill. `pulse` is for genuinely in-flight states only. */
export function StatusBadge({ status, pulse = false, className }: Props) {
  const live = pulse && (status === "RUNNING" || status === "RETRYING");
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1",
        statusClass(status),
        className,
      )}
    >
      {live && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
        </span>
      )}
      {status.replace(/_/g, " ")}
    </span>
  );
}

export function Dot({ status }: { status: string }) {
  const tone = toneFor(status);
  const colors: Record<string, string> = {
    ok: "bg-emerald-400",
    bad: "bg-red-400",
    warn: "bg-amber-400",
    info: "bg-sky-400",
    busy: "bg-violet-400",
    idle: "bg-slate-500",
  };
  return <span className={cx("inline-block h-2 w-2 rounded-full", colors[tone])} />;
}
