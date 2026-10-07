"""
DAG engine -- pure graph algorithms.

Deliberately free of asyncio, I/O and policy. Everything here is deterministic
and trivially unit-testable: validation, cycle detection, ready-set computation,
topological ordering and parallel execution-wave planning.

Execution *policy* (concurrency limits, retries, cancellation) lives in
`app.engine.scheduler`. Orchestration *lifecycle* lives in
`app.engine.execution`. This module answers "what can run next?", never
"should we run it?".
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping

from app.models.domain import DAG, Task, TaskStatus


class DAGError(ValueError):
    """Raised when a graph is structurally invalid."""


class CycleDetectedError(DAGError):
    def __init__(self, path: list[str]):
        self.path = path
        super().__init__("Cycle detected: " + " -> ".join(path))


class MissingDependencyError(DAGError):
    def __init__(self, task_id: str, missing: Iterable[str]):
        self.task_id = task_id
        self.missing = sorted(missing)
        super().__init__(
            f"Task {task_id} depends on non-existent task(s): {', '.join(self.missing)}"
        )


class DAGEngine:
    """
    Read-only query surface over a `DAG`.

    The dependency source of truth is `Task.dependencies`. `DAG.edges` is a
    derived cache that `rebuild_edges()` refreshes; queries always recompute
    from the tasks so a stale cache can never corrupt scheduling.
    """

    def __init__(self, dag: DAG | Mapping[str, Task]):
        self.dag = dag if isinstance(dag, DAG) else DAG(tasks=dict(dag))

    # ── basic access ────────────────────────────────────────────────────────

    @property
    def tasks(self) -> dict[str, Task]:
        return self.dag.tasks

    def get(self, task_id: str) -> Task:
        try:
            return self.dag.tasks[task_id]
        except KeyError:
            raise KeyError(f"Unknown task: {task_id}") from None

    def has(self, task_id: str) -> bool:
        return task_id in self.dag.tasks

    def dependencies_of(self, task_id: str) -> list[str]:
        return list(self.get(task_id).dependencies)

    def dependents_of(self, task_id: str) -> list[str]:
        """Tasks that list `task_id` as a dependency."""
        return [t.id for t in self.dag.tasks.values() if task_id in t.dependencies]

    # ── validation ──────────────────────────────────────────────────────────

    def validate(self) -> bool:
        """
        Full structural validation.

        Raises `MissingDependencyError` or `CycleDetectedError` (both
        `DAGError`) on failure, and additionally rejects duplicate ids and
        self-dependencies. Returns True on success.
        """
        self._check_self_dependencies()
        self.check_dependencies_resolve()
        self.detect_cycles()
        return True

    def _check_self_dependencies(self) -> None:
        for task in self.dag.tasks.values():
            if task.id in task.dependencies:
                raise DAGError(f"Task {task.id} cannot depend on itself")

    def check_dependencies_resolve(self) -> None:
        problems: dict[str, list[str]] = {}
        for task in self.dag.tasks.values():
            missing = [d for d in task.dependencies if d not in self.dag.tasks]
            if missing:
                problems[task.id] = missing
        if problems:
            task_id = next(iter(problems))
            raise MissingDependencyError(task_id, problems[task_id])

    def detect_cycles(self) -> list[str] | None:
        """
        Iterative DFS with colours. Returns the cycle path, or None if acyclic.

        Iterative rather than recursive so a deep generated DAG cannot blow
        the Python stack.
        """
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {tid: WHITE for tid in self.dag.tasks}

        for root in self.dag.tasks:
            if colour[root] != WHITE:
                continue
            stack: list[tuple[str, int]] = [(root, 0)]
            path: list[str] = [root]
            colour[root] = GREY

            while stack:
                node, index = stack[-1]
                deps = self.dag.tasks[node].dependencies
                if index < len(deps):
                    stack[-1] = (node, index + 1)
                    dep = deps[index]
                    if dep not in colour:
                        continue  # dangling edge; reported by check_dependencies_resolve
                    if colour[dep] == GREY:
                        cycle = path[path.index(dep) :] + [dep]
                        raise CycleDetectedError(cycle)
                    if colour[dep] == WHITE:
                        colour[dep] = GREY
                        path.append(dep)
                        stack.append((dep, 0))
                else:
                    colour[node] = BLACK
                    stack.pop()
                    if path and path[-1] == node:
                        path.pop()
        return None

    # ── scheduling queries ──────────────────────────────────────────────────

    def roots(self) -> list[str]:
        return [t.id for t in self.dag.tasks.values() if not t.dependencies]

    def leaves(self) -> list[str]:
        return [t.id for t in self.dag.tasks.values() if not self.dependents_of(t.id)]

    def get_ready_tasks(self, completed: Iterable[str]) -> list[str]:
        """
        Tasks whose dependencies are ALL in `completed`.

        Sorted by (priority, creation order) so the highest-priority work is
        always dispatched first -- deterministic, not dict-iteration order.
        """
        completed_set = set(completed)
        ready = [
            task
            for task in self.dag.tasks.values()
            if task.id not in completed_set
            and all(dep in completed_set for dep in task.dependencies)
        ]
        return [t.id for t in self._sorted(ready)]

    def get_blocked_tasks(self, failed: Iterable[str]) -> list[tuple[str, str]]:
        """
        Tasks that can never run because a dependency failed or was cancelled.

        Returns (task_id, reason) pairs so the caller can record *why* rather
        than silently leaving tasks PENDING forever.
        """
        dead = set(failed)
        blocked: list[tuple[str, str]] = []
        changed = True
        while changed:
            changed = False
            for task in self.dag.tasks.values():
                if task.id in dead:
                    continue
                culprits = [d for d in task.dependencies if d in dead]
                if culprits:
                    names = ", ".join(
                        f"{self.dag.tasks[d].title if d in self.dag.tasks else d}"
                        for d in culprits
                    )
                    blocked.append((task.id, f"dependency failed: {names}"))
                    dead.add(task.id)
                    changed = True
        return blocked

    def ancestors(self, task_id: str) -> set[str]:
        """All transitive dependencies of `task_id`."""
        seen: set[str] = set()
        stack = list(self.dependencies_of(task_id))
        while stack:
            node = stack.pop()
            if node in seen or node not in self.dag.tasks:
                continue
            seen.add(node)
            stack.extend(self.dependencies_of(node))
        return seen

    def descendants(self, task_id: str) -> set[str]:
        """All tasks transitively depending on `task_id`."""
        seen: set[str] = set()
        stack = list(self.dependents_of(task_id))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.dependents_of(node))
        return seen

    def topological_order(self) -> list[str]:
        """Kahn's algorithm. Raises CycleDetectedError if the graph has a cycle."""
        indegree = {tid: len(set(t.dependencies)) for tid, t in self.dag.tasks.items()}
        queue = deque(tid for tid, d in indegree.items() if d == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.dependents_of(node):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.dag.tasks):
            self.detect_cycles()  # raises with the actual path
            raise DAGError("Graph is not a DAG")
        return order

    def parallel_waves(self) -> list[list[str]]:
        """
        Group tasks into execution waves.

        Every task in wave N depends only on tasks in waves < N, so an entire
        wave can run concurrently. This is what the UI renders as the graph
        and what the scheduler uses to reason about parallelism.
        """
        depth = self._depths()
        if depth is None:
            self.detect_cycles()
        max_depth = max(depth.values(), default=-1)
        waves: list[list[str]] = [[] for _ in range(max_depth + 1)]
        for task in self._sorted(self.dag.tasks.values()):
            waves[depth[task.id]].append(task.id)
        return waves

    def _depths(self) -> dict[str, int] | None:
        """Longest-path depth per node; None if a cycle exists."""
        depth: dict[str, int] = {}
        visiting: set[str] = set()

        def visit(node: str) -> int:
            if node in depth:
                return depth[node]
            if node in visiting:
                return -1
            visiting.add(node)
            task = self.dag.tasks.get(node)
            if task is None:
                visiting.discard(node)
                return 0
            best = 0
            for dep in task.dependencies:
                if dep not in self.dag.tasks:
                    continue
                sub = visit(dep)
                if sub < 0:
                    visiting.discard(node)
                    return -1
                best = max(best, sub + 1)
            visiting.discard(node)
            depth[node] = best
            return best

        for tid in self.dag.tasks:
            if visit(tid) < 0:
                return None
        return depth

    def critical_path(self) -> list[str]:
        """
        Longest dependency chain -- the theoretical critical path of the run.

        Ties on depth are broken by how load-bearing each node is (descendant
        count), then by priority, so the path shown is the one the scheduler is
        most likely to be waiting on.
        """
        depth = self._depths()
        if not depth:
            return []

        def rank(node_id: str) -> tuple[int, int, int, str]:
            return (
                depth.get(node_id, 0),
                len(self.descendants(node_id)),
                -self.tasks[node_id].priority,
                node_id,
            )

        end = max(depth, key=rank)
        path = [end]
        while True:
            deps = [d for d in self.tasks[path[-1]].dependencies if d in self.dag.tasks]
            if not deps:
                break
            path.append(max(deps, key=rank))
        return list(reversed(path))

    def estimated_parallelism(self, max_concurrency: int | None = None) -> int:
        """Widest wave -- the most tasks that could ever run at once."""
        waves = self.parallel_waves()
        widest = max((len(w) for w in waves), default=0)
        return min(widest, max_concurrency) if max_concurrency else widest

    # ── helpers ─────────────────────────────────────────────────────────────

    def _sorted(self, tasks: Iterable[Task]) -> list[Task]:
        """Priority first (1 = highest), then creation order for stability."""
        return sorted(
            tasks,
            key=lambda t: (t.priority, t.created_at, t.id),
        )

    def sync_statuses(self) -> None:
        """
        Mark BLOCKED tasks and refresh READY flags to match current statuses.

        Called by the scheduler so the persisted state always tells the truth,
        even if a process died mid-run.
        """
        succeeded = {t.id for t in self.dag.tasks.values() if t.status == TaskStatus.SUCCESS}
        terminal_failed = {
            t.id
            for t in self.dag.tasks.values()
            if t.status in (TaskStatus.FAILED, TaskStatus.CANCELLED)
        }
        for task_id, reason in self.get_blocked_tasks(terminal_failed):
            task = self.dag.tasks[task_id]
            if task.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.BLOCKED):
                task.status = TaskStatus.BLOCKED
                task.blocked_reason = reason
        ready = set(self.get_ready_tasks(succeeded | terminal_failed))
        for task in self.dag.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.BLOCKED) and task.id in ready:
                if task.status != TaskStatus.BLOCKED or task.blocked_reason is None:
                    task.status = TaskStatus.READY

    def summary(self) -> dict:
        """Compact graph description, also used by the API's /dag endpoint."""
        waves = self.parallel_waves()
        by_status: dict[str, int] = {}
        for task in self.dag.tasks.values():
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1
        return {
            "total_tasks": len(self.dag.tasks),
            "waves": waves,
            "max_wave_size": max((len(w) for w in waves), default=0),
            "critical_path": self.critical_path(),
            "status_counts": by_status,
            "edges": {tid: self.dependents_of(tid) for tid in self.dag.tasks},
        }
