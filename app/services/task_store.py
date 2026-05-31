"""PostgreSQL-backed task repository.

Replaces the previous in-memory TaskStore with async database access
using the Repository pattern. Each method manages its own session,
making it safe to call from both async route handlers and async
background tasks.
"""

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.models import Task
from app.db.session import async_session_factory
from app.models.schemas import TaskStatusEnum

logger = get_logger("task_repository")


class TaskRepository:
    """Async task repository backed by PostgreSQL."""

    async def create(self, task_id: str) -> dict:
        """Insert a new task with *queued* status."""
        async with async_session_factory() as session:
            task = Task(id=task_id, status=TaskStatusEnum.QUEUED.value)
            session.add(task)
            await session.commit()
            await session.refresh(task)
            logger.info("Task created: %s", task_id)
            return self._to_dict(task)

    async def update(self, task_id: str, **fields) -> None:
        """Update one or more fields on an existing task."""
        async with async_session_factory() as session:
            stmt = select(Task).where(Task.id == task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if task is None:
                raise KeyError(f"Task {task_id} not found")

            for key, value in fields.items():
                if key == "status" and isinstance(value, TaskStatusEnum):
                    value = value.value
                setattr(task, key, value)

            await session.commit()
            logger.debug("Task %s updated: %s", task_id, list(fields.keys()))

    async def get(self, task_id: str) -> dict | None:
        """Return a snapshot of the task, or *None* if not found."""
        async with async_session_factory() as session:
            stmt = select(Task).where(Task.id == task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            return self._to_dict(task) if task else None

    async def exists(self, task_id: str) -> bool:
        """Check whether a task with the given ID exists."""
        async with async_session_factory() as session:
            stmt = select(Task.id).where(Task.id == task_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none() is not None

    @staticmethod
    def _to_dict(task: Task) -> dict:
        """Convert an ORM Task instance to the dict format expected by routes."""
        return {
            "status": TaskStatusEnum(task.status),
            "output_url": task.output_url,
            "error": task.error,
        }
