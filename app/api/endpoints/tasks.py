"""Task status and download endpoints."""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import TaskResponse, TaskStatusEnum

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """Check the status of a summarization task."""
    from app.api.app import task_store

    task = await task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse(task_id=task_id, **task)


@router.get("/{task_id}/download")
async def download_result(task_id: str):
    """Download the summarized video for a completed task."""
    from app.api.app import task_store

    task = await task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Task is not completed (status: {task['status'].value})",
        )

    output_path = os.path.join(settings.OUTPUT_DIR, f"{task_id}_summary.mp4")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"{task_id}_summary.mp4",
    )
