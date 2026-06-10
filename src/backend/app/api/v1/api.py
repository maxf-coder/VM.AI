from fastapi import APIRouter
from app.api.v1.endpoints import schedule, tasks, provisional, stats, duration

api_router = APIRouter()

api_router.include_router(schedule.router, prefix="/schedule", tags=["Schedule"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
api_router.include_router(provisional.router, prefix="/provisional", tags=["Provisional"])
api_router.include_router(stats.router, prefix="/tasks", tags=["Stats"])
api_router.include_router(duration.router, prefix="/tasks", tags=["Tasks"])
