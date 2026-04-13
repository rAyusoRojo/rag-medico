from fastapi import APIRouter

from app.api.routes.ask import router as ask_router

router = APIRouter()
router.include_router(ask_router)
