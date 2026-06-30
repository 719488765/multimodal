from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, shutdown_router, startup_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _name in ("app", "app.adapters", "app.services", "utils.emotion_inference_service"):
    logging.getLogger(_name).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_router()
    yield
    shutdown_router()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

origins = [o.strip() for o in settings.cors_allow_origin.split(",") if o.strip()]
if not origins:
    origins = ["*"]

# allow_credentials=True 与 allow_origins=["*"] 组合会导致浏览器丢弃 CORS 响应
_allow_credentials = "*" not in origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# 单端口模式：frontend/dist 存在时由 8000 同时提供页面 + API（只需转发 8000）
_EMOTION_AGENT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIST = _EMOTION_AGENT_ROOT / "frontend" / "dist"


@app.get("/")
def root():
    index = _FRONTEND_DIST / "index.html"
    if index.is_file():
        from fastapi.responses import FileResponse

        return FileResponse(index)
    return {"service": settings.app_name, "env": settings.app_env, "hint": "Run: emotion-agent/scripts/start_demo.sh"}


@app.get("/docs/system-architecture")
def system_architecture_figure():
    """系统架构图 HTML（论文/答辩用），与 project/docs/figures 同步。"""
    from fastapi.responses import FileResponse

    root = Path(settings.project_root).resolve()
    html_path = root / "docs" / "figures" / "system_architecture_figure.html"
    if not html_path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Not found: {html_path}")
    return FileResponse(html_path, media_type="text/html; charset=utf-8")


if _FRONTEND_DIST.is_dir() and (_FRONTEND_DIST / "index.html").is_file():
    assets_dir = _FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")
    logger.info("Serving frontend from %s (single-port http://0.0.0.0:8000)", _FRONTEND_DIST)
