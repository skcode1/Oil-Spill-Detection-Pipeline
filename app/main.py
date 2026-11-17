from __future__ import annotations
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from app.api.deps import AppServices, build_services
from app.api.routes import router as api_router
from app.config import AppSettings
from app.errors import register_exception_handlers
from app.logging_config import configure_logging
logger = logging.getLogger(__name__)
API_PREFIX = '/api/v1'
__all__ = ['API_PREFIX', 'create_app', 'app']

def create_app(settings: Optional[AppSettings]=None, *, services: Optional[AppServices]=None, load_model: bool=True) -> FastAPI:
    resolved = settings if settings is not None else _load_or_none()
    server = resolved.server if resolved is not None else None
    configure_logging()
    app = FastAPI(title=server.title if server else 'Marine Oil Spill Detection & Attribution', version=server.version if server else '1.0.0', description=server.description if server else '', docs_url=server.docs_url if server else '/docs', openapi_url=server.openapi_url if server else '/openapi.json', lifespan=_lifespan(resolved, services, load_model))
    app.state.settings = resolved
    app.state.services = services
    _add_cors(app, resolved)
    _add_request_context(app)
    register_exception_handlers(app)
    app.include_router(api_router, prefix=API_PREFIX)

    @app.get('/', include_in_schema=False)
    def index() -> Dict[str, object]:
        return {'service': app.title, 'version': app.version, 'docs': app.docs_url, 'endpoints': {'health': f'{API_PREFIX}/health', 'regions': f'{API_PREFIX}/regions', 'pipeline': f'{API_PREFIX}/pipeline', 'pipeline_upload': f'{API_PREFIX}/pipeline/upload', 'drift_hindcast': f'{API_PREFIX}/drift/hindcast', 'drift_forecast': f'{API_PREFIX}/drift/forecast'}}
    logger.info('%s v%s ready at %s', app.title, app.version, API_PREFIX)
    return app

def _load_or_none() -> Optional[AppSettings]:
    from app.config import load_settings
    try:
        return load_settings()
    except Exception as exc:
        logger.error('config.yaml could not be read at startup: %s', exc)
        return None

def _lifespan(settings: Optional[AppSettings], services: Optional[AppServices], load_model: bool):

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, 'services', None) is None:
            app.state.services = build_services(settings, load_model=load_model)
        container: AppServices = app.state.services
        if container.settings.ais.allow_synthetic_ais:
            raise RuntimeError('attribution.allow_synthetic_ais is true: fabricating AIS vessel tracks invalidates the evidence chain. Set it to false in config.yaml.')
        logger.info('Startup complete: model ready=%s on %s, services built at %s', container.model.ready, container.model.device, container.started_at_utc)
        try:
            yield
        finally:
            container.close()
            logger.info('Shutdown complete: segmentation checkpoint released')
    return lifespan

def _add_cors(app: FastAPI, settings: Optional[AppSettings]) -> None:
    origins = list(settings.server.cors_origins) if settings else ['*']
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=origins != ['*'], allow_methods=['*'], allow_headers=['*'], expose_headers=['X-Request-ID', 'X-Process-Time-Ms'])

def _add_request_context(app: FastAPI) -> None:

    @app.middleware('http')
    async def request_context(request: Request, call_next) -> Response:
        request_id = request.headers.get('x-request-id') or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception('Unhandled error on %s %s (request_id=%s, %.0f ms)', request.method, request.url.path, request_id, (time.perf_counter() - started) * 1000.0)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers['X-Request-ID'] = request_id
        response.headers['X-Process-Time-Ms'] = f'{elapsed_ms:.1f}'
        logger.info('%s %s -> %s (%.0f ms, request_id=%s)', request.method, request.url.path, response.status_code, elapsed_ms, request_id)
        return response

def run() -> None:
    import uvicorn
    from app.config import load_settings
    settings = load_settings()
    uvicorn.run('app.main:app', host=settings.server.host, port=settings.server.port, reload=False)
app = create_app()
if __name__ == '__main__':
    run()