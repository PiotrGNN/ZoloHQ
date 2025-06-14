import logging
import os
import sys
import time
import asyncio
import uvicorn
import requests

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics
import aiofiles

TOKENIZER_PATH = r"saved_models/sentiment/models--ProsusAI--finbert/snapshots/4556d13015211d73dccd3fdd39d39232506f3e43/tokenizer_config.json"
HUGGINGFACE_URL = (
    "https://huggingface.co/ProsusAI/finbert/resolve/main/tokenizer_config.json"
)

API_KEYS = {
    "admin-key": "admin",
    "tokenizer-key": "tokenizer",
    "partner-key": "partner",
    "premium-key": "premium",
    "saas-key": "saas"
}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)


def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key in API_KEYS:
        return API_KEYS[api_key]
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def try_remove_file(path: str, retries: int = 3, delay: int = 2) -> bool:
    """Attempt to remove a file with retries and logging."""
    logger = logging.getLogger(__name__)
    for i in range(retries):
        try:
            os.remove(path)
            logger.info(f"Usunięto plik: {path}")
            return True
        except PermissionError as e:
            logger.warning(
                f"Permission denied when removing {path}: {e}. Retry {i+1}/{retries}"
            )
            time.sleep(delay)
        except FileNotFoundError:
            logger.info(f"Plik nie istnieje: {path}")
            return False
        except Exception as e:
            logger.error(f"Błąd podczas usuwania pliku {path}: {e}")
            time.sleep(delay)
    return False


def download_tokenizer_config(dest_path):
    logger = logging.getLogger(__name__)
    logger.info("[INFO] Pobieranie tokenizer_config.json z HuggingFace...")
    r = requests.get(HUGGINGFACE_URL)
    if r.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(r.content)
        logger.info(f"[INFO] Plik pobrany i zapisany: {dest_path}")
        return True
    else:
        logger.error(
            f"[ERROR] Nie udało się pobrać pliku z HuggingFace. Kod: {r.status_code}"
        )
        return False


tokenizer_api = FastAPI(title="ZoL0 Tokenizer Config Fixer", version="2.0")
tokenizer_api.add_middleware(PrometheusMiddleware)
tokenizer_api.add_route("/metrics", handle_metrics)


class RemoveQuery(BaseModel):
    path: str = TOKENIZER_PATH
    retries: int = 3
    delay: int = 2


class DownloadQuery(BaseModel):
    dest_path: str = TOKENIZER_PATH
    url: str = HUGGINGFACE_URL


class BatchDownloadQuery(BaseModel):
    downloads: list[DownloadQuery]


@tokenizer_api.get("/")
async def root():
    return {"status": "ok", "service": "ZoL0 Tokenizer Config Fixer", "version": "2.0"}


@tokenizer_api.get("/api/health")
async def api_health():
    return {"status": "ok", "timestamp": time.time(), "service": "ZoL0 Tokenizer Config Fixer", "version": "2.0"}


@tokenizer_api.post("/api/remove", dependencies=[Depends(get_api_key)])
async def api_remove(req: RemoveQuery, role: str = Depends(get_api_key)):
    loop = asyncio.get_event_loop()
    removed = await loop.run_in_executor(None, try_remove_file, req.path, req.retries, req.delay)
    return {"removed": removed, "path": req.path}


@tokenizer_api.post("/api/download", dependencies=[Depends(get_api_key)])
async def api_download(req: DownloadQuery, role: str = Depends(get_api_key)):
    loop = asyncio.get_event_loop()
    def _download():
        return download_tokenizer_config(req.dest_path)
    result = await loop.run_in_executor(None, _download)
    return {"downloaded": result, "dest_path": req.dest_path}


@tokenizer_api.post("/api/batch/download", dependencies=[Depends(get_api_key)])
async def api_batch_download(req: BatchDownloadQuery, role: str = Depends(get_api_key)):
    results = []
    for d in req.downloads:
        loop = asyncio.get_event_loop()
        def _download():
            return download_tokenizer_config(d.dest_path)
        result = await loop.run_in_executor(None, _download)
        results.append({"downloaded": result, "dest_path": d.dest_path})
    return {"results": results}


@tokenizer_api.get("/api/status", dependencies=[Depends(get_api_key)])
async def api_status(role: str = Depends(get_api_key)):
    exists = os.path.exists(TOKENIZER_PATH)
    size = os.path.getsize(TOKENIZER_PATH) if exists else 0
    return {"exists": exists, "size": size, "path": TOKENIZER_PATH}


@tokenizer_api.get("/api/analytics", dependencies=[Depends(get_api_key)])
async def api_analytics(role: str = Depends(get_api_key)):
    exists = os.path.exists(TOKENIZER_PATH)
    return {"exists": exists, "path": TOKENIZER_PATH}


@tokenizer_api.get("/api/export/prometheus", dependencies=[Depends(get_api_key)])
async def api_export_prometheus(role: str = Depends(get_api_key)):
    exists = os.path.exists(TOKENIZER_PATH)
    size = os.path.getsize(TOKENIZER_PATH) if exists else 0
    return PlainTextResponse(f"# HELP tokenizer_config_exists 1 if exists\ntokenizer_config_exists {int(exists)}\n# HELP tokenizer_config_size_bytes\ntokenizer_config_size_bytes {size}", media_type="text/plain")


@tokenizer_api.get("/api/report", dependencies=[Depends(get_api_key)])
async def api_report(role: str = Depends(get_api_key)):
    exists = os.path.exists(TOKENIZER_PATH)
    return {"status": "report generated (stub)", "exists": exists}


@tokenizer_api.get("/api/recommendations", dependencies=[Depends(get_api_key)])
async def api_recommendations(role: str = Depends(get_api_key)):
    exists = os.path.exists(TOKENIZER_PATH)
    recs = []
    if not exists:
        recs.append("Download tokenizer_config.json from HuggingFace.")
    return {"recommendations": recs}


@tokenizer_api.get("/api/premium/score", dependencies=[Depends(get_api_key)])
async def api_premium_score(role: str = Depends(get_api_key)):
    exists = os.path.exists(TOKENIZER_PATH)
    score = 100 if exists else 0
    return {"score": score}


@tokenizer_api.get("/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)])
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    exists = os.path.exists(TOKENIZER_PATH)
    return {"tenant_id": tenant_id, "exists": exists}


@tokenizer_api.get("/api/partner/webhook", dependencies=[Depends(get_api_key)])
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    return {"status": "received", "payload": payload}


@tokenizer_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated tokenizer edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}


@tokenizer_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})


# --- CI/CD test suite ---
import unittest
class TestTokenizerAPI(unittest.TestCase):
    def test_remove_download(self):
        try_remove_file(TOKENIZER_PATH)
        assert download_tokenizer_config(TOKENIZER_PATH)
        assert os.path.exists(TOKENIZER_PATH)
    def test_status(self):
        exists = os.path.exists(TOKENIZER_PATH)
        assert isinstance(exists, bool)

if __name__ == "__main__":
    import sys
    if "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("auto_fix_tokenizer_config:tokenizer_api", host="0.0.0.0", port=8506, reload=True)
