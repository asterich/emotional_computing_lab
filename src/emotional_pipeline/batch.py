from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import ApiConfig, Paths


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class BatchClient:
    def __init__(self, config: ApiConfig):
        self.config = config

    def submit_and_wait(self, input_path: Path, output_path: Path, state_path: Path, poll_seconds: int) -> Optional[str]:
        if not self.config.enabled:
            return None
        state = self._load_state(state_path)
        batch_id = state.get("batch_id")
        if not batch_id:
            file_id = self.upload_file(input_path)
            batch = self.create_batch(file_id)
            batch_id = str(batch["id"])
            state.update(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "input_file_id": file_id,
                    "batch_id": batch_id,
                    "submitted_at": int(time.time()),
                }
            )
            self._save_state(state_path, state)

        return self.wait_for_batch(str(batch_id), output_path, state_path, poll_seconds)

    def wait_for_batch(self, batch_id: str, output_path: Path, state_path: Path, poll_seconds: int) -> Optional[str]:
        state = self._load_state(state_path)
        deadline = time.time() + poll_seconds
        last_status = ""
        while time.time() <= deadline:
            batch = self.get_batch(batch_id)
            last_status = str(batch.get("status", ""))
            state.update(
                {
                    "batch_id": batch_id,
                    "output_path": str(output_path),
                    "last_status": last_status,
                    "last_checked_at": int(time.time()),
                    "batch": batch,
                }
            )
            self._save_state(state_path, state)
            if last_status in {"completed", "succeeded", "success"}:
                output_file_id = batch.get("output_file_id") or batch.get("outputFileId")
                if output_file_id:
                    self.download_file(str(output_file_id), output_path)
                    return "completed"
                return "completed_no_output_file"
            if last_status in {"failed", "expired", "cancelled", "canceled"}:
                return last_status
            time.sleep(min(10, max(2, poll_seconds // 6 or 2)))
        return f"timeout:{last_status or 'unknown'}"

    def upload_file(self, path: Path) -> str:
        boundary = f"----emotional-pipeline-{uuid.uuid4().hex}"
        file_bytes = path.read_bytes()
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
                b"batch\r\n",
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
                b"Content-Type: application/jsonl\r\n\r\n",
                file_bytes,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        response = self._request(
            "POST",
            "/files",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        file_id = response.get("id")
        if not file_id:
            raise RuntimeError(f"File upload did not return id: {response}")
        return str(file_id)

    def create_batch(self, file_id: str) -> Dict[str, object]:
        return self._request(
            "POST",
            "/batches",
            payload={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )

    def get_batch(self, batch_id: str) -> Dict[str, object]:
        return self._request("GET", f"/batches/{batch_id}")

    def download_file(self, file_id: str, output_path: Path) -> None:
        data = self._request_bytes("GET", f"/files/{file_id}/content")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, object]] = None,
        body: Optional[bytes] = None,
        content_type: str = "application/json",
    ) -> Dict[str, object]:
        data = body
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        raw = self._request_bytes(method, path, data=data, content_type=content_type)
        return json.loads(raw.decode("utf-8"))

    def _request_bytes(
        self,
        method: str,
        path: str,
        data: Optional[bytes] = None,
        body: Optional[bytes] = None,
        content_type: str = "application/json",
    ) -> bytes:
        url = f"{self.config.base_url}{path}"
        request_data = body if body is not None else data
        request = urllib.request.Request(url, data=request_data, method=method)
        request.add_header("Authorization", f"Bearer {self.config.api_key}")
        request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {method} {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error for {method} {path}: {exc}") from exc

    @staticmethod
    def _load_state(path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_state(path: Path, state: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
