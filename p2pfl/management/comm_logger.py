from __future__ import annotations
from collections import defaultdict


import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import csv
import threading
from collections import defaultdict

@dataclass
class RoundCommStats:
    packets_out: int = 0
    bytes_out: int = 0
    packets_in: int = 0
    bytes_in: int = 0


class CommLogger:
    """
    Collects communication stats per node, per round.

    You call:
      - start_round(round_idx)
      - record_send(payload)
      - record_recv(payload)
      - end_round(round_idx)
      - export_csv(path)

    payload can be bytes OR any python object (we pickle to estimate bytes).
    """

    def __init__(self, node_addr: str, auto_save_path: Optional[str] = None) -> None:
        self.node_addr = node_addr
        self._lock = threading.Lock()
        self._round_stats: Dict[int, RoundCommStats] = {}
        self._active_round: Optional[int] = None
        self._created_at = time.time()
        self.bytes_out = defaultdict(int)
        self.packets_out = defaultdict(int)
        self.bytes_in = defaultdict(int)
        self.packets_in = defaultdict(int)
        self.file_path: Optional[str] = auto_save_path
        self.auto_save: bool = auto_save_path is not None

    def start_round(self, round_idx: int) -> None:
        with self._lock:
            self._active_round = round_idx
            self._round_stats.setdefault(round_idx, RoundCommStats())

    def end_round(self, round_idx: int) -> None:
        with self._lock:
            if self._active_round == round_idx:
                self._active_round = None
        
        # Auto-save after each round if enabled
        if self.auto_save and self.file_path:
            self.save()

    def _payload_size(self, payload: Any) -> int:
        if payload is None:
            return 0
        if isinstance(payload, (bytes, bytearray)):
            return len(payload)
        # estimate size by serialization
        return len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    def record_send(self, payload: Any, round_idx: Optional[int] = None) -> None:
        size = self._payload_size(payload)
        with self._lock:
            r = self._active_round if round_idx is None else round_idx
            if r is None:
                return
            st = self._round_stats.setdefault(r, RoundCommStats())
            st.packets_out += 1
            st.bytes_out += size

    def record_recv(self, payload: Any, round_idx: Optional[int] = None) -> None:
        size = self._payload_size(payload)
        with self._lock:
            r = self._active_round if round_idx is None else round_idx
            if r is None:
                return
            st = self._round_stats.setdefault(r, RoundCommStats())
            st.packets_in += 1
            st.bytes_in += size

    def snapshot_round(self, round_idx: int) -> RoundCommStats:
        with self._lock:
            return self._round_stats.get(round_idx, RoundCommStats())

    def export_csv(self, path: str) -> None:
        # Simple CSV writer without pandas dependency
        with self._lock:
            rows = [(r, s) for r, s in sorted(self._round_stats.items(), key=lambda x: x[0])]

        with open(path, "w", encoding="utf-8") as f:
            f.write("node,round,packets_out,bytes_out,packets_in,bytes_in\n")
            for r, s in rows:
                f.write(
                    f"{self.node_addr},{r},{s.packets_out},{s.bytes_out},{s.packets_in},{s.bytes_in}\n"
                )
    def record_send_bytes(self, num_bytes: int, round_idx: int) -> None:
        with self._lock:
            st = self._round_stats.setdefault(round_idx, RoundCommStats())
            st.packets_out += 1
            st.bytes_out += int(num_bytes)

    def record_recv_bytes(self, num_bytes: int, round_idx: int) -> None:
        with self._lock:
            st = self._round_stats.setdefault(round_idx, RoundCommStats())
            st.packets_in += 1
            st.bytes_in += int(num_bytes)
    def save_csv(self, path: str) -> None:
        with self._lock:
            items = sorted(self._round_stats.items(), key=lambda x: x[0])

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "packets_out", "bytes_out", "packets_in", "bytes_in"])
            for r, s in items:
                writer.writerow([r, s.packets_out, s.bytes_out, s.packets_in, s.bytes_in])

    def set_file_path(self, path: str, auto_save: bool = True) -> None:
        """Set the file path for automatic saving."""
        self.file_path = path
        self.auto_save = auto_save

    def save(self) -> None:
        """Save the logs to the configured file path."""
        if self.file_path:
            self.export_csv(self.file_path)
