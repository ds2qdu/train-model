"""
ClushMLflowLogger — 독립 MLflow 래퍼 (프레임워크 무관)
Lightning / Pure PyTorch / sklearn 어디서든 사용 가능

로깅 항목:
  [1] 모델 정보   : 모델명, 총 크기(MB), 총 파라미터 수
  [2] 하이퍼파라미터: epochs, batch_size
  [3] 데이터 정보  : train/val 샘플 수, train/val 크기(bytes)
  [4] GPU 정보    : GPU 이름, 사용 GPU 노드 수
  [5] Metric     : 시작/현재 시간, epoch 소요시간, checkpoint 저장시간,
                   train_loss, val_loss

──────────────────────────────────────────────────────────────
AI 활용 가이드 — PyTorch 분산 학습 전형적 삽입 패턴
──────────────────────────────────────────────────────────────

# [환경 변수] K8s Secret 또는 컨테이너 env 에 주입
#   MLFLOW_TRACKING_URI=http://<mlflow-service>:5000
#   미설정 시 로컬 파일 ./mlruns 에 저장

# [1] import + 초기화 (rank 0 only, DDP 학습 기준)
from clush_mlflow_logger import ClushMLflowLogger
from k8s_metric_resolver import resolve_metric_id   # K8s metric_id 자동 조회

mlflow_logger = None
if rank == 0:
    mlflow_logger = ClushMLflowLogger(
        experiment_name="my-experiment",        # MLflow 실험 이름
        run_name="run_v1",                      # 선택적 Run 표시 이름
        # tracking_uri="http://...:5000"        # env 대신 직접 지정 시
    )
    mlflow_logger.start(
        k8s_metric_id=resolve_metric_id()       # ← pod_name → DB → metric_id → tag
    )                                           # ← 반드시 먼저 호출
    mlflow_logger.log_hyperparams(epochs=100, batch_size=32)
    mlflow_logger.log_params({"lr": 0.0001, "d_model": 256})
    mlflow_logger.log_gpu_info(num_gpu_nodes=world_size)
    mlflow_logger.log_model_info("MyModel", model)   # DDP 래핑 전 순수 nn.Module
    mlflow_logger.log_dataset_info(train_dataset, val_dataset)

# [2] 학습 루프
for epoch in range(epochs):
    if mlflow_logger is not None:
        mlflow_logger.log_epoch_start()         # ← epoch 타이머 시작
    # ... train ...
    if rank == 0:
        val_loss = evaluate(...)
        if mlflow_logger is not None:
            mlflow_logger.log_epoch_end(epoch, train_loss, val_loss)
        # checkpoint 저장 시간 측정
        if mlflow_logger is not None:
            mlflow_logger.log_checkpoint_start()
        save_checkpoint(...)
        if mlflow_logger is not None:
            mlflow_logger.log_checkpoint_end(step=epoch)

# [3] 학습 종료 — Sweep CSV 기록 + Run 종료
if mlflow_logger is not None and args.sweep_csv:
    mlflow_logger.log_sweep_csv(            # 내부적으로 self._params 사용
        csv_path=args.sweep_csv,
        status="success",
        training_time_sec=training_time_sec,
        best_loss=best_loss,
    )
    mlflow_logger.end()                         # ← 반드시 마지막에 호출

──────────────────────────────────────────────────────────────
주의사항:
  - start() 를 먼저 호출해야 이후 log_* 메서드가 정상 동작
  - 분산 학습에서는 반드시 rank 0 에서만 인스턴스 생성 및 사용
  - log_model_info() 는 DDP 래핑 전 순수 nn.Module 전달 필요
  - log_epoch_start() 는 rank 0 이 아닌 곳에서 호출해도 무방하나
    log_epoch_end() 와 반드시 같은 rank 에서 쌍으로 호출해야 함
──────────────────────────────────────────────────────────────
"""

import csv
import os
import time
import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# ClushMLflowLogger
# ============================================================
class ClushMLflowLogger:

    def __init__(self, experiment_name: str, tracking_uri: str = None, run_name: str = None):
        """
        Parameters
        ----------
        experiment_name : MLflow 실험 이름
        tracking_uri    : MLflow 서버 URI (None이면 환경변수 또는 로컬 파일)
        run_name        : Run 표시 이름
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(uri)

        mlflow.set_experiment(experiment_name)

        self._run_name      = run_name
        self._run           = None
        self._start_time    = None
        self._epoch_start   = None
        self._ckpt_start    = None
        self._params: dict  = {}   # 로깅된 파라미터 누적 — log_sweep_csv 에서 재사용

    # ── Run 생명주기 ─────────────────────────────────────────

    def start(self, k8s_metric_id: str = None) -> None:
        """Run 시작 + 시작 시간 기록

        모든 log_* 메서드 호출 전에 반드시 먼저 실행해야 합니다.
        학습 루프 진입 직전(setup 완료 후)에 1회 호출하세요.

        Parameters
        ----------
        k8s_metric_id : K8s 플랫폼 metric ID (kub_work_list.id).
                        k8s_metric_resolver.resolve_metric_id() 로 자동 조회 가능.
                        None 이면 tag 저장 생략.
        """
        self._run        = mlflow.start_run(run_name=self._run_name)
        self._start_time = time.time()
        mlflow.log_param("time/start", time.strftime("%Y-%m-%d %H:%M:%S"))
        if k8s_metric_id is not None:
            mlflow.set_tag("k8s_metric_id", k8s_metric_id)
            self._params["k8s_metric_id"] = k8s_metric_id
            print(f"[MLflow] k8s_metric_id : {k8s_metric_id}", flush=True)
        print(f"[MLflow] run_id  : {self._run.info.run_id}", flush=True)
        print(f"[MLflow] run_url : {mlflow.get_tracking_uri()}/#/experiments/"
              f"{self._run.info.experiment_id}/runs/{self._run.info.run_id}", flush=True)

    def end(self) -> None:
        """Run 종료

        학습 완료 후 마지막으로 1회 호출하세요.
        with 문 사용 시(__enter__/__exit__) 자동 호출됩니다.
        """
        mlflow.end_run()
        self._run = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.end()

    # ── [1] 모델 정보 ────────────────────────────────────────

    def log_model_info(self, model_name: str, model) -> None:
        """
        Parameters
        ----------
        model_name : 모델 식별 이름 (e.g. "TinyLlama-1.1B")
        model      : torch.nn.Module

        주의: DDP(DistributedDataParallel) 래핑 전 순수 nn.Module 을
              전달해야 파라미터 수가 정확하게 집계됩니다.
        """
        import torch
        total_params = sum(p.numel() for p in model.parameters())
        total_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
        total_mb     = total_bytes / 1024 ** 2

        _p = {
            "model/name"        : model_name,
            "model/total_params": total_params,
            "model/size_mb"     : round(total_mb, 2),
        }
        mlflow.log_params(_p)
        self._params.update(_p)

    # ── [2] 하이퍼파라미터 ────────────────────────────────────

    def log_hyperparams(self, epochs: int, batch_size: int) -> None:
        _p = {"hparam/epochs": epochs, "hparam/batch_size": batch_size,
              "total_epochs": epochs}
        mlflow.log_params(_p)
        self._params.update(_p)

    # ── [3] 학습 데이터 정보 ─────────────────────────────────

    def log_dataset_info(self, train_dataset, val_dataset) -> None:
        """
        Parameters
        ----------
        train_dataset / val_dataset : HuggingFace Dataset 또는 torch Dataset
        크기(bytes)는 Arrow 포맷 nbytes 또는 추정값 사용
        """
        train_count = len(train_dataset)
        val_count   = len(val_dataset)

        def _estimate_bytes(dataset) -> int:
            # 1순위: HuggingFace Dataset nbytes (Arrow 포맷 기준)
            nb = getattr(dataset, "nbytes", None)
            if nb and nb > 0:
                return nb
            # 2순위: torch 텐서 크기로 추정 (샘플 1개 기준 × 전체 샘플 수)
            try:
                sample = dataset[0]
                import torch
                if isinstance(sample, dict):
                    one_sample_bytes = sum(
                        v.element_size() * v.nelement()
                        for v in sample.values()
                        if isinstance(v, torch.Tensor)
                    )
                elif isinstance(sample, torch.Tensor):
                    one_sample_bytes = sample.element_size() * sample.nelement()
                elif isinstance(sample, (tuple, list)):
                    one_sample_bytes = sum(
                        v.element_size() * v.nelement()
                        for v in sample
                        if isinstance(v, torch.Tensor)
                    )
                else:
                    return -1
                return one_sample_bytes * len(dataset)
            except Exception:
                return -1

        train_bytes = _estimate_bytes(train_dataset)
        val_bytes   = _estimate_bytes(val_dataset)

        _p = {
            "data/train_count"  : train_count,
            "data/val_count"    : val_count,
            "data/train_bytes"  : train_bytes,
            "data/val_bytes"    : val_bytes,
        }
        mlflow.log_params(_p)
        self._params.update(_p)

    # ── [4] GPU 정보 ─────────────────────────────────────────

    def log_gpu_info(self, num_gpu_nodes: int = 1) -> None:
        """
        Parameters
        ----------
        num_gpu_nodes : 학습에 사용하는 GPU 노드(물리 서버) 수
        """
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            _p = {"gpu/name": props.name, "gpu/num_nodes": num_gpu_nodes}
        else:
            _p = {"gpu/name": "cpu", "gpu/num_nodes": 0}
        mlflow.log_params(_p)
        self._params.update(_p)

    # ── [5] Metric ───────────────────────────────────────────

    def log_epoch_start(self) -> None:
        """epoch 시작 시각 기록 (소요시간 측정 시작)

        매 epoch 루프 최상단에서 1회 호출하세요.
        반드시 log_epoch_end() 와 쌍으로 사용해야 time/epoch_sec 가 기록됩니다.
        """
        self._epoch_start = time.time()
        mlflow.log_metric("time/current", time.time())

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        best_loss: float = None,
        no_improv_count: int = None,
    ) -> None:
        """epoch 완료 시 loss + 소요시간 기록"""
        elapsed = time.time() - self._epoch_start if self._epoch_start else -1
        metrics = {
            # 신규 표준명
            "train_loss"           : train_loss,
            "val_loss"             : val_loss,
            "sec_per_epoch"        : round(elapsed, 2),
            # 기존명 유지 (하위 호환)
            "metric/train_loss"    : train_loss,
            "metric/val_loss"      : val_loss,
            "time/epoch_sec"       : round(elapsed, 2),
            "time/current"         : time.time(),
        }
        if best_loss is not None:
            metrics["best_loss"] = best_loss
        if no_improv_count is not None:
            metrics["no_improv_count"] = no_improv_count
        mlflow.log_metrics(metrics, step=epoch)

    def log_checkpoint_start(self) -> None:
        """checkpoint 저장 시작 시각 기록

        save_checkpoint() 직전에 호출하세요.
        반드시 log_checkpoint_end() 와 쌍으로 사용해야 합니다.
        """
        self._ckpt_start = time.time()

    def log_checkpoint_end(self, step: int) -> None:
        """checkpoint 저장 완료 — 소요시간 기록"""
        elapsed = time.time() - self._ckpt_start if self._ckpt_start else -1
        mlflow.log_metric("time/checkpoint_save_sec", round(elapsed, 2), step=step)

    # ── [6] Sweep CSV 기록 ─────────────────────────────────────

    def log_sweep_csv(
        self,
        csv_path: str,
        status: str,
        training_time_sec: float,
        best_loss: float,
    ) -> None:
        """Sweep 결과를 통합 CSV 에 기록합니다.

        이 메서드가 직접 받는 인자는 4개만이며,
        나머지 컬럼(epochs, batch_size, gpu_model 등)은
        이전에 log_hyperparams / log_gpu_info / log_model_info /
        log_dataset_info / log_params 를 통해 누적된
        self._params 에서 자동으로 읽어옵니다.

        CSV 컬럼 스키마가 이 메서드 안에 중앙화되어 있어
        쉘 스크립트에서 컬럼을 별도 관리할 필요가 없습니다.
        파일 락(fcntl.flock)으로 n1/n2/n3 동시 쓰기를 방지합니다.

        성공 케이스에만 호출하세요.
        실패 fallback row 는 쉘 스크립트에서 처리합니다.

        Parameters
        ----------
        csv_path          : 기록할 CSV 파일 경로 (없으면 자동 생성)
        status            : "success" | "failed"
        training_time_sec : 전체 학습 소요 시간 (초)
        best_loss         : 최소 val loss
        """
        import fcntl
        from pathlib import Path as _Path

        p = self._params
        _tb = max(int(p.get("data/train_bytes", 0) or 0), 0)
        _vb = max(int(p.get("data/val_bytes", 0) or 0), 0)
        _ds_mb = round((_tb + _vb) / 1024 ** 2, 2) if (_tb + _vb) > 0 else ""

        row = {
            # 기본
            "run_id":            self._run.info.run_id if self._run else time.strftime("%Y%m%d_%H%M%S"),
            "k8s_metric_id":     p.get("k8s_metric_id", ""),
            "status":            status,
            "training_time_sec": training_time_sec,
            # [1] 모델 정보
            "model_name":        p.get("model/name", ""),
            "total_params":      p.get("model/total_params", ""),
            "model_size_mb":     p.get("model/size_mb", ""),
            # [2] 하이퍼파라미터
            "epochs":            p.get("hparam/epochs", ""),
            "batch_size":        p.get("hparam/batch_size", ""),
            # [3] 데이터 정보
            "train_count":       p.get("data/train_count", ""),
            "val_count":         p.get("data/val_count", ""),
            "dataset_size_mb":   _ds_mb,
            # [4] GPU 정보
            "gpu_name":          p.get("gpu/name", ""),
            "gpu_num_nodes":     p.get("gpu/num_nodes", ""),
        }

        csv_file = _Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        lock_path = csv_file.parent / ".sweep_csv.lock"
        with open(lock_path, "w") as _lock:
            fcntl.flock(_lock, fcntl.LOCK_EX)
            try:
                write_header = not csv_file.exists()
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    _w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        _w.writeheader()
                    _w.writerow(row)
            finally:
                fcntl.flock(_lock, fcntl.LOCK_UN)
        print(f"[CSV] Sweep 결과 기록 완료 → {csv_path}")

        # MLflow 에도 핵심 메트릭 기록 (run 활성화 시)
        if self._run is not None:
            mlflow.log_metrics({
                "sweep/training_time_sec": float(training_time_sec),
                "sweep/best_loss":         round(float(best_loss), 6),
            })
            mlflow.log_params({"sweep/status": status})

    # ── 범용 기록 메서드 ─────────────────────────────────────

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)
        self._params.update(params)

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_tags(self, tags: dict) -> None:
        mlflow.set_tags(tags)

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, data: dict, artifact_file: str) -> None:
        mlflow.log_dict(data, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        mlflow.log_text(text, artifact_file)

    # ── 조회 메서드 ──────────────────────────────────────────

    def get_run_id(self) -> str:
        return self._run.info.run_id if self._run else None

    def get_run_url(self) -> str:
        if self._run is None:
            return None
        tracking_uri = mlflow.get_tracking_uri()
        return f"{tracking_uri}/#/experiments/{self._run.info.experiment_id}/runs/{self._run.info.run_id}"
