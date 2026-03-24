"""
k8s_metric_resolver — K8s Pod 이름으로 Metric ID 조회 유틸리티

K8s TrainJob(Kubeflow Training Operator v2) 실행 시 컨테이너 hostname이
곧 pod 이름이 됩니다.  pod 이름은 다음 형태:

    {trainjob-name}-{replicatedjob-name}-{replica_index}-{pod_index}
    예) stock-training-tensorboard-7twqmj-trainer-0-0

API 서버에 pod_name 을 파라미터로 전달해 metric_id 를 조회합니다.
    GET http://{host}:{port}/api/v2/metric?pod-name={pod_name}
    응답 예) {"pod_name": "stock-training-tensorboard-ty1smc", "metric_id": 221}

사용법:
    from k8s_metric_resolver import resolve_metric_id
    metric_id = resolve_metric_id()   # hostname 자동 감지
    mlflow_logger.start(k8s_metric_id=metric_id)

환경변수 (K8s Secret 또는 컨테이너 env 로 주입 권장):
    METRIC_API_HOST  (기본: 192.168.0.16)
    METRIC_API_PORT  (기본: 8080)
"""

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request


# ──────────────────────────────────────────────────────────────────────────────
# 기본 API 설정  (환경변수가 우선 — K8s Secret 주입 권장)
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_API = {
    "host": "192.168.0.16",
    "port": 8080,
}


def get_pod_name() -> str:
    """현재 컨테이너의 hostname 반환 (K8s pod name 과 동일)"""
    return socket.gethostname()


def get_k8s_metric_id(pod_name: str = None) -> str:
    """pod_name 으로 API 서버에서 metric_id 조회

    Parameters
    ----------
    pod_name : None 이면 socket.gethostname() 자동 사용

    Returns
    -------
    str  : metric_id
    None : 미발견 또는 API 호출 오류
    """
    if pod_name is None:
        pod_name = get_pod_name()

    host = os.getenv("METRIC_API_HOST", _DEFAULT_API["host"])
    port = int(os.getenv("METRIC_API_PORT", str(_DEFAULT_API["port"])))

    url = f"http://{host}:{port}/api/v2/metric/{urllib.parse.quote(pod_name, safe='')}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        metric_id = str(data["metric_id"])
        print(
            f"[k8s_metric_resolver] pod={pod_name!r} → metric_id={metric_id}",
            flush=True,
        )
        return metric_id
    except urllib.error.URLError as exc:
        print(f"[k8s_metric_resolver] API 호출 실패: {exc}", flush=True)
        return None
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"[k8s_metric_resolver] 응답 파싱 실패: {exc}", flush=True)
        return None


def resolve_metric_id() -> str:
    """현재 pod 의 metric_id 를 자동으로 조회 (편의 함수)

    pod_name = socket.gethostname() 자동 감지 후 API 호출.

    Returns
    -------
    str  : metric_id
    None : 미발견 또는 오류
    """
    pod_name = get_pod_name()
    print(f"[k8s_metric_resolver] pod_name={pod_name!r}", flush=True)
    return get_k8s_metric_id(pod_name)

