"""
k8s_metric_resolver — K8s Pod 이름으로 Metric ID 조회 유틸리티

K8s TrainJob(Kubeflow Training Operator v2) 실행 시 컨테이너 hostname이
곧 pod 이름이 됩니다.  pod 이름은 다음 형태:

    {trainjob-name}-{replicatedjob-name}-{replica_index}-{pod_index}
    예) stock-training-tensorboard-7twqmj-trainer-0-0

DB(kub_work_list.name) 에는 trainjob-name 부분만 저장되어 있으므로
pod_name.startswith(name + "-") 로 매핑합니다.

사용법:
    from k8s_metric_resolver import resolve_metric_id
    metric_id = resolve_metric_id()   # hostname 자동 감지
    mlflow_logger.start(k8s_metric_id=metric_id)

환경변수 (K8s Secret 또는 컨테이너 env 로 주입 권장):
    K8S_DB_HOST      (기본: 10.123.4.153)
    K8S_DB_PORT      (기본: 5442)
    K8S_DB_USER      (기본: clush)
    K8S_DB_PASSWORD
    K8S_DB_NAME      (기본: de_ncp_gpus)
"""

import os
import socket


# ──────────────────────────────────────────────────────────────────────────────
# 기본 DB 설정  (환경변수가 우선 — K8s Secret 주입 권장)
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_DB = {
    "host":     "10.123.4.153",
    "port":     5442,
    "dbname":   "de_ncp_gpus",
    "user":     "clush",
    "password": "!clush12!@",
}


def get_pod_name() -> str:
    """현재 컨테이너의 hostname 반환 (K8s pod name 과 동일)"""
    return socket.gethostname()


def get_k8s_metric_id(pod_name: str = None) -> str:
    """pod_name 으로 kub_work_list.id (metric_id) 조회

    Parameters
    ----------
    pod_name : None 이면 socket.gethostname() 자동 사용

    Returns
    -------
    str  : metric_id (kub_work_list.id 의 문자열 표현)
    None : 미발견 또는 DB 접속 오류

    Notes
    -----
    - pod_name 이 job_name 으로 시작하는 row 를 최신순으로 검색
    - psycopg2 미설치 시 None 반환 (학습은 정상 진행)
    """
    if pod_name is None:
        pod_name = get_pod_name()

    db = {
        "host":     os.getenv("K8S_DB_HOST",     _DEFAULT_DB["host"]),
        "port":     int(os.getenv("K8S_DB_PORT",  str(_DEFAULT_DB["port"]))),
        "dbname":   os.getenv("K8S_DB_NAME",     _DEFAULT_DB["dbname"]),
        "user":     os.getenv("K8S_DB_USER",     _DEFAULT_DB["user"]),
        "password": os.getenv("K8S_DB_PASSWORD", _DEFAULT_DB["password"]),
    }

    try:
        import psycopg2
    except ImportError:
        print("[k8s_metric_resolver] psycopg2 미설치 — metric_id 조회 건너뜀", flush=True)
        return None

    conn = None
    try:
        conn = psycopg2.connect(**db, connect_timeout=5)
        cur = conn.cursor()
        # pod_name 이 name+'-' 로 시작하는 row 를 최신순으로 검색
        # %s 파라미터 바인딩으로 SQL Injection 방지
        cur.execute(
            """
            SELECT id, name
            FROM   public.kub_work_list
            WHERE  %s LIKE (name || '-%')
            ORDER  BY reg_dttm DESC
            LIMIT  1
            """,
            (pod_name,),
        )
        row = cur.fetchone()
        if row:
            metric_id = str(row[0])
            print(
                f"[k8s_metric_resolver] pod={pod_name!r} → "
                f"job={row[1]!r} → metric_id={metric_id}",
                flush=True,
            )
            return metric_id
        print(f"[k8s_metric_resolver] metric_id 미발견: pod_name={pod_name!r}", flush=True)
        return None
    except Exception as exc:
        print(f"[k8s_metric_resolver] DB 조회 실패: {exc}", flush=True)
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def resolve_metric_id() -> str:
    """현재 pod 의 metric_id 를 자동으로 조회 (편의 함수)

    pod_name = socket.gethostname() 자동 감지 후 DB 조회.

    Returns
    -------
    str  : metric_id
    None : 미발견 또는 오류
    """
    pod_name = get_pod_name()
    print(f"[k8s_metric_resolver] pod_name={pod_name!r}", flush=True)
    return get_k8s_metric_id(pod_name)
