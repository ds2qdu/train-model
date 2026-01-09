from kfp import dsl

# ✅ 너 GPU 이미지로 변경
BASE_IMAGE = "oratia/sp500-kfp:0.1-gpu"

# ✅ train 스크립트가 들어있는 GitLab 레포(기본값)
DEFAULT_REPO_URL = "https://github.com/oratia5702/kubeflow_sp500-pipeline.git"
DEFAULT_REPO_REF = "main"   # branch/tag/commit


def _clone_and_run(repo_url: str, repo_ref: str, cmd: str) -> str:
    """
    KFP ContainerSpec args로 넣을 bash 스니펫 생성.
    - private repo면 GIT_TOKEN 환경변수로 인증하도록 설계
      (예: CI/CD secret로 env 주입)
    """
    # repo_url이 https://... 형식이라고 가정
    # private repo면 아래 방식으로 토큰을 URL에 끼워넣을 수 있음:
    # https://oauth2:${GIT_TOKEN}@gitlab.example.com/group/repo.git
    return f"""
set -euo pipefail

echo "[INFO] Repo URL: {repo_url}"
echo "[INFO] Repo ref: {repo_ref}"

WORKDIR="/work"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 토큰이 있으면 URL에 주입 (GitLab OAuth2 token 방식)
# 일단 public repo라면 그냥 clone 됨
if [ -n "${{GIT_TOKEN:-}}" ]; then
  echo "[INFO] Using GIT_TOKEN for clone"
  CLONE_URL="$(echo "{repo_url}" | sed -E 's#https://#https://oauth2:${{GIT_TOKEN}}@#')"
else
  CLONE_URL="{repo_url}"
fi

rm -rf repo
git clone --depth 1 --branch "{repo_ref}" "$CLONE_URL" repo

cd repo
{cmd}
"""


@dsl.container_component
def step1_prepare_raw(
    repo_url: str,
    repo_ref: str,
    ticker: str,
    train_years: int,
    val_months: int,
    use_rates: bool,
    raw_out: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["bash", "-lc"],
        args=[_clone_and_run(
            repo_url, repo_ref,
            cmd=f"""
python scripts/prepare_raw.py \
  --ticker "{ticker}" \
  --train_years {train_years} \
  --val_months {val_months} \
  {"--use_rates" if use_rates else ""} \
  --out_raw "{raw_out.path}"
"""
        )],
    )


@dsl.container_component
def step2_make_dataset(
    repo_url: str,
    repo_ref: str,
    seq_len: int,
    val_months: int,
    use_rates: bool,
    raw_in: dsl.Input[dsl.Dataset],
    train_npz: dsl.Output[dsl.Dataset],
    val_npz: dsl.Output[dsl.Dataset],
    meta_out: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["bash", "-lc"],
        args=[_clone_and_run(
            repo_url, repo_ref,
            cmd=f"""
python scripts/make_dataset.py \
  --raw_in "{raw_in.path}" \
  --seq_len {seq_len} \
  --val_months {val_months} \
  {"--use_rates" if use_rates else ""} \
  --out_train "{train_npz.path}" \
  --out_val "{val_npz.path}" \
  --out_meta "{meta_out.path}"
"""
        )],
    )


@dsl.container_component
def step3_train(
    repo_url: str,
    repo_ref: str,
    lr: float,
    batch_size: int,
    epochs: int,
    train_years: int,
    val_months: int,
    seq_len: int,
    use_rates: bool,
    tb_root: str,
    run_name: str,
    # (옵션) step2 산출물을 쓰고 싶으면 여기서 입력 받게 확장 가능
    model_out: dsl.Output[dsl.Model],
    metrics_out: dsl.Output[dsl.Metrics],
    tb_logs: dsl.Output[dsl.Artifact],
):
    """
    ✅ 지금은 Katib용 train.py를 그대로 호출하는 형태.
    - stdout에 val_loss=... 출력
    - model 저장 파일은 model_out.path로
    - metrics json도 함께 저장하고, KFP metrics에 best_val_loss 기록
    - TB 로그는 tb_root 아래에 생기는데, 여기서는 tb_logs 아티팩트로 복사해둠
    """
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["bash", "-lc"],
        args=[_clone_and_run(
            repo_url, repo_ref,
            cmd=f"""
set -e

OUT_MODEL="{model_out.path}"
OUT_METRICS="/tmp/metrics.json"

# TensorBoard 경로: tb_root/sp500_lstm/run_name 에 생성되게
TB_ROOT="{tb_root}"
RUN_NAME="{run_name}"

mkdir -p "$(dirname "$OUT_MODEL")"

python scripts/train.py \
  --lr {lr} \
  --batch_size {batch_size} \
  --epochs {epochs} \
  --train_years {train_years} \
  --val_months {val_months} \
  --seq_len {seq_len} \
  {"--use_rates" if use_rates else ""} \
  --tb_root "$TB_ROOT" \
  --run_name "$RUN_NAME" \
  --out_model "$OUT_MODEL" \
  --out_metrics "$OUT_METRICS"

# KFP Metrics 기록 (json → metrics_out.path)
python - << 'PY'
import json, os
m_path = "{metrics_out.path}"
j_path = "/tmp/metrics.json"
with open(j_path,"r") as f:
    j=json.load(f)

# KFP v2 metrics artifact 포맷
payload = {{
  "metrics": [
    {{"name":"best_val_loss","numberValue": float(j["best_val_loss"]) if j["best_val_loss"] is not None else None}}
  ]
}}
os.makedirs(os.path.dirname(m_path), exist_ok=True)
with open(m_path,"w") as f:
    json.dump(payload,f)
print("[INFO] wrote kfp metrics:", m_path)
PY

# TB 로그를 아티팩트로 남기기 (옵션)
# tb_root/sp500_lstm/run_name → tb_logs.path 로 복사
TB_SRC="$TB_ROOT/sp500_lstm/$RUN_NAME"
if [ -d "$TB_SRC" ]; then
  mkdir -p "{tb_logs.path}"
  cp -r "$TB_SRC" "{tb_logs.path}/"
  echo "[INFO] copied tb logs from $TB_SRC to {tb_logs.path}"
fi
"""
        )],
    )


@dsl.container_component
def step4_eval_predict(
    repo_url: str,
    repo_ref: str,
    raw_in: dsl.Input[dsl.Dataset],
    meta_in: dsl.Input[dsl.Artifact],
    model_in: dsl.Input[dsl.Model],
    predictions_out: dsl.Output[dsl.Dataset],
    metrics_out: dsl.Output[dsl.Metrics],
):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["bash", "-lc"],
        args=[_clone_and_run(
            repo_url, repo_ref,
            cmd=f"""
python scripts/eval_predict.py \
  --raw_in "{raw_in.path}" \
  --meta_in "{meta_in.path}" \
  --model_in "{model_in.path}" \
  --out_predictions "{predictions_out.path}" \
  --out_metrics "{metrics_out.path}"
"""
        )],
    )


@dsl.pipeline(name="sp500-lstm-gitclone-pipeline")
def sp500_pipeline(
    repo_url: str = DEFAULT_REPO_URL,
    repo_ref: str = DEFAULT_REPO_REF,

    ticker: str = "^GSPC",
    seq_len: int = 60,
    val_months: int = 3,
    train_years: int = 3,

    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 5,

    use_rates: bool = True,

    # TB 로그는 PVC가 없으면 컨테이너 로컬에 생김
    # 일단 기본 경로 유지(나중에 PVC 마운트로 개선)
    tb_root: str = "/home/jovyan/tb_logs",
    run_name: str = "run",
):
    s1 = step1_prepare_raw(
        repo_url=repo_url,
        repo_ref=repo_ref,
        ticker=ticker,
        train_years=train_years,
        val_months=val_months,
        use_rates=use_rates,
    )

    s2 = step2_make_dataset(
        repo_url=repo_url,
        repo_ref=repo_ref,
        raw_in=s1.outputs["raw_out"],
        seq_len=seq_len,
        val_months=val_months,
        use_rates=use_rates,
    )

    s3 = step3_train(
        repo_url=repo_url,
        repo_ref=repo_ref,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        train_years=train_years,
        val_months=val_months,
        seq_len=seq_len,
        use_rates=use_rates,
        tb_root=tb_root,
        run_name=run_name,
    )

    step4_eval_predict(
        repo_url=repo_url,
        repo_ref=repo_ref,
        raw_in=s1.outputs["raw_out"],
        meta_in=s2.outputs["meta_out"],
        model_in=s3.outputs["model_out"],
    )


from kfp import compiler

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=sp500_pipeline,   # ← 위에서 만든 진짜 파이프라인
        package_path="sp500_pipeline.yaml",
    )
    print("compiled -> sp500_pipeline.yaml")
