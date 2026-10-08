# 실제 검증 기록

검증일: 2026-10-08 (Asia/Seoul). 본 문서는 실제 실행한 내용만 기록합니다.

## 생성 파일

- `requirements.txt`, `.gitignore`, `configs/default.yaml`
- `src/__init__.py`, `dataset.py`, `models.py`, `features.py`, `train.py`, `evaluate.py`, `utils.py`
- `scripts/__init__.py`, `run_experiment.py`, `compare_results.py`
- `tests/test_dataset.py`, `test_models.py`, `test_training.py`, `test_comparison.py`, `test_dino.py`
- `README.md`, `VALIDATION.md`

`data/`, `cache/`, `results/`, `.venv/`는 실행 생성물이며 Git에서 제외합니다.
`cache/download_cifar_verified.py`는 이번 다운로드 검증용 임시 도구이며 프로젝트
실행에 필요하지 않습니다. 공식 서버의 byte range로 받은 파일을 torchvision의
공식 CIFAR-10 MD5 `c58f30108f718f92721af3b95e74349a`와 대조했습니다.

## 환경

- 호스트에 Python 3.10.11만 설치되어 있어 이번 실행은 해당 버전을 사용했습니다.
  권장 Python 3.11에서의 실행은 미검증입니다.
- PyTorch 2.9.1+cpu, torchvision 0.24.1+cpu; CUDA 없음.
- 전역 Transformers 4.44.2는 변경하지 않았습니다. 프로젝트 `.venv`를
  `--system-site-packages`로 만들고 Transformers 4.56.2, huggingface-hub 0.36.2,
  tokenizers 0.22.2를 그 환경에만 설치했습니다.
- 기존 환경 패키지 중 sam3/open-clip-torch의 의존성 충돌 경고가 있었습니다.
  이 프로젝트는 해당 패키지를 사용하지 않습니다. 팀원은 README처럼 새
  Python 3.11 venv에 전체 requirements를 설치하는 방식이 권장됩니다.

## 테스트 및 확인한 기능

```powershell
.\.venv\Scripts\python.exe -m pytest -q
python -m ruff check src scripts tests
```

최종 전체 단위 테스트: **8 passed**. Ruff 검사: **All checks passed**.

1. 클래스 균형, 분할 비중, train/validation 분리, 고정 Seed, Nested Sampling.
2. ResNet18 Frozen 특징 차원, 파라미터 동결.
3. 부분 fine-tuning의 trainable 계층과 frozen BatchNorm 통계 유지.
4. validation 최저 loss 상태 복원과 동일 Seed 학습 재현.
5. 특징 캐시의 값/라벨 일치, 재사용, 변경된 메타데이터로 무효화.
6. 단일 Seed에서 SD를 만들어내지 않고 비교 그래프 생성.
7. 중복 실험 행을 반복 Seed 결과로 집계하지 않도록 거부.
8. 작은 **임의 초기화** DINOv3 테스트 모델을 로컬로 저장/로딩하여
   AutoModel, 공식 processor API, pooled CLS shape, 동결을 오프라인 검증.
   임의 초기화 모델은 tests의 임시 디렉터리에서만 사용하며 실험 실행기에
   대체 모델로 사용하지 않습니다.

## 실제 CIFAR-10 + 사전학습 ResNet18 Smoke Test

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --smoke --models resnet18_frozen resnet18_finetune
```

정상 종료 (exit code 0). 결과:
`results/20261008_134448_395384_smoke/`.

학습 10장, 검증 20장, 테스트 20장, Seed 42, 1 epoch, CPU.
Frozen 특징은 train pool 100장 + validation 20장 + test 20장에서 추출하고,
그중 클래스별 1장씩의 학습 10장을 선택했습니다.

| 모델 | Accuracy | Macro F1 | 학습 가능 파라미터 |
| --- | ---: | ---: | ---: |
| ResNet18 Frozen + Linear | 0.20 | 0.080672 | 5,130 |
| ResNet18 Fine-tuning (layer4 + fc) | 0.05 | 0.015385 | 8,398,858 |

**이 결과는 파이프라인 검증용이며 연구 성능 비교에 사용할 수 없습니다.**
CSV, 최적 체크포인트, 손실 이력, 클래스별 지표, confusion matrix PNG/JSON,
평균/SD 표, Accuracy/Macro F1/학습 시간/추출 포함 시간 그래프가 생성됐습니다.
비교 그래프를 직접 열어 smoke 표시, 축, 범례를 확인했습니다.

추가 통합 실행:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --smoke --models dinov3_frozen resnet18_frozen resnet18_finetune
```

결과: `results/20261008_134510_817086_smoke/`, 정상 종료 (exit code 0).

- DINOv3 processor 접근은 **HTTP 401**로 실패했습니다. 실패 기록은 해당
  폴더의 `failures.json`에 있습니다.
- DINOv3를 건너뛴 뒤 두 ResNet18 실험이 완료되는 것을 확인했습니다.
- Frozen 결과의 `cache_hit=True`, 원래 추출 시간 유지가 확인됐습니다.
- 두 실행에서 동일 test Accuracy/Macro F1과 동일 split hash가 확인됐습니다.
- `best_model.pt`를 `weights_only=True`로 다시 읽는 데 성공했습니다.

## 외부 접근 실패와 미검증 항목

- 별도 공식 DINOv3 `config.json` 요청은 **HTTP 403**으로 거부됐습니다.
  실행기에서 processor 요청은 HTTP 401이었습니다. 접근 승인/인증이 필요합니다.
  [공식 모델 페이지](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)에서
  조건을 확인하고 접근 승인된 계정으로 `hf auth login` 또는 `HF_TOKEN`을
  설정한 뒤 다시 실행하세요. **공식 DINOv3 사전학습 가중치의 실제 특징 추출 및
  분류 성능은 미검증**입니다.
- 최초 sandbox 실행의 CIFAR 다운로드는 Windows socket access 오류
  (WinError 10013)로 실패했습니다. 허용된 네트워크 실행으로 공식 파일을
  다운로드한 뒤 로컬 smoke test는 정상 완료했습니다.
- 전체 데이터 4개 비율 × 3개 Seed 본 실험, Python 3.11, GPU 실행은 수행하지
  않았습니다. 해당 성능 수치나 표준편차를 생성하지 않았습니다.
- EfficientNet-B0 및 Streamlit 데모는 선택 확장으로 남겨두었습니다.
  핵심 실험과 단순한 코드 구조를 우선했습니다.

## 다음 실행

1. 팀 공통 Python 3.11 venv를 README대로 설치합니다.
2. DINOv3 접근 승인/인증 후 `--smoke --models dinov3_frozen`을 실행합니다.
3. 우선 단일 Seed의 전체 비율을 실행합니다.
4. GPU에서 `--seeds 42 43 44`로 세 모델 본 실험을 실행합니다.
5. `summary.csv`의 실제 Seed 수와 결과의 smoke 여부를 확인하고 보고서에
   CIFAR 확대 한계, 사전학습 조건 차이, 캐시 공유 시간의 의미를 명시합니다.

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --models resnet18_frozen
.\.venv\Scripts\python.exe scripts/run_experiment.py
.\.venv\Scripts\python.exe scripts/run_experiment.py --seeds 42 43 44
```
