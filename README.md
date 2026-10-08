# DINOv3-S와 CNN 전이학습 비교

소량 라벨 CIFAR-10에서 DINOv3-S Frozen + Linear, ResNet18 Frozen + Linear,
ResNet18 Fine-tuning을 비교하는 캡스톤 프로젝트입니다. 핵심 코드는 `src/`,
실험 실행과 결과 비교는 `scripts/`에 있습니다.

## 설치 (Windows PowerShell)

권장 환경은 **Python 3.11**입니다. 기존 머신러닝 환경과 분리해 설치하세요.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

`requirements.txt`는 검증 대상 버전을 고정합니다. torch 2.9.1 / torchvision
0.24.1은 대응 버전이며, Transformers 4.56.2에는 DINOv3 API가 있습니다.
GPU 환경에서는 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)로
해당 버전의 CUDA wheel을 설치하고 `torch.cuda.is_available()`을 확인하세요.
`device: auto`는 CUDA가 있으면 GPU, 없으면 CPU를 선택합니다.

## 먼저 실행할 최소 실험

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --smoke --models resnet18_frozen
.\.venv\Scripts\python.exe scripts/run_experiment.py --smoke --models resnet18_finetune
```

Smoke test도 **실제 CIFAR-10과 ImageNet 사전학습 ResNet18**을 사용합니다.
최초 실행은 CIFAR-10 약 170MB와 가중치 약 45MB 다운로드가 필요합니다.
CPU에서 1 epoch, 학습 10장(클래스별 1장), 검증 20장, 테스트 20장으로
파이프라인만 검증합니다. 이 수치는 연구 성능 결과로 사용할 수 없습니다.
외부 다운로드가 없는 단위 테스트는 `python -m pytest -q`로 실행합니다.

## DINOv3 접근 설정

실제 공식 Small 체크포인트는
[`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)입니다.
모델 페이지에서 접근 조건과 **DINOv3 라이선스**를 확인하고 동의/접근 신청을
진행하세요. 접근 승인된 계정의 토큰을 사용합니다.

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe scripts/run_experiment.py --smoke --models dinov3_frozen resnet18_frozen
```

또는 셸 환경 변수 `HF_TOKEN`을 설정합니다. 토큰은 YAML, 소스, Git에 저장하지
마세요. 인증 실패·미지원 Transformers 버전이면 DINOv3를 건너뛰고 다른 모델은
계속 실행하며 `failures.json`에 원인을 남깁니다. 선택한 모델이 모두 실패하면
종료 코드는 1입니다. CIFAR-10/ResNet 다운로드 실패는 기본 실험을 실행할 수
없으므로 오류로 종료합니다.

## 본 실험

```powershell
# Phase 1: ResNet18 Frozen, 전체 비율
.\.venv\Scripts\python.exe scripts/run_experiment.py --models resnet18_frozen
# Phase 2~4: 세 모델, 기본 1 Seed, 100/50/20/10%
.\.venv\Scripts\python.exe scripts/run_experiment.py
# 최종 반복 실험: 세 Seed의 평균과 표본 표준편차
.\.venv\Scripts\python.exe scripts/run_experiment.py --seeds 42 43 44
# 필요한 조건만 지정
.\.venv\Scripts\python.exe scripts/run_experiment.py --models resnet18_finetune --ratios 0.1 0.2 --seeds 42
```

설정은 `configs/default.yaml`에서 관리합니다. `--config`로 다른 YAML을 사용할
수 있습니다. 모든 상대 데이터/캐시/출력 경로는 프로젝트 루트 기준입니다.
기본 20 epoch, validation loss로 최적 모델 복원, patience 5 early stopping,
선형 head 학습률 0.001, fine-tuning 학습률 0.0001입니다. `patience: 0`은
early stopping을 끕니다. CNN 학습 계층은 `finetune_layers`로 설정하고 반드시
`fc`를 포함합니다. 전체 fine-tuning은 다음 목록을 사용합니다.

```yaml
finetune_layers: [conv1, bn1, layer1, layer2, layer3, layer4, fc]
```

CPU에서는 본 실험이 오래 걸릴 수 있습니다. Frozen 특징은 각 모델당 전체
train/validation/test에서 한 번 추출한 뒤 각 비율·Seed에서 재사용합니다.
변경되지 않은 모델 가중치·리비전·전처리·샘플·라벨·라이브러리 조건만 캐시를
재사용합니다. 원본 CIFAR-10 파일은 torchvision의 무결성 검사를 받습니다.

## 실험 설계와 결과 해석

- 공식 training 50,000장에서 고정 `split_seed`로 클래스별 10%를 validation으로
  분리합니다. 기본 train 45,000장, validation 5,000장, 공식 test 10,000장입니다.
- 비율의 분모는 **validation을 제외한 train**입니다. 10/20/50/100%는 각각
  4,500/9,000/22,500/45,000장입니다. 클래스별 순열의 prefix를 사용하므로
  같은 Seed에서 작은 비율 집합이 큰 비율 집합에 포함됩니다.
- 모델별로 같은 이미지 인덱스·라벨을 사용합니다. validation/test는 모델,
  비율, 반복 Seed 간 고정입니다. 반복 Seed는 라벨 부분집합·head 초기화·배치
  순서를 바꿉니다. test는 학습 및 모델 선택에 사용하지 않습니다.
- Frozen backbone은 `eval()`과 `inference_mode()`로 추출합니다. CNN에서
  동결된 BatchNorm의 running statistics도 변경하지 않습니다.
- ResNet18은 `IMAGENET1K_V1.transforms()`의 Resize/Crop/Normalize,
  DINOv3는 해당 체크포인트의 `AutoImageProcessor`를 적용합니다. 모든 모델에
  증강을 추가하지 않았습니다. 전처리는 가중치 권장 조건을 따르므로 서로
  다를 수 있습니다.
- [공식 DINOv3 문서](https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/dinov3)의
  `AutoModel` 및 `pooler_output`(CLS feature)을 사용하고 backbone은 모두 동결합니다.
  ResNet18 특징은 평균 풀링 후 512차원입니다. 두 Frozen 모델의 head는 하나의
  Linear 층, 같은 optimizer·학습률·epoch·선택 기준을 사용합니다.
- CIFAR-10 32×32 이미지를 사전학습 모델 입력 크기로 확대하는 한계가 있습니다.
  사전학습 데이터·학습 방식·전처리도 다르므로 아키텍처 자체의 인과적 우열이
  아닌 **사전학습 모델과 전이학습 전략의 실용적 비교**로 해석하세요.
- 최적 hyperparameter 탐색을 수행한 결과가 아닙니다. 단일 Seed의 SD는 CSV에서
  빈 값입니다. 여러 Seed의 error bar는 표본 SD이며 신뢰구간이 아닙니다.
- CUDA 연산 전후 동기화로 시간을 측정합니다. head/CNN 훈련 시간에는 validation과
  최적 상태 복원이 포함됩니다. 데이터/모델 다운로드·초기 로딩·테스트 추론·그래프
  생성 시간은 포함하지 않습니다.

## 출력 파일과 시간 비교

실행마다 `results/<실행시각>/`을 새로 생성합니다. Smoke는 `_smoke`로 표시합니다.

| 파일 | 내용 |
| --- | --- |
| `config.json`, `environment.json` | 적용 설정, 버전, device, smoke 여부 |
| `splits.json`, 모델별 `train_indices.json` | 재현용 분할과 실제 선택 샘플 |
| `*_backbone.json` | 전처리, 특징 차원, 체크포인트 및 resolved revision |
| `results.csv` | 모델/Seed/비율별 Accuracy, Macro F1, 시간, trainable parameters |
| `summary.csv` | 실제 반복 수, 평균, 표본 SD |
| `*_comparison.png` | Accuracy/F1/훈련 시간/전체 계산 시간 비교 |
| 모델별 `best_model.pt`, `history.json` | validation 기준 최적 가중치, epoch별 손실 |
| 모델별 `confusion_matrix.*`, `classification_report.json` | test confusion matrix, 클래스별 precision/recall/F1 |
| `failures.json` | 건너뛴 DINOv3와 원인 |

`linear_training_seconds`와 `finetuning_seconds`를 별도로 기록합니다.
`feature_extraction_seconds`는 **전체 train pool + validation + test**의 원래
추출 시간입니다. `train_feature_seconds`는 그중 train pool 시간입니다.
`cache_hit`가 true여도 원래 추출 비용을 보존합니다.
`total_compute_seconds = feature_extraction_seconds + training_seconds`입니다.
이는 모든 비율에 전체 pool 추출 비용을 더한 비용이며, 해당 소량 subset만
독립적으로 추출한 시간은 아닙니다. 같은 추출이 여러 실험에 공유되므로 CSV
행의 total을 합하면 실제 이번 실행의 경과 시간보다 커집니다. 소량 데이터
환경의 독립 실행 비용을 보고하려면 subset만 추출하는 별도 측정이 필요합니다.
단순 head 훈련 시간만으로 계산 효율을 결론 내리지 마세요.

그래프만 다시 만들려면:

```powershell
.\.venv\Scripts\python.exe scripts/compare_results.py --input results/<실행시각>/results.csv
```

비교기는 서로 다른 분할/smoke/test 조건 및 중복 Seed 행을 거부합니다.
