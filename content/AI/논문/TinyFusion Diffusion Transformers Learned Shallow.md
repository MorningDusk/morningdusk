---
{"publish":true,"aliases":"TinyFusion: Diffusion Transformers Learned Shallow","created":"2025-07-09","modified":"2025-07-31T16:25:23.228+09:00","tags":["ai","article","computer_vision","diffusion"],"cssclasses":""}
---

# Introduction
## 연구 배경과 동기
Diffusion Transformer는 이미지 생성 분야에서 혁신적인 성능을 보여주고 있다. 특히 DiT, MAR, SiT 같은 모델들이 놀라운 이미지 생성 품질을 달성했다.  하지만 이런 모델들은 엄청난 양의 매개변수를 가지고 있어서 실제 사용할 때 큰 문제가 생긴다. 예를 들어, DiT-XL/2는 6.75억개의 매개변수를 가지고 있어서 일반 사용자가 자신의 컴퓨터에서 돌리기에는 너무 무겁다.
이런 문제를 해결하기 위해 모델 경량화 기법들이 연구되고 있다. 그 중에서도 **depth pruning**은 전체 레이어를 통째로 제거하는 방법으로, 실제 속도 향상에 가장 효과적이다. 왜냐하면 GPU 같은 병렬 처리 장치에서는 레이어의 개수가 처리 시간에 직접적으로 영향으 주기 때문이다.
![[Image/Pasted image 20250710115716.png]]
## 기존 방법의 한계점
기존의 depth pruning 방법들은 주로 **loss minimization principle**을 따랐다. 즉, 다음과 같은 목표를 가지고 있었다:
$$
\min_m \mathbb{E}_x [L(x, \Phi, m)]
$$
여기서 $m$은 binary mask이고, $\Phi$는 모델의 매개변수다. 이 방법은 pruning 직후의 성능 저하를 최소화하려고 한다. 하지만 연구진이 실험해본 결과, 이런 접근법이 diffusion transformer에서는 효과적이지 않다는 것을 발견했다.
연구진은 100,000개의 random pruning 모델을 만들어서 calibration loss와 fine-tuning 후 성능의 관계를 분석했다. 놀랍게도 calibration loss가 낮은 모델이 fine-tuning 후에 반드시 좋은 성능을 보이는 것은 아니었다. 이는 기존의 접근법이 diffusion transformer에는 적합하지 않다는 것을 보여준다.
## 새로운 접근법: Recoverability
이 논문에서는 **Recoverability**<sub>회복 가능성</sub>라는 새로운 개념을 제안한다. Pruning 이후에 성능이 좋은 것보다, fine-tuning을 통해 성능을 회복할 수 있는 가능성이 더 중요하다는 아이디어다. 이를 수식으로 표현하면 다음과 같다:
$$
\min_m \min_{\Delta\Phi} \mathbb{E}_x [L(x, \Phi + \Delta\Phi, m)]
$$
여기서 $\Delta \Phi$는 fine-tuning으로 얻어지는 가중치 업데이터를 의미한다. 즉, fine-tuning 후의 성능을 직접 최적화하겠다는 뜻이다.
# Related Works
## Network Pruning의 발전
Network pruning은 크게 두 가지 방향으로 발전해왔다. 첫 번째는 **width pruning**으로, 각 레이어의 뉴런 수를 줄이는 방법이다. 두 번째는 **depth pruning**으로, 전체 레이어를 제거하는 방법이다.
Width pruning의 경우 메모리 사용량은 줄일 수 있지만, 실제 속도 향상은 제한적이다. 예를 들어, 50% width pruning은 1.6배 속도 향상만 가져다준다. 반면 depth pruning은 레이어 수에 비례해서 선형적으로 속도가 향상된다. 50% depth pruning은 정확히 2배 속도 향상을 가져다준다.
## Diffusion Transformer의 효율성 연구
Diffusion Transformer를 효율적으로 만들기 위한 연구들이 활발히 진행되고 있다. 여기에는 linear attention, compact architecture, quantization 등 다양한 방법들이 포함된다. 이 논문은 그 중에서도 depth pruning에 집중해서 실용적인 해결책을 제공한다.
# Method
## TinyFusion의 핵심 아이디어
TinyFusion은 pruning과 fine-tuning을 하나의 최적화 문제로 통합한다. 핵심 아이디어는 다음과 같다:
1. **Probabilistic Perspective**: 모든 가능한 pruning mask에 확률을 부여한다
2. **Local Structure Sampling**: 전체 모델을 작은 블록으로 나누어서 처리한다
3. **Differentiable Sampling**: Gumbel-Softmax를 사용해서 sampling을 미분 가능하게 만든다
4. **Joint Optimization**: LoRA를 사용해서 fine-tuning을 시뮬레이션한다
## Local Structure Sampling
전체 모델을 $K$개 블록으로 나눈다: $\Phi = [\Phi_1, \Phi_2, \cdots, \Phi_K]^T$. 각 블록 $\Phi_k$는 $M$개의 레이어를 포함하고, 그 중에서 N개를 선택하는 **N:M pruning scheme**을 사용한다.
예를 들어, 1:2 scheme이라면 2개 레이어 중 1개를 남긴다는 뜻이다. 이때 가능한 모든 mask는 다음과 같이 나타낼 수 있다:
$$
\hat{m}_{1:2}=[[1,0][0,1]]
$$
각 블록의 확률 분포는 독립적이므로, 전체 분포는:
$$
p(m)=p(m_1) \cdot p(m_2) \cdots p(m_K)
$$
![[Image/Pasted image 20250710115759.png]]
## Differentiable Sampling
Sampling 과정을 미분 가능하게 만들기 위해 **Gumbel-Softmax**를 사용한다:
$$
y = \text{one-hot}\left(\frac{\exp((g_i + \log p_i)/\tau)}{\sum_j \exp((g_j + \log p_j)/\tau)}\right)
$$
여기서 $g_i$는 Gumbel 노이즈이고, $\tau$는 temperature parameter다. 이를 통해 다음과 같이 mask를 샘플링할 수 있다:
$$
m = y^T \hat{m}
$$
## Joint Optimization with LoRA
Fine-tuning을 시뮬레이션하기 위해 LoRA<sub>Low-Rank Adaptation</sub>를 사용한다. 원래 가중치 $W$를 다음과 같이 업데이트한다:
$$
W_{fine-tuned}=W+\alpha \Delta W=W+\alpha BA
$$
여기서 $\alpha$는 scaling factor이고, $B$와 $A$는 low-rank 행렬이다. 이렇게 하면 매개변수 수를 크게 줄이면서도 fine-tuning의 효과를 시뮬레이션할 수 있다.
![[Image/Pasted image 20250710115932.png]]
# Experiments
## 실험 설정
### 데이터셋
ImageNet 256×256을 사용했다. 이미지를 center crop하고 적절히 resize한 후 평균 0.5, 표준편차 0.5로 정규화했다.
### 평가 지표
- **FID**<sub>Frechet Inception Distance</sub>: 생성된 이미지와 실제 이미지 간의 분포 차이
- **IS**<sub>Inception Score</sub>: 생성된 이미지의 품질과 다양성
- **sFID, Precision, Recall**: 추가적인 품질 지표들
- **Sampling Speed**: 초당 생성 가능한 이미지 수
## 주요 실험 결과
### DiT-XL/2 결과
DiT-XL/2 (28 레이어, 6.75 억 파라미터)에서 50% 레이어를 제거한 결과

| 모델               | 레이어 수 | 매개변수  | FID  | 속도 (it/s) | 학습 비용       |
| ---------------- | ----- | ----- | ---- | --------- | ----------- |
| DiT-XL/2 (원본)    | 28    | 675 M | 2.27 | 6.91      | 7000K steps |
| TinyDiT-D14      | 14    | 340 M | 2.86 | 13.54     | 500K steps  |
| TinyDiT-D14 (KD) | 14    | 340 M | 2.86 | 13.54     | 500K steps  |

놀라운 점은 원래 학습 비용의 7%만 사용해서 거의 비슷한 성능을 달성했다는 것이다. 속도는 2배 향상 되었고, FID는 2.27에서 2.86으로 0.59만 증가했다.
![[Image/Pasted image 20250710120017.png]]
### 다른 아키텍처 결과
#### MAR 모델
- MAR-Large (32 layers) -> tinyMAR-D16 (16 layers)
- FID: 1.78 -> 2.28
- 학습 비용: 400 epochs -> 40 epochs (10%)
#### SiT 모델
- SiT-XL/2 (28 layers) -> TinySiT-D14 (14 layers)
- FID: 2.06 -> 3.02
- 학습 비용: 1,400 epochs -> 100 epochs (7%)
## 핵심 분석 실험
### Calibration Loss vs Recoverability
연구진은 100,000개의 random pruning 모델을 생성해서 calibration loss와 fine-tuning 후 성능의 관계를 분석했다.

| 방법          | Calibration Loss | FID (after fine-tuning) |
| ----------- | ---------------- | ----------------------- |
| Min Loss    | 0.20             | 20.69                   |
| Median Loss | 0.99             | 6.45                    |
| Short GPT   | 0.20             | 22.28                   |
| TinyFusion  | 0.98             | 5.73                    |
이 결과는 calibration loss가 낮다고 해서 fine-tuning 후 성능이 좋은 것은 아니라는 것을 보여준다. 오히려 TinyFusion처럼 중간 정도의 calibration loss를 가진 모델이 가장 좋은 recoverability를 보였다.
![[Image/Pasted image 20250710120036.png]]
### Local Block 크기의 영향
다양한 N:M scheme을 실험했다.

| Pattern | FID   | IS    | 특징                      |
| ------- | ----- | ----- | ----------------------- |
| 1:2     | 33.39 | 54.75 | 작은 search space, 안정적 학습 |
| 2:4     | 34.21 | 53.07 | 균형잡힌 성능                 |
| 7:14    | 49.41 | 34.97 | 큰 search space, 최적화 어려움 |
1:2나 2:4 scheme이 가장 실용적인 것으로 나타났다.
![[Image/Pasted image 20250710120045.png]]
## Knowledge Distillation 향상
### Massive Activation 문제
Diffusion Transformer의 hidden state에는 매우 큰 값을 가진 activation들이 존재한다. 이런 값들이 knowledge distillation을 방해한다. 연구진은 이를 해결하기 위해 **Masked RepKD**를 제안했다.
### Masked Knowledge Distillation
다음 조건을 만족하는 activation만 distillation에 사용한다.
$$
|x-\mu_x| < k \sigma_x
$$
여기서 $k$는 threshold parameter다.

| 방법        | $k$ 값     | FID         |
| --------- | --------- | ----------- |
| 일반 RepKD  | -         | NaN (학습 실패) |
| Masked KD | $2\sigma$ | 3.73        |
| Masked KD | $4\sigma$ | 3.73        |
Masked KD를 사용하면 FID가 5.73에서 3.73으로 크게 개선된다.
![[Image/Pasted image 20250710120055.png]]
# Conclusions
## 주요 기여와 혁신
이 논문의 가장 중요한 기여는 diffusion transformer에서 recoverability가 calibration loss보다 중요하다는 것을 증명한 것이다. 이는 기존의 pruning 패러다임을 완전히 바꾸는 발견이다.
**TinyFusion**은 이런 insight를 바탕으로 개발된 최초의 learnable depth pruning 방법이다. Differentiable sampling과 joint optimization을 통해 pruning과 fine-tuning을 하나의 최적화 문제로 통합했다.
**Masked Knowledge Distillation**은 diffusion transformer의 특성을 고려한 새로운 distillation 방법이다. Massive activation 문제를 해결해서 더 안정적인 학습을 가능하게 했다.
## 실용적 의미와 파급효과
이 연구는 실제 산업 응용에 큰 의미를 가진다:
1. **배포 효율성**: 원래 모델의 7% 비용으로 2배 빠른 모델을 만들 수 있다
2. **범용성**: DiT, MAR, SiT 등 다양한 아키텍처에 적용 가능하다
3. **실용성**: 일반 사용자도 고품질 이미지 생성 모델을 사용할 수 있게 된다
## 한계점과 향후 연구 방향
현재 연구는 conditional image generation에 집중되어 있다. 
- **Text-to-Image generation**: DALL-E, Midjourney 같은 텍스트 기반 이미지 생성에 적용
- **Fine-grained pruning**: Attention과 MLP를 별도로 제거하는 더 세밀한 전략
- **Other modalities**: 비디오, 오디오 등 다른 도메인으로의 확장