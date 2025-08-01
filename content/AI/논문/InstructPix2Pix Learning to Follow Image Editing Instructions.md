---
{"publish":true,"created":"2025-07-11","modified":"2025-08-01T11:47:56.321+09:00","tags":["ai","article","computer_vision"],"cssclasses":""}
---

# Introduction
본 논문은 이미지 편집을 위해 사람이 작성한 명령을 따르도록 생성 모델을 가르치는 방법을 제시한다. 이 task를 위한 학습 데이터는 대규모로 획득하기 어렵기 때문에 서로 다른 modality로 사전 학습된 두 개의 대형 모델--언어 모델(GPT-3)과 text-to-image 모델(Stable Diffusion)--을 결합하여 쌍으로 된 데이터셋을 생성하는 접근 방식을 제안한다.
기존의 이미지 편집 방법들은 다음과 같은 한계를 가지고 있었다.
- 전체 이미지 설명이나 마스크 등 복잡한 입력이 필요
- 단일 편집 작업에만 특화 (스타일 변환만, 객체 제거만 등)
- 각 이미지마다 개별적인 fine-tuning이나 inversion 과정이 필요
InstructPix2Pix는 이러한 문제들을 해결하여 `Replace the fruits with cake`, `Add fireworks to the sky`와 같은 자연스러운 명령어만으로 이미지를 편집할 수 있다. 모델은 forward pass에서 이미지 편집을 직접 수행하며 추가 예제 이미지, 입력/출력 이미지에 대한 전체 설명 또는 예제별 fine-tuning이 필요하지 않다.
합성 예제에 대해 전적으로 학습을 받았음에도 불구하고 모델은 임의의 실제 이미지와 자연적인 사람이 작성한 명령 모두에 대해 zero-shot 일반화를 달성한다. 모델은 사람의 명령에 따라 객체 교체, 이미지 스타일 변경, 설정 변경, 예술적 매체 등 다양한 편집들을 수행할 수 있는 직관적인 이미지 편집이 가능하다.
![[Image/Pasted image 20250711162812.png]]
# Related Work
## 대형 사전학습 모델의 조합
최근 몇 년간 서로 다른 modality에 특화된 큰 AI 모델들을 조합해서 새로운 능력을 만드는 연구가 활발해졌다. 예를 들어, 언어 모델과 시각 모델을 결합해서 이미지를 보고 설명하거나, 질문에 답하는 시스템들이 개발되었다. 이 논문도 이런 접근법의 연장선에 있지만, 차이점은 모델들을 직접 결합하는 대신 이들을 이용해 학습 데이터를 생성한다는 점이다
## Diffusion Model의 발전
Diffusion Model은 최근 이미지 생성 분야에서 가장 뛰어난 성과를 보이는 기술이다. 이 모델들은 깨끗한 이미지에 점진적으로 노이즈를 추가하는 과정을 학습한 다음, 역방향으로 노이즈를 제거해서 새로운 이미지를 생성한다. DALL-E 2, Midjourney, Stable Diffusion 등이 모두 이 기술을 기반으로 한다.
# Method
## Generating a Multi-Modal Training Dataset
서로 다른 modality에서 작동하는 두 개의 대규모 사전 학습된 모델의 능력을 결합하여 텍스트 편집 명령과 편집 전후의 해당 이미지를 포함하는 multi-modal 학습 데이터셋을 생성한다. 전체 데이터 생성 과정은 두 단계로 구성된다.
1. **텍스트 편집 생성**: GPT-3를 사용하여 `(입력 캡션, 편집 명령, 출력 캡션)` triplet 생성
2. **이미지 쌍 생성**: Stable Diffusion + Prompt-to-Prompt를 사용하여 캡션 쌍을 이미지 쌍으로 변환
### Generating Instructions and Paired Captions
먼저 텍스트 도메인에서 대규모 언어 모델을 활용하여 이미지 캡션을 가져오고 편집 명령과 편집 후 결과 텍스트 캡션을 생성한다. 예를 들어, 입력 캡션 `photograph of a girl riding a horse`이 제공되면 언어 모델은 `have her ride a dragon`라는 그럴듯한 편집 명령과 적절하게 수정된 출력 캡션 `photograph of a girl riding a dragon`을 모두 생성할 수 있다.
![[Image/Pasted image 20250711162833.png]]
#### GPT-3 Fine-Tuning 과정
- LAION-Aesthetics V2 6.5+ 데이터셋에서 700개의 입력 캡션을 샘플링
- 각 캡션에 대해 수동으로 편집 명령과 출력 캡션을 작성
- 이 데이터로 GPT-3 Davinci 모델을 단일 epoch 동안 fine-tuning
- Fine-tuning된 모델로 454,455개의 편집 예제를 생성
#### 데이터 형식
```
입력 캡션: "Yefim Volkov, Misty Morning"
편집 명령: "make it afternoon"
출력 캡션: "Yefim Volkov, Misty Afternoon"
```
LAION 데이터셋을 선택한 이유는 큰 크기, 콘텐츠의 다양성(사진, 그림, 디지털 아트워크), 그리고 고유명사와 대중문화 참조를 포함한 풍부한 내용 때문이다. LAION의 잠재적인 단점은 상당히 noisy하고 무의미하거나 설명이 없는 캡션이 많이 포함되어 있다는 거지만, 이는 dataset filtering과 classifier-free guidance를 통해 완화되었다.
![[Image/Pasted image 20250711162903.png]]
### Generating Paired Images from Paired Captions
캡션 쌍을 이미지 쌍으로 변환하는 과정에서 가장 큰 문제는 text-to-image 모델이 조건 프롬프트의 아주 작은 변경에서도 이미지 일관성에 대한 보장을 제공하지 않는다는 것이다. 예를 들어 `a picture of a cat`과 `a picture of a black cat`이라는 매우 유사한 두 개의 프롬프트는 완전히 다른 고양이 이미지를 생성할 수 있다.
#### Prompt-to-Prompt 기술 적용
이 문제를 해결하기 위해 text-to-image diffusion model에서 여러 생성 결과가 유사하도록 장려하는 Prompt-to-Prompt 기법을 사용한다. 이는 몇 가지 denoising step에서 cross-attention 가중치를 공유함으로써 수행된다.
#### 최적화된 생성 과정
- 캡션 쌍당 100개의 샘플 이미지 쌍을 생성
- 각각 임의의 $p\sim \mathcal{U}(0.1,0.9)$ 값 사용 ($p$는 attention 가중치를 공유할 denoising step의 비율)
- CLIP 기반 metric으로 샘플 필터링:
	- 이미지-이미지 CLIP 유사도 > 0.75
	- 이미지-캡션 CLIP 유사도 > 0.2
	- Directional CLIP similarity > 0.2
![[Image/Pasted image 20250711162919.png]]
## InstructPix2Pix
생성된 학습 데이터를 사용하여 명령으로부터 이미지를 편집하는 조건부 diffusion model을 학습시킨다. 대규모 text-to-image latent diffusion model인 Stable Diffusion을 기반으로 한다.
### 기본 구조
#### Latent Diffusion 원리
이미지 $x$에 대해 VAE 인코더 $\mathcal{E}$와 디코더 $\mathcal{D}$를 사용하여 latent space에서 작동한다. Diffusion process는 임코딩된 latent $z=\mathcal{E}(x)$에 noise를 추가하여 timestep $t \in T$에 걸쳐 noise level이 증가하는 noisy latent $z_t$를 생성한다.
#### 손실 함수
이미지 조건 $c_I$와 텍스트 명령 조건 $c_T$가 주어지면 noisy latent $z_t$에 추가된 noise를 예측하는 네트워크 $\epsilon_\theta$를 학습한다.
$$
L = \mathbb{E}_{\mathcal{E}(x), \mathcal{E}(c_I), c_T, \epsilon \sim \mathcal{N}(0,1), t} [\| \epsilon - \epsilon_\theta (z_t, t, \mathcal{E}(c_I), c_T) \|_2^2]
$$
#### 모델 수정사항
- **이미지 conditioning 지원**: 첫 번째 convolution layer에 추가 입력 채널을 추가하여 $z_t$와 $\mathcal{E}(c_I)$를 concatenate
- **가중치 초기화**: 사전 학습된 Stable Diffusion 체크포인트로 초기화, 새로 추가된 입력 채널 가중치는 0으로 초기화
- **텍스트 conditioning**: 원래 캡션용 텍스트 conditioning 메커니즘을 재사용하여 편집 명령 $c_T$를 입력으로 사용
## Classifier-free Guidance for Two Conditionings
Classifier-free diffusion guidance는 diffusion model에 의해 생성된 샘플의 품질과 다양성을 절충하는 방법이다. 일반적인 단일 조건에 대한 classifier-free guidance는 다음과 같다.
$$
\tilde{\epsilon_\theta}(z_t, c) = \epsilon_\theta(z_t, \emptyset) + s \cdot (\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset))
$$
### 이중 조건부 Guidance
본 연구의 score network $\epsilon_{\theta}(z_t,c_I,c_T)$는 입력 이미지 $c_I$와 텍스트 명령 $c_T$라는 두 가지 conditioning을 가진다. 두 conditioning에 대해 각각 독립적인 guidance scale을 도입한다.
$$
\tilde{\epsilon_\theta}(z_t, c_I, c_T) = \epsilon_\theta(z_t, \emptyset, \emptyset) + s_I \cdot (\epsilon_\theta(z_t, c_I, \emptyset) - \epsilon_\theta(z_t, \emptyset, \emptyset)) + s_T \cdot (\epsilon_\theta(z_t, c_I, c_T) - \epsilon_\theta(z_t, c_I, \emptyset))
$$
여기서
- $s_I$: 입력 이미지와의 일관성을 조절하는 guidance scale
- $s_T$: 텍스트 명령어 준수 정도를 조절하는 guidance scale
### 학습 전략
학습하는 동안 다음과 같이 conditioning을 랜덤하게 설정한다.
- 5%: $c_I=\emptyset_I$ (이미지 조건만 제거)
- 5%: $c_T=\emptyset_T$ (텍스트 조건만 제거)
- 5%: $c_I=\emptyset_I, c_T=\emptyset_T$ (둘 다 제거)
- 85%: 둘 다 사용
이를 통해 모델은 조건부 입력 둘 다 또는 둘 중 하나에 대해 conditional 또는 unconditional denoising이 가능하다.
# Results
## 다양한 편집 결과
### 예술 작품 변환
#### 모나리자 변환 실험
- `Make it a Modigliani painting`: 모딜리아니의 특징적인 긴 얼굴과 스타일로 변환
- `Make it a Miro painting`: 미로의 추상적이고 컬러풀한 스타일 적용
- `Make it an Egyptian sculpture`: 이집트 조각상의 정면성과 양식화된 특징 구현
- `Make it a marble Roman sculpture`: 고전 로마 조각의 사실적 표현과 대리석 질감
![[Image/Pasted image 20250711162951.png]]
#### 미켈란젤로의 천지창조 변환
- `Put them in outer space`: 우주 배경으로 변경하면서 핵심 구성 유지
- `Turn the humans into robots`: 인물들을 로봇으로 변환하면서 동적 에너지 보존
![[Image/Pasted image 20250711162956.png]]
### 현대 문화 아이콘 편집
#### 비틀즈 Abbey Road 앨범 커버
- **지리적 변환**: 파리, 홍콩, 맨하탄, 프라하 등 다양한 도시 배경으로 변환
- **시간적 변환**: `Make it evening` - 저녁 분위기로 변경
- **활동 변환**: `Put them on roller skates` - 롤러스케이트 착용
- **스타일 변환**: `Make it Minecraft`, `Make it a Claymation` 등
![[Pasted image 20250711163002.png]]
### 반복적 편집 (Iterative Editing)
하나의 이미지에 여러 명령을 순차적으로 적용할 수 있다.
1. `Insert a train`
2. `Add an eerie thunderstorm`
3. `Turn into an oil pastel drawing`
4. `Give it a dark creepy vibe`
각 단계에서 이전 편집 결과를 바탕으로 새로운 편집이 자연스럽게 적용된다.
![[Image/Pasted image 20250711163015.png]]
### 다양한 결과 생성
동일한 입력 이미지와 명령에 대해 latent noise를 변경하여 여러 가지 다른 편집 결과를 생성할 수 있다. 예를 들어 `in a race car video game` 명령으로 다양한 스타일의 레이싱 게임 그래픽을 만들 수 있다.
![[Image/Pasted image 20250711163023.png]]
## Baseline Comparisons
### 정성적 비교
#### SDEdit 대비 장점
- 더 정확한 객체 정체성 보존
- 격리된 편집 가능 (원하는 부분만 변경)
- 전체 이미지 설명 대신 간단한 편집 명령만 필요
![[Image/Pasted image 20250711163044.png]]
#### Text2Live 대비 장점
- 더 광범위한 편집 유형 지원
- 객체 교체나 구조적 변경 가능 (Text2Live는 additive layer로 제한)
### 정량적 비교
#### CLIP Image Similarity
원본과 편집본 간의 일관성
#### CLIP Text-Image Direction Similarity
편집 방향과 텍스트 명령 간의 일치도
결과적으로 InstructPix2Pix는 동일한 directional similarity에서 SDEdity보다 현저히 높은 image consistency를 보여줬다.
## Ablation Studies
### 데이터셋 크기의 영향
- **전체 데이터 (454,455개)**: 모든 유형의 편집에서 우수한 성능
- **10% 데이터**: 큰 구조적 편집 능력 감소, 주로 스타일 변경만 가능
- **1% 데이터**: 매우 제한적인 편집 능력
### CLIP 필터링의 효과
CLIP 기반 필터링을 제거했을 때 전체적인 이미지 일관성이 감소하여 고품질 학습 데이터 선별의 중요성을 입증했다.
### Classifier-free Guidance 매개변수 분석
- **$s_T$ 증가**: 편집 명령에 더 강하게 반응하지만 과도한 편집 위험
- **$s_I$ 증가**: 원본 이미지와의 일관성 증가, 공간적 구조 보존
최적 범위: $s_T \in [5,10], s_I \in [1.0,1.5]$
## Limitations
1. **기반 모델의 의존성**: Stable Diffusion의 품질에 제한됨
2. **공간적 추론 부족**: `move it to the left`, `swap their positions` 같은 명령 처리 어려움
3. **개수 세기**: 객체 수를 정확히 조절하는 편집의 한계
4. **편향 문제**: 학습 데이터와 사전학습 모델의 편향이 결과에 반영될 수 있음
### 실패 사례들
- 시점 변경 (`Zoom into the image`)
- 과도한 변경 (원하는 것보다 더 많은 부분이 변경됨)
- 객체 분리 실패 (`Color the tie blue`에서 다른 부분도 변경됨)
- 위치 교환 (`Have the people swap places`)
![[Image/Pasted image 20250711163113.png]]
# Conclusion
InstructPix2Pix는 자연어 명령을 통한 직관적인 이미지 편집이라는 새로운 패러다임을 제시했다. 두 개의 대형 사전학습 모델을 창의적으로 결합하여 학습 데이터 부족 문제를 해결하고, 다양한 편집 작업을 단일 모델로 수행할 수 있게 했다.
## 주요 기여
1. 대규모 multi-modal 학습 데이터 자동 생성 방법론
2. 이중 조건부 classifier-free guidance 메커니즘
3. 실시간에 가까운 속도의 고품질 이미지 편집
4. 합성 데이터로 학습했음에도 실제 이미지에 대한 우수한 일반화 성능
향후 연구에서는 공간적 추론 능력 향상, 편향 완화, 그리고 human feedback을 통한 모델 개선 등이 중요한 과제가 될 것이다. 이 연구는 AI와 인간의 창의적 협업에서 중요한 이정표를 제시했으며, 이미지 편집 도구의 민주화에 기여할 것으로 기대된다.