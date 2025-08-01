---
{"publish":true,"aliases":"Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps","created":"2025-07-01","modified":"2025-07-31T16:25:20.020+09:00","tags":["computer_vision","ai","article","diffusion","text-to-image"],"cssclasses":""}
---

# Introduction
최근 몇 년간 text-to-image 확산 모델이 눈부신 발전을 이루어 텍스트 프롬프트만으로도 고품질 이미지를 생성할 수 있게 되었다. 하지만 여전히 생성된 이미지가 텍스트의 의미를 정확히 반영하지 못하는 경우가 많다.
가장 대표적인 문제는 두 가지가 있다.
## Missing Objects (객체 누락)
`a black car and a white clock`라는 프롬프트에서 자동차나 시계 중 하나가 아예 생성되지 않는 경우이다. 이는 두 객체의 cross-attention map이 너무 겹쳐서 하나의 객체로 인식되기 때문이다.
## Attribute Mis-Binding (속성 오결합)
`a black car and a white clock`에서 자동차가 흰색으로 나오거나 시계가 검은색으로 나오는 경우이다. 이는 속성(색상)과 객체 간의 cross-attention map이 제대로 정렬되지 않아서 발생한다.
![[Image/Pasted image 20250701134827.png]]
이 논문의 핵심 아이디어는 **cross-attention map**에 주목하는 것이다. Cross-attention map은 각 텍스트 토큰이 이미지의 어느 부분에 영향을 미치는지를 보여주는 일종의 가중치 맵이다. 예를 들어 `black`라는 단어의 cross-attention map은 검은색 영역에 높은 값을 가져야 하고, `car`의 map은 자동차 모양 영역에 높은 값을 가져야 한다.
연구진은 문제의 근본 원인을 두 가지로 파악했다:
- Text embedding의 유사성이 cross-attention map의 유사성을 강하게 결정한다
- 하지만 text embedding 자체는 단어 간의 문법적 관계를 제대로 담지 못한다
이를 해결하기 위해 text encoder 내부의 **self-attention map**을 활용하는 새로운 방법을 제안한다. Self-attention map은 문법적 관계를 잘 포착하지만 지금까지 활용되지 못했던 정보의 보고다.
# Related Work
## Text-to-Image Generation
현재 주류는 **latent diffusion model**이다. 대표적으로 Stable Diffusion이 있으며, 이들은 CLIP이나 T5같은 text encoder를 사용해서 텍스트를 처리한다. 텍스트 데이터는 tokenization -> embedding -> multi-head self-attention 과정을 거쳐 처리된다.
## Cross-Attention Control Methods
기존 연구들은 대부분 외부 정보에 의존해서 cross-attention을 조정했다:
- **CONFORM**은 사람이 직접 긍정/부정 그룹을 만들어야 한다. 예를 들어 `red apple and blue ball`에서 (`red`, `apple`), (`blue`, `ball`)을 긍정 쌍으로, (`red`, `ball`), (`blue`, `apple`)를 부정 쌍으로 수동으로 지정해야 한다.
- **Linguistic Binding**은 SpaCy 같은 외부 텍스트 파서를 사용해서 (modifier, entity-noun) 쌍을 자동을 찾는다. 하지만 파서의 성능에 의존적이고 복잡한 문장 구조에서는 한계가 있다.
- **Attend-and-Excite**는 특정 토큰들의 attention을 강화하는 방법이지만, 역시 사람이 수동으로 토큰을 선택해야 한다.
## Attention Sinks
Transformer 모델에서 `<bos>`, `<eos>` 같은 특수 토큰에 attention이 과도하게 집중되는 현상이다. CLIP text encoder에서도 `<bos>` 토큰이 전체 attention의 약 20배를 차지한다. 이로 인해 다른 토큰 간의 관계 정보가 희석된다. 
# Preliminaries
## Text Encoder (텍스트 인코더)
Text Encoder는 우리가 입력하는 텍스트 프롬프트를 컴퓨터가 이해할 수 있는 숫자 형태로 바꾸는 역할을 한다. 마치 번역기처럼 `검은 자동차`라는 한국어를 컴퓨터 언어로 번역하는 것이다.
### 과정
1. **Tokenization**: `a black car`를 [`a`, `black`, `car`] 같은 토큰들로 나눈다
2. **Embedding**: 각 토큰을 고차원 벡터로 변환한다
3. **Self-attention**: 토큰들 사이의 관계를 파악한다
Text encoder의 $\ell$번째 layer에서 각 토큰은 key vector $e_i^{(\ell)} \in \mathbb{R}^{H_eD_e}$로 표현된다. 여기서
- $s$: 텍스트 시퀀스 길이 (토큰 개수)
- $D_e$: head당 embedding 차원
- $H_e$: text encoder의 head 개수
Self-attention matrix는 다음과 같이 계산된다:
$$
T_{ij}^{(\ell,h)} = \frac{\exp(\omega_{ij})}{\sum_k \exp(\omega_{ik})}, \quad \omega_{ij} = e_i^{(\ell)T} W_{en}^{(\ell,h)} e_j^{(\ell)}​
$$

여기서 $W_{en}^{(\ell,h)} \in \mathbb{R}^{H_eD_e \times H_e D_e}$는 학습된 가중치 행렬이다.
Self-attention에서 각 토큰이 다른 토큰들에게 얼마나 주목하는지가 결정된다. `black car`에서 `black`이 `car`에게 높은 attention을 주면, 이 둘이 관련있다는 것을 의미한다.
최종적으로 모든 layer와 head에 대해 평균을 내서 하나의 self-attention matrix를 만든다:
$$
T' = \frac{1}{L_e H_e} \sum_{\ell=1}^{L_e} \sum_{h=1}^{H_e} T^{(\ell,h)}
$$
그리고 `<bos>`와 `<eos>` 같은 특수 토큰을 제거하고 정규화한다:
$$
T_{ij} = \frac{T'_{ij}}{\sum_{m=2}^s T'_{im}}
$$
Text encoder의 최종 출력은 각 토큰에 대한 embedding vector $k_i \in \mathbb{R}^{H_e D_e}$​이다. 이것이 다음 단계인 diffusion model에 조건 정보로 전달된다.
## Denoising Latent Variables (잡음 제거 과정)
Latent Diffusion Model은 이미지를 직접 생성하는 대신 latent space라는 압축된 공간에서 작업한다. 이는 계산 효율성을 높이기 위함이다.
### 과정
1. 순수한 가우시안 노이즈 $z_\tau$에서 시작한다
2. 매 step마다 조금씩 노이즈를 제거한다
3. 최종적으로 깨끗한 latent representation $z_0$를 얻는다
4. 이를 실제 이미지로 디코딩한다
수학적으로는 다음과 같이 표현된다:
$$
z_{t-1} = z_t - D_\theta(z_t; \{k_i\}), \quad 0 < t \leq \tau
$$
여기서:
- $z_t$: $t$ 시점의 latent variable
- $D_\theta$: 학습된 denoising network (U-Net)
- $\{k_i\}$: text encoder에서 온 conditioning 정보
- $\tau$: 전체 timestep 수 (보통 1000)
직관적 설명: 마치 안개 낀 사진에서 점검 안개를 걷어내서 선명한 사진을 만드는 과정과 비슷한다. 단, 여기서는 텍스트 정보 $\{k_i\}$가 "어떤 종류의 이미지를 만들어야 하는지" 가이드 역할을 한다.
## Cross-Attention Module (교차 주의 모듈)
Cross-Attention은 텍스트와 이미지 정보가 만나는 핵심 지점이다. 여기서 "이 텍스트 토큰이 이미지의 어느 부분에 영향을 줄지"가 결정된다.
### 구조
- Query: 이미지의 각 픽셀 위치에서 나오는 벡터 $q_a^{(\ell)} \in \mathbb{R}^{H_c D_c}$
- Key: 텍스트의 각 토큰에서 나오는 벡터 $k_i^{(\ell)}$​ (text encoder 출력)
- Value: 역시 텍스트 토큰에서 나옴
Cross-attention map은 다음과 같이 계산된다:
$$
A_{ai}^{(\ell,h)} = \frac{\exp(\Omega_{ai})}{\sum_{j=1}^s \exp(\Omega_{aj})}, \quad \Omega_{ai} = q_a^{(\ell)T} W_c^{(\ell,h)} k_i^{(\ell)}​
$$
여기서:
- $a = 1, \cdots, N_c$​: 이미지의 공간적 위치 인덱스
- $i = 1, \cdots, s$: 텍스트 토큰 인덱스
- $W_c^{(\ell,h)} \in \mathbb{R}^{H_c D_c \times H_c D_c}$​: 학습된 projection matrix
### 물리적 의미
$A_{ai}^{(\ell,h)}$는 이미지의 $a$ 위치가 텍스트의 $i$번째 토큰에 얼마나 주목하는지를 나타낸다. 예를 들어 `red car`에서 이미지의 자동차 영역이 `red`와 `car` 토큰에 높은 attention을 가져야 한다. 
실제 사용할 때는 여러 layer과 head에 대해 평균을 낸다:
$$
A = \frac{1}{L_M H_c} \sum_{\ell=1}^{L_M} \sum_{h=1}^{H_c} A^{(\ell,h)}
$$
여기서 $L_M$​은 $N_c = M$인 cross-attention layer의 개수다.
## Cross-Attention Similarity Matrix (교차 주의 유사도 정렬)
서로 다른 토큰들의 cross-attention map이 얼마나 비슷한지 측정한다.
### 계산 과정
1. 먼저 토큰 $i$와 $j$의 cross-attention map 간 cosine similarity를 구한다:
$$
C_{ij} = \frac{\sum_{a=1}^{N_c} A_{ai} A_{aj}}{\sqrt{\sum_{a=1}^{N_c} A_{ai}^2} \sqrt{\sum_{a=1}^{N_c} A_{aj}^2}}
$$
2. 이를 정규화해서 similarity matrix $S$를 만든다:
$$
S_{ij} = \frac{C_{ij}}{\sum_{k=1}^s C_{ik}}
$$
### 직관적 의미
- $S_{ij}$​가 크다 = 토큰 $i$와 $j$가 이미지에서 비슷한 영역에 영향을 준다
- $S_{ij}$​가 작다 = 토큰 $i$와 $j$가 이미지에서 다른 영역에 영향을 준다
### 예시
`black car and white clock`
- `black`와 `car`의 $S_{ij}$는 커야 한다 (같은 객체를 가리킴)
- `car`와 `clock`의 $S_{ij}$는 작아야 한다 (다른 객체들)
- `black`와 `clock`의 $S_{ij}$는 작아야 한다 (관계없는 속성과 객체)
이 similarity matrix가 텍스트의 문법적 관계를 제대로 반영하지 못할 때 missing objects나 attribute mis-binding 같은 문제가 발생한다. 따라서 이를 text self-attention matrix와 맞춰주는 것이 이 논문의 핵심 아이디어다.
# Understanding and Resolving Text-to-Image Semantic Discrepancy
## Why do Generated Images Misrepresent Text?
연구진은 체계적인 실험을 통해 세 가지 핵심 발견을 했다.
![[Image/Pasted image 20250701152032.png]]
### Finding 1: Text Embedding 유사성과 Cross-Attention 유사성의 강한 상관관계
Cross-attention map $A^{(\ell,h)} \in \mathbb{R}^{N_c \times s}$에서 토큰 $i$와 $j$의 cosine similarity는 다음과 같이 정의된다:
$$
C_{ij} = \frac{\sum_{a=1}^{N_c} A_{ai}A_{aj}}{\sqrt{\sum_{a=1}^{N_c} A_{ai}^2} \sqrt{\sum_{a=1}^{N_c} A_{aj}^2}}​​
$$
실험 결과, text embedding $k_i$​와 $k_j$​의 cosine similarity와 $C_{ij}$​ 사이에 강한 양의 상관관계가 있다는 것을 발견했다. 이는 denoising 초기 단계부터 마지막까지 일관되게 나타났다.
수학적으로도 이를 증명할 수 있다. 특정 가정 하에서 cross-attention map의 유사성은 다음과 같이 근사할 수 있다:
$$
\cos(A_i^{(\ell,h)}, A_j^{(\ell,h)}) = \exp\left(-\frac{1}{2}(k_i - k_j)^T W^2 (k_i - k_j)\right)
$$
여기서 $W^2 = W_c^{(\ell,h)T} \Sigma^{(\ell)} W_c^{(\ell,h)}$​이고, $\Sigma^{(\ell)}$은 query vector들의 공분산 행렬이다.
![[Image/Pasted image 20250701152106.png]]
### Finding 2: Text Embedding vs Self-Attention의 문법 정보 격차
연구진은 `[attribute1] [object1] and [attribute2] [object2]` 형태의 프롬프트들을 분석했다. 문법적으로 연결된 토큰들(예: `black`-`car`, `white`-`clock`)과 연결되지 않은 토큰들(예: `car`-`clock`, `black`-`clock`)의 text embedding 유사성을 비교했다.
두 그룹의 embedding 유사성 분포에는 유의미한 차이가 없었다. 즉, text embedding만으로는 문법적 관계를 구별할 수 없다는 뜻이다.
반면 text self-attention matrix $T$에는 명확한 차이가 나타났다. Self-attention은 다음과 같이 계산된다:
$$
T_{ij}^{(\ell,h)} = \frac{\exp(\omega_{ij})}{\sum_k \exp(\omega_{ik})}, \quad \omega_{ij} = e_i^{(\ell)T} W_{en}^{(\ell,h)} e_j^{(\ell)}​
$$
여러 layer와 head에 대해 평균을 내고 특수 토큰을 제거하면:
$$
T_{ij} = \frac{T'_{ij}}{\sum_{m=2}^s T'_{im}}
$$
이렇게 얻은 $T$에서는 문법적으로 연결된 토큰 쌍들이 더 높은 attention 값을 보였다.
![[Image/Pasted image 20250701152255.png]]
### Finding 3: Attention Sink가 정보 전달을 방해
왜 self-attention의 문법 정보가 text embedding에 반영되지 않을까? 연구진은 **attention sink** 현상에 주목했다. `<bos>` 토큰이 전체 attention의 대부분을 차지하면서 다른 토큰 간의 상호작용이 약해진다.
수학적으로, attention sink 상황에서는 다음이 성립한다:
$$
\epsilon = \frac{\sum_{j \neq 1} T_{ij}}{T_{i1}} \ll 1
$$
이 경우 self-attention layer의 출력은:
$$
o_i^{(\ell,h)} = \sum_{j=1}^s T_{ij}^{(\ell,h)} W_v^{(\ell,h)} e_j^{(\ell)}​
$$
Attention sink 조건 하에서 서로 다른 토큰들의 출력 간 cosine similarity는:
$$
\cos(o_i^{(\ell,h)}, o_j^{(\ell,h)}) = 1 - O(\epsilon)
$$
즉, 모든 토큰의 출력이 거의 비슷해져서 개별 토큰 간의 관계 정보가 사라진다.
![[Image/Pasted image 20250701152222.png]]
## Text Self-Attention Maps (T-SAM) Guidance
이 문제를 해결하기 위해 T-SAM 방법을 제안했다. 핵심 아이디어는 text self-attention map의 문법 정보를 cross-attention에 직접 전달하는 것이다.
### 알고리즘
1. Text encoder에서 self-attention matrix $T$를 추출한다
2. Cross-attention map들이 similarity matrix $S$를 계산한다
3. $S$와 $T$ 사이의 거리를 최소화하도록 latent noise $z_t$를 최적화한다
손실 함수는 다음과 같다:
$$
L(z_t) = \sum_{i=1,j \leq i}^s \rho_i |T_{ij}^{\gamma} - S_{ij}(z_t)|
$$
여기서:
- $\rho_i=i/s$는 가중치
- $\gamma = 4$는 temperature parameter로 큰 값을 증폭하고 작은 값을 압축한다
- $S_{ij}$는 cross-attention similarity matrix
최적화는 다음과 같이 수행한다:
$$
z'_t = z_t - \alpha \cdot \nabla_{z_t} L(z_t)
$$
여기서 $\alpha$는 학습률이다. 이 과정을 denoising step 1-25에서 수행한다.
#### 직관적 설명
만약 `black`과 `car`가 문법적으로 연결되어 있다면 ($T_{ij}$가 크다면), 이들의 cross-attention map도 비슷한 영역을 가리켜야 한다 ($S_{ij}$도 커야 한다). 반대로 `car`과 `clock`처럼 독립적인 객체들은 서로 다른 영역을 가리켜야 한다.
# Experiments
## 실험 설정
### 데이터셋
- **TIFA v1.0**: 4,000개의 다양한 프롬프트 (COCO, DrawBench, PartiPrompt, PaintSkill)
- **Attend-n-Excite prompts**: 구조화된 템플릿 (`attribute1] [object1] and [attribute2] [object2]`)
### 구현 세부사항
- Stable Diffusion v1.5 기반
- 50 sampling iteration, step 1-25에서 최적화
- TIFA: $\alpha =40$, Attend-n-Excite: $\alpha=10$
- Cross-attention map은 Gaussian smooting 적용
### 평가 지표
- TIFA 점수: GPT-3.5가 생성한 질문에 대해 vision-language 모델이 답하는 방식
- CLIP 유사도: Fully prompt similarity, Minimum object similarity, Prompt-caption similarity
## 실험 결과
### 정량적 결과

| 방법    | External Info. | TIFA     | CLIP-I   | CLIP-T |
| ----- | -------------- | -------- | -------- | ------ |
| SD    | ❌              | 0.79     | 0.33     | 0.77   |
| LB    | ✅              | 0.80     | 0.33     | 0.76   |
| T-SAM | ❌              | **0.83** | **0.34** | 0.77   |
T-SAM이 외부 정보 없이도 가장 높은 성능을 달성했다. 특히 색상, 모양, 개수, 활동 등 대부분 카테고리에서 개선을 보였다.
![[Image/Pasted image 20250701152400.png]]
### 정성적 결과
복잡한 MSCOCO 캡션에서도 누락되기 쉬운 객체들을 성공적으로 생성했다.
- `A small room with a futon couch, a sewing machine on a table, and a flatscreen TV.`: `sewing machine`을 정확히 생성
- `Entire front yard is filled with snow while people walk around.`: `people`을 포함해서 생성
- `Cow being fed a popsicle by a person over a fence.`: `popsicle`을 정확히 표현
구조화된 템플릿에서도 CONFORM과 비슷하거나 더 나은 성능을 보였으며, 특히 인위적인 분리 없이 자연스러운 이미지를 생성했다.
![[Image/Pasted image 20250701152422.png]]
![[Image/Pasted image 20250701152434.png]]
# Conclusion
이 연구는 text-to-image 확산 모델의 근본적인 문제를 해결하는 혁신적인 접근법을 제시했다. 핵심 기여는 다음과 같다:
1. 문제 진단: Text embedding의 유사성이 cross-attention map을 결정하지만, embedding 자체는 문법 정보가 부족하다는 것을 수학적, 실험적으로 증명했다
2. 해결책 발굴: Text encoder 내부의 self-attention map이 문법 정보를 잘 포착한다는 것을 발견하고, 이를 활용하는 방법을 개발했다
3. 실용적 방법: 외부 도구나 수동 작업 없이도 다양한 문장 구조에서 텍스트-이미지 정렬을 개선하는 T-SAM을 제안했다
이 방법의 가장 큰 장점은 self-contained라는 점이다. 기존 모델에 이미 존재하는 정보를 재활용하기 때문에 추가 비용이나 복잡성 없이 성능을 향상시킬 수 있다. 또한 다양한 문장 구조에 일반화 가능하다는 강력한 장점이 있다.