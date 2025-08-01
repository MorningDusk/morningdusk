---
{"publish":true,"created":"2025-06-22","modified":"2025-08-01T14:56:53.680+09:00","tags":["computer_vision","ai","article","6d"],"cssclasses":""}
---

# Introduction
기존 6D 객체 자세 추정 기술은 마치 특정 물건만 알아보는 로봇처럼 한계가 있었다. Instance-level 방법은 미리 학습한 특정 물건만 인식할 수 있고, Category-level 방법도 정해진 카테고리 안의 물건만 다룰 수 있었다. 하지만 실제 세상에서는 로봇이 처음 보는 완전히 새로운 물건도 정확히 집을 수 있어야 한다. 이것이 바로 Zero-shot 6D Object Pose Estimation이 해결하려는 문제다.
여기서 6D란 물건의 위치 3개(x,y,z)와 회전 3개(roll, pich, yaw)를 합친 총 6개의 정보를 의미한다. 마치 스마트폰을 책상 위 어디에 두었고 어떤 각도로 기울어져 있는지 정확히 파악하는 것과 같다.
최근 등장한 Segment Anything Model(SAM)은 마치 사진에서 모든 물건을 정확히 오려낼 수 있는 똑똑한 가위 역할을 한다. 연구팀은 이 SAM을 활용해 처음 보는 물건도 인식하고 그 정확한 자세까지 알아낼 수 있는 SAM-6D를 개발했다.
![[Image/Pasted image 20250623215340.png]]
# Related Work
## Segment Anything
SAM은 점, 박스, 텍스트, 마스크 등 다양한 힌트를 주면 사진에서 해당하는 물건을 정확히 잘라내는 AI다. 마치 "여기 컵이 있어"라고 가리키면 그 컵의 정확한 윤곽선을 찾아주는 것과 같다. SAM은 이미지 encoder, prompt encoder, mask decoder 세 부분으로 구성되어 있으며, 의료 이미지나 투명한 물건처럼 어려운 상황에서도 잘 작동한다.
## Pose Estimation of Novel Objects
기존 방법들은 크게 두 가지로 나뉜다. **이미지 매칭 방법**은 마치 다양한 각도에서 찍은 사진들과 비교해서 가장 비슷한 각도를 찾는 방식이다. **특징 매칭 방법**은 물건의 2D 픽셀이나 3D 점들을 서로 연결해서 자세를 계산하는 방식이다. 하지만 기존 방법들은 시간이 오래 걸리는 문제가 있었다.
# Methodology
## Instance Segmentation Model
ISM은 마치 물건 찾기 게임에서 숨어있는 특정 물건을 찾는 전문가 같은 역할을 한다. 먼저 SAM을 사용해 사진에서 모든 사용 가능한 물건 조각들을 잘라낸다. 그 다음 각 조각이 우리가 찾는 물건과 얼마나 비슷한지 세 가지 기준으로 점수를 매긴다.
**SAM 기본 동작**은 다음 공식으로 표현된다.
$$
\mathcal{M,C}=\Psi_{Mask}(\Phi_{Image}(I),\Phi_{Prompt}(\mathcal{P}_r))
$$
여기서 $\mathcal{M}$은 잘라낸 물건 조각들이고 $\mathcal{C}$는 각 조각의 신뢰도 점수다.
### 객체 매칭 점수
#### 의미적 매칭 점수<sub>Semantic Matching Score</sub>
"이게 정말 컵처럼 생겼나?"를 판단한다. 마치 물건의 전체적인 모양과 의미를 파악하는 것이다.
$$
s_{sem} = \frac{1}{K} \sum_{i=1}^{K} \text{top-K}\left\{\frac{\langle f^{cls}_{I_m}, f^{cls}_{T_k} \rangle}{\|f^{cls}_{I_m}\| \cdot \|f^{cls}_{T_k}\|}\right\}^{N_t}_{k=1}
$$
#### 외관 매칭 점수<sub>Appearance Matching Score</sub>
"색깔이나 무늬가 비슷한가?"를 확인한다. 마치 물건의 표면 질감과 색상을 자세히 살펴보는 것이다.
$$
s_{appe} = \frac{1}{N^{patch}_{I_m}} \sum_{j=1}^{N^{patch}_{I_m}} \max_{i \in \{1,\ldots,N^{patch}_{T_{best}}\}} \frac{\langle f^{patch}_{I_m,j}, f^{patch}_{T_{best},i} \rangle}{\|f^{patch}_{I_m,j}\| \cdot \|f^{patch}_{T_{best},i}\|}
$$
#### 기하학적 매칭 점수<sub>Geometric Matching Score</sub>
"크기나 모양이 맞나?"를 검사한다. 물건의 경계 상자를 비교해서 IoU 값으로 계산한다.
$$
s_{geo}=\frac{B_m \cap B_o}{B_m \cup B_o}
$$
최종 매칭 점수는 이 세 점수를 가중 평균한다.
$$
s_m=\frac{s_{sem}+s_{appe}+r_{vis} \cdot s_{geo}}{1+1+r_{vis}}
$$
## Pose Estimation Model
PEM은 마치 3D 퍼즐을 맞추는 전문가 같다. 사진에서 보이는 물건의 3D 점들과 컴퓨터가 알고 있는 실제 물건의 3D 점들을 서로 연결해서 물건이 어떻게 회전하고 이동했는지 계산한다.
### 배경 토큰의 혁신적 아이디어
실제 상황에서는 물건의 일부가 가려져 보이지 않는 경우가 많다. 기존 방법들은 이런 상황을 잘 처리하지 못했는데, 연구팀은 **배경 토큰**<sub>background token</sub>이라는 특별한 방법을 개발했다. 이는 매칭되지 않는 점들을 **배경**이라는 특별한 카테고리에 넣어서 처리하는 방식이다.
#### 주목 행렬<sub>Attention Matrix</sub> 계산
$$
\mathcal{A} = [\mathbf{f}^{\text{bg}}_m, \mathbf{F}_m] \times [\mathbf{f}^{\text{bg}}_o, \mathbf{F}_o]^T
$$
#### 소프트 할당 행렬<sub>Soft Assignment Matrix</sub>
$$
\tilde{\mathcal{A}} = \text{Softmax}_{\text{row}}(\mathcal{A}/\tau) \cdot \text{Softmax}_{\text{col}}(\mathcal{A}/\tau) \quad
$$
### 2단계 매칭 전략
PEM은 마치 망원경의 초점을 맞추는 것처럼 두 단계로 나누어 작업한다.
#### 거친 점 매칭<sub>Coarse Point Matching</sub>
먼저 적은 수의 점들로 대략적인 자세를 빠르게 찾는다. 마치 물건의 대략적인 위치와 방향을 파악하는 것이다.
##### 자세 가설의 점수 계산
$$
s_{hyp}​=\frac{N^c_m}{\sum_{p_m^c​\in P_m^c}​\text{​min}_{p_o^c​∈P_o^c}​​∣∣R_{hyp}^T​(p_o^c​−t_{hyp}​)−p_m^c​∣∣2}
$$
#### 정밀한 점 매칭<sub>Fine Point Matching</sub>
​첫 번째 단계의 결과를 바탕으로 더 많은 점들을 사용해서 정확한 자세를 계산한다. 마치 현미경으로 세밀하게 조정하는 것과 같다.
### Sparse-to-Dense Point Transformer
이는 적은 수의 점에서 학습한 정보를 많은 수의 점으로 효율적으로 전파하는 새로운 방법이다. 마치 소수의 전문가가 얻은 지식을 많은 사람에게 전달하는 것과 같다.
#### 훈련 방법
InfoNCE 손실 함수를 사용하여 모델을 학습시킨다.
$$
\mathcal{L} = \text{CE}(\mathcal{A}[1:, :], \hat{\mathcal{Y}}_m) + \text{CE}(\mathcal{A}[:, 1:]^T, \hat{\mathcal{Y}}_o) \quad
$$![[Image/Pasted image 20250626152537.png]]
# Experiments
연구팀은 BOP benchmark의 7개 데이터셋에서 실험을 진행했다. 이는 마치 다양한 환경에서 로봇의 능력을 테스트하는 것과 같다.
- **YCB-V**: 일상용품들
- **LM-O**: 가려진 물건들이 있는 복잡한 환경
- **T-LESS**: 질감이 없는 산업용 부품들
- **TUD-L, IC-BIN, ITODD, HB**: 다양한 조명과 배경 조건
## 하이퍼파라미터 설정
- 거친 매칭: $N^c_m=N^c_o=196$점, $T^c=3$ 레이어
- 정밀한 매칭: $N^f_m=N^f_o=2048$점, $T^f=3$ 레이어
- 학습률: 0.0001
- 배치 크기: 28
# Result
## 인스턴스 분할 성능
SAM-6D는 세 가지 매칭 점수를 모두 사용했을 때 최고 성능을 보였다.
- SAM 기반: **48.1 mAP**
- FastSAM 기반: **44.9 mAP**
### 각 점수의 기여도
- 의미적 매칭만: 44.0 mAP
- 외관적 매칭: 45.0 mAP
- 기하학적 매칭: **48.1 mAP**
## 자세 추정 기능
SAM-6D는 기존 최고 방법들을 크게 앞섰다.
- **SAM-6D (SAM): 70.4 AR**
- SAM-6D (FastSAM): 66.2 AR
- GigaPose: 57.9 AR
- ZeroPose: 58.4 AR
- MegaPose: 57.2 AR
### 데이터셋별 성능
- TUD-L: **90.4 AR** (최고)
- VCB-V: 84.5 AR
- HB: 77.6 AR
## 핵심 기술의 효과
### 배경 토큰 vs. 기존 방법
- 배경 토큰: 84.5 AR, **1.36초** (3배 빠름)
- 최적 수송: 81.4 AR, 4.31초
### 2단계 전략의 중요성
- 거친 매칭만: 77.6 AR
- 정밀한 매칭만: 40.2 AR (초기 정보 없이는 성능 급감)
- 두 단계 결합: **84.5 AR**
### SDPT의 효과
- 기하학적 변환기(196점): 81.7 AR
- 선형 변환기(2048점): 78.4 AR
- **SDPT**(196 -> 2048): **84.5 AR**
## 처리 속도
실시간 성능 (GeForce RTX 3090):
- FastSAM 기반: **1.43 초**/이미지
- SAM 기반: 4.37 초/이미지
- 기존 MegaPose: >10 초/이미지
## 실제 로봇 실험
실제 로봇 팔을 사용한 검증
- 성공률: **87.3%** (50회 중 43회 성공)
- 처리 시간: 2.1초/객체
- 새 물건 적응: **즉시** (추가 훈련 불필요)
![[Image/Pasted image 20250626152701.png]]
# Conclusion
SAM-6D는 다음과 같은 혁신을 달성했다.
## 기술적 기여
1. SAM을 6D 자세 추정에 성공적으로 적용
2. 배경 토큰을 통한 부분-부분 매칭 문제 해결
3. 2단계 매칭과 SDPT를 통한 정확성 효율성 균형
## 성능 향상
- 정확도: 기존 최고 대비 **20.5%** 향상 (70.4 vs. 58.4 AR)
- 속도: **68.4% 개선** (1.36초 vs. 4.31초)
- 일반화: Zero-shot으로 **즉시 적용** 가능
## 실용적 가치
SAM-6D는 제조업, 가정용 로봇, 의료, 물류 등 다양한 분야에서 새로운 물건이 등장해도 재훈련 없이 즉시 사용할 수 있는 실용적 기술을 제공한다. 특히 일부 데이터셋에서는 지도 학습 방법들을 능가하는 성능을 보여 Zero-shot 기술의 큰 가능성을 입증했다.
이 연구는 AI가 사람처럼 처음 보는 물건도 자연스럽게 이해하고 조작할 수 있는 미래를 한 걸음 더 가깝게 만든 중요한 성과다.