# 3차원 기동 표적을 위한 LSTM-MPC 요격 유도

표적의 최근 2초간 위치와 속도를 LSTM에 입력하고, 예측 정보를 MPC에 결합하여 요격체의 횡가속도를 결정하는 3차원 시뮬레이터이다. 표적의 미래 가속도를 운동 모델의 외란으로 사용하는 A 방식과 미래 위치에서 도달 가능한 요격점을 선택하는 B 방식을 비교한다.

## 1. 문제 설정

회피 기동하는 표적을 요격체가 추격한다. 두 비행체 모두 횡가속도로 조종되는
점질량으로 모델링하며, 요격체의 속력은 일정하게 유지한다.

|               | 표적                | 요격체              |
| ------------- | ------------------- | ------------------- |
| 속력          | $150\ \mathrm{m/s}$ | $300\ \mathrm{m/s}$ |
| 횡가속도 한계 | $8g$                | $20g$               |
| 제어 입력     | $[a_p,\ a_v,\ a_s]$ | $[a_v,\ a_s]$       |

요격체의 상태는 $\mathbf{x}=[p_x,p_y,p_z,\gamma,\psi]^\top$ 이고 $\gamma$ 는 비행경로각,
$\psi$ 는 방위각이다.

각속도와 속도는 다음과 같이 정의된다.

$$
\begin{aligned}
\dot{\gamma} &= \frac{a_v}{V}, \\
\dot{\psi} &= \frac{a_s}{V\cos\gamma}, \\
\dot{\mathbf p} &= V
\begin{bmatrix}
\cos\gamma\cos\psi \\
\cos\gamma\sin\psi \\
\sin\gamma
\end{bmatrix}.
\end{aligned}
$$

교전은 다음 초기 조건에서 시작한다.

|                            | 표적                                 | 요격체                    |
| -------------------------- | ------------------------------------ | ------------------------- |
| 초기 위치                  | $(-2000,\ -1500,\ 1200)\ \mathrm{m}$ | $(0,\ 0,\ 0)\ \mathrm{m}$ |
| 초기 비행경로각 $\gamma_0$ | $5^\circ$                            | $25.6^\circ$              |
| 초기 방위각 $\psi_0$       | $80^\circ$                           | $-143.1^\circ$            |

요격체는 발사 시점에 표적 방향으로 조준되어 출발하므로, $\gamma_0$ 와 $\psi_0$ 는
초기 상대위치 벡터로부터 결정된다.

제어 주기는 $\Delta t=0.05\ \mathrm{s}$ ($20\ \mathrm{Hz}$)이다. 요격체는 유도를
시작하기 전 $2\ \mathrm{s}$ 동안 표적을 관측하며, 이 구간이 LSTM 입력 $40$스텝을
채운다.

## 2. 표적 기준 좌표계 정규화

두 예측기는 동일한 입력을 받는다 — 표적의 최근 $40$스텝 상태(위치와 속도,
$2\ \mathrm{s}$). 절대 좌표를 그대로 넣으면 LSTM이 좌표 자체를 외워버리므로,
표적 기준의 상대좌표계로 변환해서 넣는다.

현재 속도 $\mathbf v_t$ 로부터 정규직교 기저를 만든다.

$$
\begin{aligned}
\hat{\mathbf f} &= \frac{\mathbf v_t}{\lVert\mathbf v_t\rVert}, \\
\hat{\mathbf s} &=
\frac{\hat{\mathbf z}\times\hat{\mathbf f}}
{\lVert\hat{\mathbf z}\times\hat{\mathbf f}\rVert}, \\
\hat{\mathbf u} &= \hat{\mathbf f}\times\hat{\mathbf s}, \\
\mathbf R_t &=
\begin{bmatrix}
\hat{\mathbf f}^{\top} \\
\hat{\mathbf u}^{\top} \\
\hat{\mathbf s}^{\top}
\end{bmatrix}.
\end{aligned}
$$

이후 $\mathbf{R}_t^\top$를 곱해 입력 데이터를 표적 기준 국소 좌표계로 정규화한다.

$$
\mathbf X_t =
\left[
(\mathbf p_{t-k}-\mathbf p_t)\mathbf R_t^{\top}
\;\middle|\;
\mathbf v_{t-k}\mathbf R_t^{\top}
\right]_{k=39}^{0}
\in \mathbb R^{40\times 6}.
$$

두 방식은 예측기의 출력을 어느 좌표계로 돌려주느냐에서 갈린다.

A 방식은 표적의 국소 좌표계 가속도를 그대로 반환한다.

$$
\hat{\mathbf a}^{\mathrm{loc}}_{t+j}\in\mathbb R^3,
\qquad j=1,\ldots,8.
$$

각 성분은 $[a_p,\ a_v,\ a_s]$ 이고, MPC가 이를 표적 상태방정식의 외란으로 그대로
사용하므로 절대 좌표계로 되돌리는 단계가 없다.

B 방식은 국소좌표에서 예측한 상대 위치를 절대좌표로 복원하여 MPC의 기준 위치로 사용한다.

$$
\hat{\mathbf p}_{t+j}
= \hat{\mathbf p}^{\mathrm{loc}}_{t+j}\mathbf R_t+\mathbf p_t,
\qquad j=1,\ldots,100.
$$

## 3. 공통 MPC 최적화

두 방식 모두 비선형 운동 모델을 사용하므로, 이전 제어열로 생성한 공칭 궤적 주변에서 매 예측 시점의 모델을 수치 선형화하였다.

$$
\mathbf x_{k+1}
\approx \mathbf A_k\mathbf x_k
+\mathbf B_k\mathbf u_k
+\mathbf E_k\mathbf d_k
+\mathbf c_k.
$$

이를 예측 지평 전체에 누적하면 다음과 같이 나타낼 수 있다.

$$
\mathbf X
= \mathbf S\mathbf U
+\mathbf T\mathbf x_0
+\mathbf h.
$$

이 식을 각 방식의 이차 비용함수에 대입하여 다음의 이차계획법(QP) 문제를 구성하였다.

$$
\begin{aligned}
\underset{\mathbf U}{\operatorname{minimize}}
\quad &
\frac{1}{2}\mathbf U^{\top}\mathbf P\mathbf U
+\mathbf q^{\top}\mathbf U, \\
\text{subject to}
\quad &
-\frac{u_{\max}}{\sqrt{2}}
\le u_{k,i}
\le \frac{u_{\max}}{\sqrt{2}}.
\end{aligned}
$$

## 4. A 방식 — 가속도 외란 MPC

예측된 표적 가속도를 선형화된 상태방정식의 외란으로 넣는다. 예측 길이가
MPC 지평보다 짧으므로 나머지는 0으로 채운다.

$$
\mathbf d_k =
\begin{cases}
\hat{\mathbf a}_{t+k}, & 0\le k<8, \\
\mathbf 0, & 8\le k<N.
\end{cases}
$$

$$
\mathbf x_{k+1}
= \mathbf A_k\mathbf x_k
+\mathbf B_k\mathbf u_k
+\mathbf E_k\mathbf d_k
+\mathbf c_k.
$$

유도 목표는 시선각속도를 0으로 만드는
것이다. $\hat{\boldsymbol\lambda}$ 를 시선(LOS) 단위벡터라 할 때, 상대속도의
LOS 수직 성분은

$$
\mathbf v_{\perp}
= \mathbf v_{\mathrm{rel}}
-(\mathbf v_{\mathrm{rel}}\cdot\hat{\boldsymbol\lambda})
\hat{\boldsymbol\lambda}.
$$

$$
\begin{aligned}
\underset{\mathbf U}{\operatorname{minimize}}
\quad
&\frac{1}{\sigma^2}
\sum_{k=1}^{N}\lVert\mathbf v_{\perp,k}\rVert^2
+\frac{w_r}{r_0^2}\lVert\mathbf p_{\mathrm{rel},N}\rVert^2 \\
&+\frac{w_u}{u_{\max}^2}
\sum_k\lVert\mathbf u_k\rVert^2
+\frac{w_{\Delta}}{u_{\max}^2}
\sum_k\lVert\mathbf u_k-\mathbf u_{k-1}\rVert^2.
\end{aligned}
$$

제약은 $\lVert\mathbf u_k\rVert_\infty\le u_{\max}/\sqrt2$ 이며, 두 축을 동시에
써도 합성 가속도가 $20g$ 를 넘지 않게 한다.

가중치는 $\sigma=20\ \mathrm{m/s}$, $w_r=20$, $w_u=0.02$, $w_\Delta=0.05$로 설정하였다.

## 5. B 방식 — 위치 요격 MPC

궤적을 추종하는 대신, 언제 요격이 가능한지를 먼저 풀고 그 한 점을 조준한다.

1단계 — 요격 시점 탐색. 요격체의 이동 가능 거리 안에 들어오는 가장 빠른
예측 인덱스를 고른다.

$$
j_{\mathrm{hit}}
= \min\left\{
j\;:\;
\left\lVert
\hat{\mathbf p}_{t+j}-\mathbf p_t^{\mathrm{int}}
\right\rVert
\le V_{\mathrm{int}}j\Delta t+r_{\mathrm{hit}}
\right\}.
$$

실제로는 후보 인덱스를 성긴 간격에서 촘촘한 간격으로 좁혀가며 탐색하고, 각
후보마다 작은 종말 오차 QP를 풀어 도달 가능성을 검증한다.

2단계 — 조준. 단계 가중치를 그 한 인덱스에 거의 전부 몰아준다.

$$
\begin{aligned}
\underset{\mathbf U}{\operatorname{minimize}}
\quad
&w_{\mathrm{hit}}
\left\lVert
\mathbf p_{j_{\mathrm{hit}}}
-\hat{\mathbf p}_{t+j_{\mathrm{hit}}}
\right\rVert^2 \\
&+w_{\mathrm{track}}
\sum_{k<j_{\mathrm{hit}}}
\left\lVert\mathbf p_k-\hat{\mathbf p}_{t+k}\right\rVert^2
+\frac{w_u}{u_{\max}^2}
\sum_k\lVert\mathbf u_k\rVert^2.
\end{aligned}
$$

제약은 $\lVert\mathbf u_k\rVert_\infty\le u_{\max}/\sqrt2$ 이며, 두 축을 동시에
써도 합성 가속도가 $20g$ 를 넘지 않게 한다.

$w_{\mathrm{hit}}=200$, $w_{\mathrm{track}}=0.02$ 로 네 자릿수 차이다. 이
비율 자체가 방법의 핵심이다. 궤적은 약한 힌트일 뿐이고, 요격점이 목적이다.

## 6. 결론

A 방식은 짧은 예측 지평의 표적 가속도를 예측하여 상대운동 모델의 외란으로 직접 적용한다. 현재 운동 상태에서 가까운 미래의 변화만 계산하므로 복잡한 기동에서도 위치 예측 오차가 작게 유지되었다. 또한 하나의 짧은 지평 QP만 풀기 때문에 계산 시간과 제어 노력도 낮았다.

B 방식은 긴 예측 지평에서 표적 위치를 예측하고 후보 요격 시점의 도달 가능성을 반복해서 검사한다. 위치 예측 오차가 요격점 선택에 직접 반영되며, 그 오차는 표적이 한 제어 주기 동안 이동하는 거리와 비슷한 크기였다. 요격점이 갱신될 때마다 큰 경로 수정이 발생했고, 이것이 계산량과 제어 노력 증가의 원인으로 판단된다.

다만 두 방식은 LSTM 출력뿐 아니라 예측 길이, MPC 상태 구성, 비용함수 및 요격점 탐색 방식도 서로 다르다. 따라서 관측된 성능 차이를 가속도 예측 하나의 효과로만 해석할 수는 없다. 또한 PN이나 APN과 같은 고전 유도 법칙과의 비교는 수행하지 않았다.

이러한 한계를 고려하더라도, 본 연구에서 평가한 조건에서는 A 방식이 더 안정적이고 계산 효율적인 유도 구조로 나타났다.
