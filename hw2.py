import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit    # 데이터 분포를 가장 잘 설명하는 곡선 찾기
from scipy.integrate import trapezoid   # 수치적분

bins, count = [], []
with open('hist2.csv', 'r') as f:
    for line in f.readlines():
        _b, _c = [float(i) for i in line.split(',')]
        bins.append(_b)
        count.append(_c)

def particle(x,a,b,c):
    return a * np.exp(-((x-b)**2)/(2*c**2))

# a: 최댓값, b: 중심축, c: 표준편차 (클수록 곡선이 완만하고 넓어짐)

# --------------------------------------------

def total_particle(x,a1,b1,c1,a2,b2,c2):
    return particle(x,a1,b1,c1) + particle(x,a2,b2,c2)

# 두 개의 가우시안으로 피팅
popt, pcov = curve_fit(total_particle, bins, count, maxfev = 10000)
# popt = Parmameter Optimal 최적 매개변수 (가장 정답에 가까운 파라미터 값들)
# pcov = Parameter Covariance 

x_fit = np.linspace(min(bins), max(bins), 1000)
# 측정값의 min부터 max까지 일정한 간격으로 나누는 x값 1000개

a1, b1, c1, a2, b2, c2 = popt

y1 = particle(x_fit, a1, b1, c1)
y2 = particle(x_fit, a2, b2, c2)
y_total = y1 + y2


plt.figure(figsize=(10, 6))
plt.bar(bins, count, width=(bins[1] - bins[0]), alpha=0.8, color='lightblue', label='Measured data')
plt.plot(x_fit, y_total, color = 'orange', linestyle='-', label='Total fit')
plt.plot(x_fit, y1, color = 'blue', linestyle='--', label='Particle A')
plt.plot(x_fit, y2, color = 'green', linestyle='--', label='Particle B')

plt.title('Gaussian Fit of Particle Energy Spectra')
plt.xlabel('Energy')
plt.ylabel('Count')
plt.legend()
plt.grid()
plt.show()



integral_A = trapezoid(y1, x_fit)
integral_B = trapezoid(y2, x_fit)
integral_total = integral_A + integral_B

ratio_A = integral_A / integral_total
ratio_B = integral_B / integral_total

print(f'Particle A integral: {integral_A:.2f}')
print(f'Particle B integral: {integral_B:.2f}')
print(f'Particle A ratio: {ratio_A * 100:.2f}%')
print(f'Particle B ratio: {ratio_B * 100:.2f}%')