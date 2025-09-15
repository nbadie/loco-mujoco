import jax
import jax.numpy as np

import matplotlib.pyplot as plt

# xp = np.array([-0.008, -0.006, -0.004, 0.0, 0.01])
# fp = np.array([ -3*90.0, -3*60.0,  -3*30.0, 0.0, 100.0])


# x = np.linspace(-0.04, 0.04, 500)
# F_ext = np.interp(x, xp, fp, left="extrapolate", right="extrapolate") #, right="extrapolate") #, left=fp[0], right=fp[-1])
# F = np.interp(x, xp, fp)

# # Test 
# # xp_t = np.array([-0.03,-0.008, -0.006, -0.004, 0.0, 0.004, 0.006,0.008, 0.03]) #[-0.008, -0.006, -0.004, 0.0, 0.01, 0.03])
# # fp_t = np.array([-750.0, -90.0, -60.0, -30.0, 0.0, 30.0,60.0,90.0, 750.0]) #[-90.0, -60.0,  -30.0, 0.0, 100.0, 750]) 

# # F_t = np.interp(x, xp_t, fp_t, left=fp_t[0], right="extrapolate")
# high_stiffness = 43500
# low_stiffness = 4350
# a=0.02 #0.012 #0.01
# xp_t = np.array([-0.01, 0, 0.01,0.02])
# fp_t = np.array([-high_stiffness*a, 0, low_stiffness*a, low_stiffness*a+high_stiffness*a])
# F_t = np.interp(x, xp_t, fp_t, left="extrapolate", right="extrapolate")#fp_t[0], right="extrapolate")

# # Alterantive model
# high_stiffness = 43500 #50000 #46000 #43500 #46000 #43500 #70000 #30000 #43500
# low_stiffness = 4350 #10000 #4350 #6000 #7000 #4350 #1000
# a = 0.02 #0.038 #0.2 #0.02 #0.038#2 #0.038 #0.02 #0.038
# H = 0.02 #0.025 #0.02 #0.025
# delta_shift =  0 #0.01
# xp_1 = np.array([-a+delta_shift, 0+delta_shift, H+delta_shift, H+a+delta_shift])
# # fp_1 = np.array([-high_stiffness*(a+delta_shift), 0+delta_shift, low_stiffness*(H+delta_shift), low_stiffness*(H+delta_shift)+high_stiffness*(a+delta_shift)])
# fp_1 = np.array([-high_stiffness*a, 0, low_stiffness*H, low_stiffness*H+high_stiffness*a])
            

# plt.figure()
# plt.plot(x*1e3, F, label='F_interp (N)')
# plt.scatter(xp*1e3, fp, label='data points', color='red')
# plt.plot(x*1e3, F_ext, label='F_ext (N)', linestyle='--')
# plt.plot(xp_1*1e3, fp_1, label='alternative model', linestyle=':')
# plt.plot(x*1e3, F_t, label='F_t (N)', linestyle='-.')
# plt.axhline(0, color='k')
# plt.xlabel('displacement (mm)')
# plt.ylabel('force (N)')
# plt.title('Check: force vs displacement - force should oppose x')
# plt.grid(True)
# plt.show()
# plt.legend()
# plt.savefig("force_vs_displacement.png")



# # Test a new parameter set 

# high_stiffness = [20000, 10000, 5000, 2000]
# # high_stiffness = [200000, 100000, 50000, 20000]
# low_stiffness =[ x/100 for x in high_stiffness]
# high_stiffness_low = [x/10 for x in high_stiffness]

# lim_low = -0.1
# lim_high = 0.1

# a = 0.02 
# H = 0.02

# x = np.linspace(-0.04, 0.04, 500)

# for hs, ls, hsl in zip(high_stiffness, low_stiffness, high_stiffness_low):
#     xp = np.array([
#         -lim_low,
#         0,
#         H,
#         H + lim_high
# ])

#     fp = np.array([
#         hs * a,
#         0,
#         ls * H,
#         ls * H + hs * a
#     ])

#     F = np.interp(x, xp, fp, left="extrapolate", right="extrapolate")
#     plt.plot(x*1e3, F, label=f'hs={hs}, ls={ls}')

# plt.legend()
# plt.grid(True)
# plt.savefig("force_vs_displacement_test.png")


# # high_stiffness = 200000
# # middle_stiffness = high_stiffness / 10
# # low_stiffness = high_stiffness / 100

# # lim_low = -0.1
# # lim_high = 0.1

# # a = 0.02 
# # H = 0.02

# # xp = np.array([
# #     -lim_low,
# #     0,
# #     H,
# #     H + lim_high
# # ])
# # fp = np.array([
# #     middle_stiffness * a,
# #     0,
# #     low_stiffness * H,
# #     low_stiffness * H + high_stiffness * a
# # ])
# # F = np.interp(x, xp, fp, left="extrapolate", right="extrapolate")

# # plt.plot(x*1e3, F, label=f'hs={high_stiffness}, ls={low_stiffness}')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig("force_vs_displacement_test.png")


# Compare high and low/100 with high and low/100 and high & middle/10 & low/100 
high_stiffness = 20000
middle_stiffness = high_stiffness / 10
low_stiffness_100 = high_stiffness / 100
low_stiffness_1000 = high_stiffness / 1000



lim_low = -0.1
lim_high = 0.1

a = 0.02 
H = 0.02

x = np.linspace(-0.02, 0.03, 500)

xp= np.array([
    -lim_low,
    0,
    H,
    H + lim_high
])

fp_1 = np.array([
    high_stiffness * a,
    0,
    low_stiffness_100 * H,
    low_stiffness_100 * H + high_stiffness * a
])

fp_2 = np.array([
    high_stiffness * a,
    0,
    low_stiffness_1000 * H,
    low_stiffness_1000 * H + high_stiffness * a
])

fp_3 = np.array([
    middle_stiffness * a,
    0,
    low_stiffness_100 * H,
    low_stiffness_100 * H + high_stiffness * a
])

fp_4 = np.array([
    middle_stiffness * a,
    0,
    low_stiffness_1000 * H,
    low_stiffness_1000 * H + middle_stiffness * a
])

F1 = np.interp(x, xp, fp_1, left="extrapolate", right="extrapolate")
F2 = np.interp(x, xp, fp_2, left="extrapolate", right="extrapolate")
F3 = np.interp(x, xp, fp_3, left="extrapolate", right="extrapolate")
F4 = np.interp(x, xp, fp_4, left="extrapolate", right="extrapolate")

plt.plot(x*1e3, F1, label=f'hs={high_stiffness}, ls={low_stiffness_100}')
plt.plot(x*1e3, F2, label=f'hs={high_stiffness}, ls={low_stiffness_1000}')
plt.plot(x*1e3, F3, label=f'hs={high_stiffness}, ms={middle_stiffness}, ls={low_stiffness_100}')
plt.plot(x*1e3, F4, label=f'hs={middle_stiffness}, ls={low_stiffness_1000}')

plt.legend()
plt.grid(True)
plt.savefig("force_vs_displacement_test_comparison.png")
