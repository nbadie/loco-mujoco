import pickle
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

pkl_path = '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251125_001002_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251125_000203_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_234701_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_233828_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_232641_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_231806_evaluation_results_200steps_0seed.pkl'

#'/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_230320_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_225759_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_224943_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_224007_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_223512_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_222950_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_222501_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_221434_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_220612_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_215631_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_213434_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_212752_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_210240_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_205620_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_205047_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_204129_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_203300_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_202420_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_201611_evaluation_results_200steps_0seed.pkl' 

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_200206_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_195345_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_194601_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_193712_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_192154_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_172002_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_151607_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_151221_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_150541_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_145400_evaluation_results_200steps_0seed.pkl'


# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_144919_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_143915_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_143639_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_142011_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_140130_evaluation_results_200steps_0seed.pkl'
#'/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_134850_evaluation_results_200steps_0seed.pkl'

# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_133311_evaluation_results_200steps_0seed.pkl'
# '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-11-24/07-23-17/20251124_132036_evaluation_results_200steps_0seed.pkl'

def try_load_pickle(path):
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding='latin1')

def find_values(obj, targets):
    found = []
    def _rec(o):
        if isinstance(o, dict):
            for k, v in o.items():
                key_lower = str(k).lower()
                if any(t in key_lower for t in targets):
                    found.append(v)
                _rec(v)
        elif isinstance(o, (list, tuple)):
            for item in o:
                _rec(item)
        elif hasattr(o, '__dict__'):
            _rec(vars(o))
    _rec(obj)
    return found

data = try_load_pickle(pkl_path)

# candidate target substrings to match "socket_ty" in different possible key forms
targets = ['socket_ty', 'socket ty', 'socket:ty', 'socketty', 'socket.ty', 'socket-ty']

matches = find_values(data, targets)

if not matches:
    print("No entries matching socket_ty found in the pickle file.")
    sys.exit(1)

# Prefer the longest numeric sequence found (likely the time series)
def to_1d_array(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape((1,))
    return arr.ravel()

series_list = []
for m in matches:
    try:
        a = to_1d_array(m)
        # ignore tiny scalars unless it's the only thing
        if a.size > 1 or len(matches) == 1:
            series_list.append(a)
    except Exception:
        continue

if not series_list:
    print("Found matching keys but couldn't convert them to numeric arrays.")
    sys.exit(1)

# pick the longest series (most likely the joint over time)
series = max(series_list, key=lambda a: a.size)
t = np.arange(series.size)

# Try to find a time vector in the pickle (optional)
time_candidates = find_values(data, ['time', 't', 'timestamps', 'times'])
time_vec = None
for tc in time_candidates:
    try:
        # use the reusable safe converter above
        tc_arr = to_1d_array(tc)
    except Exception:
        # skip candidates that cannot be converted to an array
        continue

    # If the array has object dtype, try to coerce elements to float
    if tc_arr.dtype == object:
        try:
            tc_arr = np.asarray([float(x) for x in tc_arr])
        except Exception:
            # could not coerce to numeric, skip this candidate
            continue

    if tc_arr.size == series.size:
        time_vec = tc_arr
        break

x = time_vec if time_vec is not None else t

plt.figure(figsize=(8,4))
plt.plot(x, series, label='socket_ty')
plt.xlabel('time (steps)' if time_vec is None else 'time')
plt.ylabel('socket_ty')
plt.title('socket_ty over time')
plt.grid(True)
plt.legend()
# plt.ylim(-1, 1)
out_png = os.path.splitext(pkl_path)[0] + '_socket_ty.png'
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"Saved plot to {out_png}")
plt.show()