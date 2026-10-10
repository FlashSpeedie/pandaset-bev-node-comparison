_base_ = ['./bevformer_base.py']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
ida_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (464, 800),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}
print("Config successfully generated: Image inputs restricted to 464x800 resolution.")

# --- ABLATION EXPERIMENT 2: NO TEMPORAL MEMORY ---
queue_length = 1
video_test_mode = False
print("Ablation Config applied: Temporal memory disabled (queue_length=1).")

# Actually wire video_test_mode into the model config (previous version
# only set a standalone variable that nothing referenced).
model = dict(
    video_test_mode=video_test_mode,
)
