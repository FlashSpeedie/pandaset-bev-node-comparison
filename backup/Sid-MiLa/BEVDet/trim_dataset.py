import pickle

pkl_path = "custom_nuscenes_infos_clean.pkl"
print(f"Loading {pkl_path}...")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict) and 'infos' in data:
    original_count = len(data['infos'])
    # Skip the first 12 washed-out frames
    data['infos'] = data['infos'][12:]
    new_count = len(data['infos'])
    print(f"Successfully trimmed infos: {original_count} -> {new_count} frames.")
else:
    original_count = len(data)
    data = data[12:]
    new_count = len(data)
    print(f"Successfully trimmed list: {original_count} -> {new_count} frames.")

with open(pkl_path, "wb") as f:
    pickle.dump(data, f)

print("Dataset metadata updated successfully!")
