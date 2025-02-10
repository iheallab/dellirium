import subprocess

print("Data processing.")

scripts = [
    "1_build_hdf5.py",
    "2_optimize.py",
    "3_train.py",
    "4_eval.py",
]

for script in scripts:

    print("-" * 40)
    print(f"Running {script}...")

    subprocess.run(["python", script])

    print(f"Finished running {script}")

print("-" * 40)
print("Finished data processing.")