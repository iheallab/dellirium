import subprocess

print("Data processing.")

scripts = [
    "1_get_scores.py",
    "2_get_brain_status.py",
    "3_cohort_selection.py",
    "4_clinical_data.py",
    "5_static_data.py",
    "6_notes.py",
    "7_outcomes.py",
    "8_final_dataset.py",
    "9_split_cohort.py"
]

for script in scripts:

    print("-" * 40)
    print(f"Running {script}...")

    subprocess.run(["python", script])

    print(f"Finished running {script}")

print("-" * 40)
print("Finished data processing.")