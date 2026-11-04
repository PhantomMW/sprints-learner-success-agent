import os
import zipfile

base_dir = r"C:\Users\mw851\sprints-learner-success-agent"
zip_path = os.path.join(base_dir, "sprints_capstone_submission.zip")

print("Building sprints_capstone_submission.zip...")

include_dirs = ['data', 'database', 'models', 'notebooks', 'n8n_workflows', 'api', 'scripts', 'docs', 'demo']
include_files = ['README.md', 'requirements.txt', 'Dockerfile', 'docker-compose.yml', 'Demo_Walkthrough.mp4']

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
    for d in include_dirs:
        dir_full = os.path.join(base_dir, d)
        if os.path.exists(dir_full):
            for root, _, files in os.walk(dir_full):
                if '__pycache__' in root or '.ipynb_checkpoints' in root:
                    continue
                for f in files:
                    if f.endswith('.npy') or f.endswith('.tmp'):
                        continue
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, base_dir)
                    zipf.write(full_p, rel_p)
                    
    for f in include_files:
        full_p = os.path.join(base_dir, f)
        if os.path.exists(full_p):
            zipf.write(full_p, f)

size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Created {zip_path} ({size_mb:.2f} MB)")
