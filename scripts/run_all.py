#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import subprocess
from pathlib import Path


scripts = [
    "e2e_scanvi.py",
    "e2e_scanvi_nb.py",
    "e2e_lbl8r_raw.py",
    "e2e_lbl8r_expr.py",
    "e2e_xgb_expr.py",
    "e2e_xgb_cnt.py",
    "lbl8r_scvi.py" "lbl8r_xgb.py",
    "lbl8r_pca.py",
    "lbl8r_expr_pca.py",
]

for script in scripts:
    try:
        cmd = ["/home/ergonyc/mambaforge/envs/lbl8r/bin/python", script]
        # Setting check=True will raise CalledProcessError if the command returns a non-zero exit status
        print(f"Running {script}")
        subprocess.run(cmd, check=True)
        print(f"Successfully ran: {script}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred when running {script}: {e}")
    except Exception as e:
        # This catches other exceptions
        print(f"An unexpected error occurred: {e}")


# %%
