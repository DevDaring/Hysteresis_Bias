#!/bin/bash
# Pipeline launcher — run on server
source /root/.server_env
cd /root/Hysteresis_Bias
exec python3.12 run_full_pipeline.py --skip-setup 2>&1 | tee /root/pipeline.log
