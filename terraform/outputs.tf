# ============================================================
# Outputs — displayed after terraform apply
# ============================================================

output "droplet_ip" {
  description = "Public IPv4 address of the GPU Droplet"
  value       = digitalocean_droplet.gpu.ipv4_address
}

output "ssh_command" {
  description = "SSH command to connect to the server"
  value       = "ssh -i ${var.ssh_private_key_path} root@${digitalocean_droplet.gpu.ipv4_address}"
}

output "droplet_id" {
  description = "Droplet ID (for doctl commands)"
  value       = digitalocean_droplet.gpu.id
}

output "droplet_name" {
  description = "Droplet name"
  value       = digitalocean_droplet.gpu.name
}

output "region" {
  description = "Deployed region"
  value       = digitalocean_droplet.gpu.region
}

output "monthly_cost" {
  description = "Estimated monthly cost (USD)"
  value       = digitalocean_droplet.gpu.price_monthly
}

output "next_steps" {
  description = "What to do after provisioning"
  value       = <<-EOT

    ✅ Server FULLY provisioned and verified!
       - All dependencies installed (global Python, no venv)
       - Flash Attention 2 verified on all causal models
       - Datasets downloaded and validated
       - All 6 models pre-cached
       - Dry run passed

    1. SSH into the server:
       ssh -i ${var.ssh_private_key_path} root@${digitalocean_droplet.gpu.ipv4_address}

    2. Run the full pipeline:
       cd ~/Hysteresis_Bias
       python3 run_full_pipeline.py

    3. Or run step-by-step:
       python3 scripts/03_parallel_baseline.py
       python3 scripts/04_parallel_injection.py
       python3 scripts/05_parallel_removal.py
       python3 scripts/06_compute_asymmetry.py
       python3 scripts/07_parallel_hessian.py
       python3 scripts/08_linear_connectivity.py
       python3 scripts/09_cultural_analysis.py
       python3 scripts/10_parallel_comparatives.py
       python3 scripts/11_comparative_asymmetry.py
       python3 scripts/12_generate_figures.py
       python3 scripts/13_generate_tables.py

    4. Check setup log if needed:
       cat /root/setup.log

    5. DESTROY when done (saves $$):
       terraform destroy -auto-approve

  EOT
}
