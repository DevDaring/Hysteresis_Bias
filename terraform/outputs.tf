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

    ✅ Server provisioned and dependencies installed!

    1. SSH into the server:
       ssh -i ${var.ssh_private_key_path} root@${digitalocean_droplet.gpu.ipv4_address}

    2. Clone your repo:
       git clone <your-repo-url> ~/Hysteresis_Bias
       cd ~/Hysteresis_Bias

    3. Create .env file:
       echo 'HF_TOKEN=hf_xxx' >> .env
       echo 'Github_Classic_Token=ghp_xxx' >> .env

    4. Run the pipeline:
       python3 run_full_pipeline.py

    5. DESTROY when done (saves money):
       terraform destroy -auto-approve

  EOT
}
