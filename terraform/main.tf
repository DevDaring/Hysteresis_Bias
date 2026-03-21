# ============================================================
# Main Infrastructure — DigitalOcean H200 GPU Droplet
#
# FULLY AUTOMATED:
#   1. GPU Droplet (H200, 141GB VRAM, 24 vCPU, 240GB RAM)
#   2. Installs PyTorch 2.5.1 + CUDA 12.4 + Flash Attention 2
#   3. Clones repo from GitHub
#   4. Creates .env with HF_TOKEN + Github_Classic_Token
#   5. Downloads datasets from HuggingFace
#   6. Pre-caches all 6 models
#   7. Runs dry run to verify full pipeline
#   8. Verifies Flash Attention on all causal models
#
# After apply: SSH in and run `python3 run_full_pipeline.py`
# ============================================================

# ----------------------------------------------------------
# Data source: existing SSH key in DigitalOcean account
# ----------------------------------------------------------
data "digitalocean_ssh_key" "main" {
  name = var.ssh_key_name
}

# ----------------------------------------------------------
# GPU Droplet — H200 x1 (141 GB VRAM)
# ----------------------------------------------------------
resource "digitalocean_droplet" "gpu" {
  name     = var.droplet_name
  region   = var.region
  size     = var.droplet_size
  image    = var.droplet_image
  ssh_keys = [data.digitalocean_ssh_key.main.id]

  graceful_shutdown = true

  connection {
    type        = "ssh"
    user        = "root"
    private_key = file(var.ssh_private_key_path)
    host        = self.ipv4_address
    timeout     = "10m"
  }

  # ----------------------------------------------------------
  # 1. Upload secrets file (HF_TOKEN, Github token)
  #    This is sourced by setup.sh to create the project .env
  # ----------------------------------------------------------
  provisioner "file" {
    content     = <<-EOF
      export HF_TOKEN="${var.hf_token}"
      export GITHUB_TOKEN="${var.github_classic_token}"
    EOF
    destination = "/root/.server_env"
  }

  # ----------------------------------------------------------
  # 2. Upload the full setup script
  # ----------------------------------------------------------
  provisioner "file" {
    source      = "${path.module}/scripts/setup.sh"
    destination = "/root/setup.sh"
  }

  # ----------------------------------------------------------
  # 3. Run full setup: deps → repo → data → models → dry run
  #    Timeout is long because model downloads take time
  # ----------------------------------------------------------
  provisioner "remote-exec" {
    inline = [
      "chmod 600 /root/.server_env",
      "chmod +x /root/setup.sh",
      "bash /root/setup.sh 2>&1 | tee /root/setup.log",
    ]
  }
}

# ----------------------------------------------------------
# Firewall — restrict access to SSH (port 22)
# ----------------------------------------------------------
resource "digitalocean_firewall" "gpu_fw" {
  name        = "${var.droplet_name}-fw"
  droplet_ids = [digitalocean_droplet.gpu.id]

  # Inbound: SSH only
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  # Outbound: allow all (needed for pip, HuggingFace, GitHub)
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
