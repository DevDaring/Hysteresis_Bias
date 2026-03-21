# ============================================================
# Main Infrastructure — DigitalOcean H200 GPU Droplet
#
# Provisions:
#   1. GPU Droplet (H200, 141GB VRAM, 24 vCPU, 240GB RAM)
#   2. Runs setup.sh via remote-exec to install all deps
#   3. Installs PyTorch 2.5.1 + CUDA 12.4 + Flash Attention 2
#
# NOTE: The SSH key (id=53764100) is already registered in
#       DigitalOcean. We reference it by ID, not re-upload.
# ============================================================

# ----------------------------------------------------------
# Data source: existing SSH key in DigitalOcean account
# ----------------------------------------------------------
data "digitalocean_ssh_key" "main" {
  name = var.ssh_key_name
}

# ----------------------------------------------------------
# GPU Droplet — H200 x1 (141 GB VRAM)
#
# Spec from user:
#   1 GPU, 141 GB VRAM, 24 vCPU, 240 GB RAM
#   Boot disk: 720 GB NVMe, Scratch disk: 5 TB NVMe
#
# If the exact slug differs, list available sizes with:
#   doctl compute size list --output json | findstr gpu
#   doctl compute image list --public | findstr gpu
# ----------------------------------------------------------
resource "digitalocean_droplet" "gpu" {
  name     = var.droplet_name
  region   = var.region
  size     = var.droplet_size
  image    = var.droplet_image
  ssh_keys = [data.digitalocean_ssh_key.main.id]

  # Graceful shutdown on destroy
  graceful_shutdown = true

  # ----------------------------------------------------------
  # Connection for provisioners
  # ----------------------------------------------------------
  connection {
    type        = "ssh"
    user        = "root"
    private_key = file(var.ssh_private_key_path)
    host        = self.ipv4_address
    timeout     = "10m"
  }

  # ----------------------------------------------------------
  # 1. Upload setup script
  # ----------------------------------------------------------
  provisioner "file" {
    source      = "${path.module}/scripts/setup.sh"
    destination = "/root/setup.sh"
  }

  # ----------------------------------------------------------
  # 2. Run setup: Python deps + PyTorch + Flash Attention 2
  # ----------------------------------------------------------
  provisioner "remote-exec" {
    inline = [
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
