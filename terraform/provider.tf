# ============================================================
# Terraform Provider — DigitalOcean
# Provisions H200 GPU Droplet for Bias Hysteresis experiments
# ============================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.46"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}
