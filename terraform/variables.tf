# ============================================================
# Input Variables
# Values loaded from terraform.tfvars (auto-loaded)
# ============================================================

variable "do_token" {
  description = "DigitalOcean API token (starts with dop_v1_)"
  type        = string
  sensitive   = true
}

variable "ssh_key_id" {
  description = "Existing SSH key fingerprint/ID registered in DigitalOcean"
  type        = string
}

variable "ssh_key_name" {
  description = "Name of the SSH key in DigitalOcean"
  type        = string
  default     = "debz_key"
}

variable "ssh_private_key_path" {
  description = "Local path to SSH private key for provisioner"
  type        = string
  default     = "C:/Users/Debz/.ssh/id_rsa"
}

variable "region" {
  description = "DigitalOcean region for GPU Droplet"
  type        = string
  default     = "nyc2"
}

# ============================================================
# GPU Droplet Size
# To list available GPU sizes, run:
#   doctl compute size list --output json | findstr gpu
#
# H200 slug follows DO naming: gpu-h200x1-141gb
# Spec: 1x H200 141GB VRAM, 24 vCPU, 240GB RAM
# ============================================================
variable "droplet_size" {
  description = "DigitalOcean GPU Droplet size slug"
  type        = string
  default     = "gpu-h200x1-141gb"
}

# ============================================================
# GPU Image
# DigitalOcean GPU Droplets use ML-ready images with CUDA.
# To list available GPU images, run:
#   doctl compute image list --public --output json | findstr gpu
# ============================================================
variable "droplet_image" {
  description = "DigitalOcean GPU-ready base image slug"
  type        = string
  default     = "gpu-h200x1-141gb-ubuntu-22-04"
}

variable "droplet_name" {
  description = "Name for the GPU Droplet"
  type        = string
  default     = "bias-hysteresis-h200"
}

variable "project_root" {
  description = "Remote path where the project repo will be cloned"
  type        = string
  default     = "/root/Hysteresis_Bias"
}

variable "hf_token" {
  description = "HuggingFace token for private dataset access"
  type        = string
  sensitive   = true
  default     = ""
}

variable "github_classic_token" {
  description = "GitHub classic token for private HF dataset repos"
  type        = string
  sensitive   = true
  default     = ""
}
