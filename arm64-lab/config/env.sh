#!/usr/bin/env bash

# Настройки виртуальной ARM64-платы.
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_DIR="$LAB_ROOT/images"
SHARED_DIR="$LAB_ROOT/shared"
BASE_IMAGE="$IMAGE_DIR/debian-12-generic-arm64.qcow2"
DISK_IMAGE="$IMAGE_DIR/videot16-arm64.qcow2"
SEED_IMAGE="$IMAGE_DIR/cloud-seed.iso"
IMAGE_URL="https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-arm64.qcow2"

# Параметры не являются характеристиками Orange Pi: это настройки QEMU.
RAM="4G"
CPUS="4"
DISK_SIZE="32G"
SSH_PORT="2224"
