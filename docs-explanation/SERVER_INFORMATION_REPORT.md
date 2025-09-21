# 🖥️ Fetch AI Company Server Information Report

**Generated:** September 18, 2025  
**Server:** vm-08 (35.197.223.41)  
**User:** abhivir  
**Purpose:** AI Data Science Platform Deployment

---

## 📋 Server Details (Known Information)

### **Basic System Info**
- **Hostname:** vm-08
- **IP Address:** 35.197.223.41
- **Operating System:** Debian GNU/Linux
- **Kernel:** Linux 6.1.0-29-cloud-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.1.123-1 (2025-01-02) x86_64
- **Architecture:** x86_64
- **SSH Access:** ✅ Working with fetch_ai_server_rsa key

---

## 🔧 Essential Commands for Server Information

### **System Information**
```bash
# Basic system info
uname -a                    # Kernel and system info
hostname                    # Server hostname
whoami                     # Current user
pwd                        # Current directory

# Operating system details
cat /etc/os-release         # OS version and details
cat /proc/version          # Kernel version
lsb_release -a             # Distribution info (if available)
```

### **Hardware Information**
```bash
# CPU Information
lscpu                      # Detailed CPU info
cat /proc/cpuinfo          # Raw CPU data
nproc                      # Number of processing units

# Memory Information
free -h                    # Memory usage (human readable)
cat /proc/meminfo          # Detailed memory info

# Disk Information
df -h                      # Disk space usage (human readable)
lsblk                      # Block devices
fdisk -l                   # Disk partitions (may need sudo)
```

### **Network Information**
```bash
# Network interfaces and IPs
ip addr show               # Network interfaces
ip route show              # Routing table
hostname -I                # All IP addresses
netstat -tuln              # Open ports (if netstat available)
ss -tuln                   # Socket statistics (modern alternative)
```

### **Process and Service Information**
```bash
# Running processes
ps aux                     # All running processes
top                        # Real-time process viewer
htop                       # Enhanced process viewer (if available)

# System services
systemctl list-units --type=service  # All services
systemctl status           # System status
```

### **Docker and Container Information**
```bash
# Docker installation check
docker --version           # Docker version
docker info                # Docker system info
docker ps -a               # All containers
docker images              # All images

# Docker Compose
docker-compose --version   # Docker Compose version
which docker-compose       # Docker Compose location

# Container management
docker system df           # Docker disk usage
docker network ls          # Docker networks
docker volume ls           # Docker volumes
```

### **Storage and File System**
```bash
# Disk usage
du -sh *                   # Directory sizes
du -sh /var/lib/docker     # Docker storage usage
find / -type f -size +100M 2>/dev/null  # Large files

# File system info
mount                      # Mounted filesystems
cat /proc/mounts          # Mount information
```

### **Performance and Monitoring**
```bash
# System load
uptime                     # System uptime and load
iostat                     # I/O statistics (if available)
vmstat                     # Virtual memory statistics

# Resource monitoring
iotop                      # I/O usage by process (if available)
nethogs                    # Network usage by process (if available)
```

---

## 📦 Software Installation Status

### **Package Management**
```bash
# Debian/Ubuntu package management
apt list --installed       # All installed packages
apt update                 # Update package lists
apt search docker         # Search for Docker packages

# Python and development tools
python3 --version          # Python version
pip3 --version            # Pip version
git --version             # Git version
curl --version            # Curl version
```

### **Required Software for Deployment**
```bash
# Check if required software is installed
docker --version || echo "Docker not installed"
docker-compose --version || echo "Docker Compose not installed"
git --version || echo "Git not installed"
curl --version || echo "Curl not installed"
python3 --version || echo "Python not installed"
```

---

## 🚀 Deployment Prerequisites Check

### **1. Docker Installation**
```bash
# Install Docker (if not installed)
sudo apt update
sudo apt install -y docker.io

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (to run without sudo)
sudo usermod -aG docker $USER
```

### **2. Docker Compose Installation**
```bash
# Install Docker Compose (if not installed)
sudo apt install -y docker-compose

# Or install latest version manually
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### **3. Port Availability Check**
```bash
# Check if required ports are available
netstat -tuln | grep -E ':(3000|8000|5432|6379|5555|5000)'
ss -tuln | grep -E ':(3000|8000|5432|6379|5555|5000)'

# Required ports for AI Data Science Platform:
# 3000 - Frontend (Next.js)
# 8000 - Backend (FastAPI)
# 5432 - PostgreSQL
# 6379 - Redis
# 5555 - Celery Flower
# 5000 - MLflow
```

### **4. System Resources Check**
```bash
# Check available resources
free -h                    # Available memory
df -h /                    # Available disk space
nproc                      # CPU cores

# Recommended minimum:
# Memory: 4GB+ (8GB preferred)
# Disk: 20GB+ free space
# CPU: 2+ cores
```

---

## 📂 File Transfer and Deployment

### **Getting Project Files to Server**
```bash
# Option 1: Git clone (if repo is accessible)
git clone [repository-url]
cd ai-data-science-platform

# Option 2: SCP from local machine (run from local terminal)
scp -i ~/.ssh/fetch_ai_server_rsa -r /Users/abhivir42/projects/ai-data-science-platform abhivir@35.197.223.41:~/

# Option 3: Create directory and upload files
mkdir -p ~/ai-data-science-platform
# Then upload individual files as needed
```

### **Deployment Commands**
```bash
# Navigate to project directory
cd ~/ai-data-science-platform

# Build and start all containers
docker-compose up -d --build

# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all containers
docker-compose down
```

---

## 🔍 Troubleshooting Commands

### **Log Analysis**
```bash
# System logs
journalctl -f              # Follow system logs
dmesg                      # Kernel messages
tail -f /var/log/syslog    # System log tail

# Docker logs
docker-compose logs [service-name]
docker logs [container-name]
```

### **Performance Debugging**
```bash
# Memory usage
cat /proc/meminfo | head -20
ps aux --sort=-%mem | head -10  # Top memory users

# Disk usage
df -h
du -sh /* 2>/dev/null | sort -hr | head -10  # Largest directories

# Network connections
ss -tuln | grep LISTEN     # Listening ports
netstat -i                 # Network interface statistics
```

### **Permission Issues**
```bash
# File permissions
ls -la                     # Detailed file listing
chmod +x filename          # Make file executable
chown user:group filename  # Change ownership

# Docker permissions
sudo usermod -aG docker $USER  # Add user to docker group
newgrp docker              # Apply group changes
```

---

## ✅ Server Readiness Checklist

- [ ] **System Information Gathered**
  - [ ] OS version confirmed
  - [ ] Hardware specs documented
  - [ ] Available ports checked

- [ ] **Software Prerequisites**
  - [ ] Docker installed and running
  - [ ] Docker Compose installed
  - [ ] Git available (optional)
  - [ ] Sufficient system resources

- [ ] **Network Configuration**
  - [ ] Required ports (3000, 8000, 5432, 6379, 5555, 5000) available
  - [ ] Firewall configured (if applicable)
  - [ ] External access configured

- [ ] **File System**
  - [ ] Project directory created
  - [ ] Sufficient disk space available
  - [ ] Proper permissions set

- [ ] **Deployment Ready**
  - [ ] docker-compose.yml uploaded
  - [ ] Environment variables configured
  - [ ] All containers build successfully
  - [ ] Services accessible externally

---

## 🎯 Next Steps

1. **Gather System Information:** Run the commands above to collect full server details
2. **Install Missing Software:** Install Docker, Docker Compose if needed
3. **Upload Project Files:** Transfer the AI Data Science Platform to the server
4. **Deploy Containers:** Run `docker-compose up -d --build`
5. **Verify Deployment:** Check all services are running and accessible
6. **Configure External Access:** Ensure services are accessible from outside the server

---

**Generated by:** AI Data Science Platform Deployment Assistant  
**Last Updated:** September 18, 2025  
**Contact:** abhivir.singh@fetch.ai
