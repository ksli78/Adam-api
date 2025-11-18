#!/bin/bash
#
# Extend LVM Partition for vLLM Deployment
# Run on RHEL9 if using LVM (most common setup)
#
# This script helps you extend /var or create a new /data partition
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "LVM Partition Extension for vLLM"
echo "========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}This script requires root privileges.${NC}"
  echo "Please run with sudo:"
  echo "  sudo bash extend-lvm-partition.sh"
  exit 1
fi

# Check if LVM is available
if ! command -v lvextend &> /dev/null; then
    echo -e "${RED}✗ LVM tools not found${NC}"
    echo "Either LVM is not installed or you're not using LVM"
    exit 1
fi

echo -e "${BLUE}Current LVM configuration:${NC}"
echo ""

echo "Volume Groups:"
vgs
echo ""

echo "Logical Volumes:"
lvs
echo ""

echo "Physical Volumes:"
pvs
echo ""

echo -e "${BLUE}Current disk usage:${NC}"
df -h | grep -E "Filesystem|/$|/usr|/var|/home"
echo ""

# Check for free space in VG
echo -e "${BLUE}Checking for available space...${NC}"
VG_NAME=$(vgs --noheadings -o vg_name | head -1 | xargs)
VG_FREE=$(vgs --noheadings -o vg_free --units G "$VG_NAME" | xargs | sed 's/G//')

echo "Volume Group: $VG_NAME"
echo "Free Space: ${VG_FREE}G"
echo ""

if (( $(echo "$VG_FREE < 5" | bc -l) )); then
    echo -e "${RED}✗ Insufficient free space in volume group ($VG_FREE GB)${NC}"
    echo ""
    echo "You need to either:"
    echo "  1. Add a new physical disk and extend the VG"
    echo "  2. Move Docker to a different partition with more space"
    echo "  3. Clean up space on existing partitions"
    echo ""
    echo "See: reconfigure-docker-data-dir.sh for option 2"
    exit 1
fi

echo -e "${GREEN}✓ Found ${VG_FREE}G free space in volume group${NC}"
echo ""

# Show options
echo -e "${BLUE}Extension Options:${NC}"
echo ""
echo "1. Extend /var (where Docker stores data)"
echo "2. Create new /data partition for Docker"
echo "3. Cancel"
echo ""

read -p "Select option (1-3): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Extending /var partition...${NC}"

        # Find /var LV
        VAR_LV=$(lvs --noheadings -o lv_path,lv_name | grep "var" | awk '{print $1}' | head -1)

        if [ -z "$VAR_LV" ]; then
            echo -e "${RED}✗ Could not find /var logical volume${NC}"
            echo "Available LVs:"
            lvs
            exit 1
        fi

        echo "Logical Volume: $VAR_LV"
        echo ""

        read -p "How much space to add? (in GB, e.g., 50): " SIZE

        if ! [[ "$SIZE" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}✗ Invalid size${NC}"
            exit 1
        fi

        if (( $(echo "$SIZE > $VG_FREE" | bc -l) )); then
            echo -e "${RED}✗ Requested size ($SIZE GB) exceeds available space ($VG_FREE GB)${NC}"
            exit 1
        fi

        echo ""
        echo -e "${YELLOW}WARNING: This will extend the /var partition${NC}"
        echo "This operation is generally safe but backup important data first."
        echo ""
        read -p "Continue? (yes/no): " confirm

        if [ "$confirm" != "yes" ]; then
            echo "Cancelled"
            exit 0
        fi

        echo ""
        echo "Extending logical volume..."
        lvextend -L +${SIZE}G "$VAR_LV"

        echo "Resizing filesystem..."
        # Detect filesystem type
        FS_TYPE=$(df -T "$VAR_LV" | tail -1 | awk '{print $2}')

        if [ "$FS_TYPE" = "xfs" ]; then
            xfs_growfs "$VAR_LV"
        elif [ "$FS_TYPE" = "ext4" ]; then
            resize2fs "$VAR_LV"
        else
            echo -e "${RED}✗ Unknown filesystem type: $FS_TYPE${NC}"
            echo "Manual resize required"
            exit 1
        fi

        echo ""
        echo -e "${GREEN}✓ /var extended successfully!${NC}"
        echo ""
        df -h | grep var
        ;;

    2)
        echo ""
        echo -e "${BLUE}Creating new /data partition...${NC}"

        read -p "Size for /data partition (in GB, e.g., 50): " SIZE

        if ! [[ "$SIZE" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}✗ Invalid size${NC}"
            exit 1
        fi

        if (( $(echo "$SIZE > $VG_FREE" | bc -l) )); then
            echo -e "${RED}✗ Requested size ($SIZE GB) exceeds available space ($VG_FREE GB)${NC}"
            exit 1
        fi

        echo ""
        echo -e "${YELLOW}WARNING: This will create a new /data partition${NC}"
        echo ""
        read -p "Continue? (yes/no): " confirm

        if [ "$confirm" != "yes" ]; then
            echo "Cancelled"
            exit 0
        fi

        echo ""
        echo "Creating logical volume..."
        lvcreate -L ${SIZE}G -n data "$VG_NAME"

        echo "Creating filesystem..."
        mkfs.xfs "/dev/$VG_NAME/data"

        echo "Creating mount point..."
        mkdir -p /data

        echo "Mounting..."
        mount "/dev/$VG_NAME/data" /data

        echo "Adding to /etc/fstab..."
        echo "/dev/$VG_NAME/data /data xfs defaults 0 0" >> /etc/fstab

        echo ""
        echo -e "${GREEN}✓ /data partition created successfully!${NC}"
        echo ""
        df -h | grep data
        echo ""
        echo "Next step: Configure Docker to use /data"
        echo "Run: bash reconfigure-docker-data-dir.sh /data/docker"
        ;;

    3)
        echo "Cancelled"
        exit 0
        ;;

    *)
        echo -e "${RED}✗ Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo "Verify with: df -h"
echo ""
