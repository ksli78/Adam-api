#!/bin/bash
# Adam RAG System Management Script for RHEL9
# Provides easy management of Adam services

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_status() {
    echo "================================================================"
    echo -e "${BLUE}Adam RAG System Status${NC}"
    echo "================================================================"
    echo ""

    echo "Containers:"
    docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAME|ollama|adam"
    echo ""

    echo "Health Check:"
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        HEALTH=$(curl -s http://localhost:8000/health)
        echo -e "${GREEN}✓ API is responding${NC}"
        echo "$HEALTH"
    else
        echo -e "${RED}✗ API is not responding${NC}"
    fi
    echo ""

    if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is responding${NC}"
    else
        echo -e "${RED}✗ Ollama is not responding${NC}"
    fi
}

start_services() {
    echo -e "${BLUE}Starting Adam services...${NC}"

    # Start Ollama first
    if ! docker ps --format '{{.Names}}' | grep -q "^ollama$"; then
        docker start ollama
        echo "Waiting for Ollama..."
        sleep 5
    else
        echo -e "${GREEN}Ollama already running${NC}"
    fi

    # Start Adam API
    if ! docker ps --format '{{.Names}}' | grep -q "^adam-api$"; then
        docker start adam-api
        echo "Waiting for Adam API..."
        sleep 5
    else
        echo -e "${GREEN}Adam API already running${NC}"
    fi

    echo -e "${GREEN}✓ Services started${NC}"
    show_status
}

stop_services() {
    echo -e "${BLUE}Stopping Adam services...${NC}"
    docker stop adam-api ollama 2>/dev/null || true
    echo -e "${GREEN}✓ Services stopped${NC}"
}

restart_services() {
    echo -e "${BLUE}Restarting Adam services...${NC}"
    docker restart ollama
    sleep 5
    docker restart adam-api
    sleep 5
    echo -e "${GREEN}✓ Services restarted${NC}"
    show_status
}

show_logs() {
    echo "Which logs do you want to see?"
    echo "1) Adam API"
    echo "2) Ollama"
    echo "3) Both"
    read -r CHOICE

    case $CHOICE in
        1)
            docker logs -f adam-api
            ;;
        2)
            docker logs -f ollama
            ;;
        3)
            docker logs --tail=50 adam-api
            echo ""
            echo "--- Ollama Logs ---"
            docker logs --tail=50 ollama
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
}

backup_data() {
    BACKUP_DIR="${1:-/opt/adam-backups}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/adam-backup-$TIMESTAMP.tar.gz"

    echo -e "${BLUE}Creating backup...${NC}"
    echo "Backup location: $BACKUP_FILE"

    mkdir -p "$BACKUP_DIR"

    docker run --rm \
        -v adam-data:/data \
        -v "$BACKUP_DIR":/backup \
        alpine tar czf "/backup/adam-backup-$TIMESTAMP.tar.gz" /data

    if [ -f "$BACKUP_FILE" ]; then
        SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
        echo -e "${GREEN}✓ Backup created: $BACKUP_FILE ($SIZE)${NC}"
    else
        echo -e "${RED}✗ Backup failed${NC}"
    fi
}

restore_data() {
    BACKUP_FILE="$1"

    if [ -z "$BACKUP_FILE" ]; then
        echo "Usage: $0 restore <backup-file>"
        exit 1
    fi

    if [ ! -f "$BACKUP_FILE" ]; then
        echo -e "${RED}✗ Backup file not found: $BACKUP_FILE${NC}"
        exit 1
    fi

    echo -e "${YELLOW}WARNING: This will restore data from backup${NC}"
    echo "Backup file: $BACKUP_FILE"
    echo "Continue? (yes/NO)"
    read -r CONFIRM

    if [[ ! $CONFIRM =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Restore cancelled"
        exit 0
    fi

    echo -e "${BLUE}Stopping services...${NC}"
    docker stop adam-api

    echo -e "${BLUE}Restoring data...${NC}"
    docker run --rm \
        -v adam-data:/data \
        -v "$(dirname "$BACKUP_FILE")":/backup \
        alpine sh -c "rm -rf /data/* && tar xzf /backup/$(basename "$BACKUP_FILE") -C /"

    echo -e "${BLUE}Starting services...${NC}"
    docker start adam-api

    echo -e "${GREEN}✓ Data restored${NC}"
}

update_images() {
    echo -e "${YELLOW}Update requires new image files${NC}"
    echo "Enter directory containing new .tar.gz files:"
    read -r IMAGE_DIR

    if [ ! -d "$IMAGE_DIR" ]; then
        echo -e "${RED}✗ Directory not found${NC}"
        exit 1
    fi

    echo -e "${BLUE}Stopping services...${NC}"
    stop_services

    echo -e "${BLUE}Importing new images...${NC}"

    # Import Ollama if present
    OLLAMA_FILE=$(find "$IMAGE_DIR" -name "ollama*.tar.gz" | head -1)
    if [ -n "$OLLAMA_FILE" ]; then
        echo "Importing Ollama..."
        gunzip -c "$OLLAMA_FILE" | docker load
    fi

    # Import Adam API if present
    ADAM_FILE=$(find "$IMAGE_DIR" -name "adam*.tar.gz" | head -1)
    if [ -n "$ADAM_FILE" ]; then
        echo "Importing Adam API..."
        gunzip -c "$ADAM_FILE" | docker load
    fi

    echo -e "${GREEN}✓ Images updated${NC}"

    echo -e "${BLUE}Starting services with new images...${NC}"
    start_services
}

show_menu() {
    echo "================================================================"
    echo -e "${BLUE}Adam RAG System Management${NC}"
    echo "================================================================"
    echo ""
    echo "1)  Show status"
    echo "2)  Start services"
    echo "3)  Stop services"
    echo "4)  Restart services"
    echo "5)  Show logs"
    echo "6)  Backup data"
    echo "7)  Restore data"
    echo "8)  Update images"
    echo "9)  Test API"
    echo "10) Open firewall ports"
    echo "q)  Quit"
    echo ""
    echo -n "Select option: "
}

test_api() {
    echo -e "${BLUE}Testing API...${NC}"
    echo ""

    echo "1. Health check:"
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
    echo ""

    echo "2. System query test:"
    curl -s -X POST http://localhost:8000/query \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "What is your name?"}' | python3 -m json.tool 2>/dev/null || echo "Install python3 for formatted output"
    echo ""
}

open_firewall() {
    if command -v firewall-cmd &> /dev/null; then
        echo -e "${BLUE}Opening firewall ports...${NC}"
        sudo firewall-cmd --permanent --add-port=8000/tcp
        sudo firewall-cmd --permanent --add-port=11434/tcp
        sudo firewall-cmd --reload
        echo -e "${GREEN}✓ Ports opened: 8000, 11434${NC}"
        echo ""
        sudo firewall-cmd --list-ports
    else
        echo -e "${YELLOW}firewalld not found${NC}"
    fi
}

# Main
if [ $# -gt 0 ]; then
    case "$1" in
        status)
            show_status
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        backup)
            backup_data "$2"
            ;;
        restore)
            restore_data "$2"
            ;;
        update)
            update_images
            ;;
        test)
            test_api
            ;;
        firewall)
            open_firewall
            ;;
        *)
            echo "Usage: $0 {status|start|stop|restart|logs|backup|restore|update|test|firewall}"
            exit 1
            ;;
    esac
else
    # Interactive menu
    while true; do
        show_menu
        read -r OPTION

        case $OPTION in
            1) show_status ;;
            2) start_services ;;
            3) stop_services ;;
            4) restart_services ;;
            5) show_logs ;;
            6) backup_data ;;
            7)
                echo "Enter backup file path:"
                read -r BACKUP_FILE
                restore_data "$BACKUP_FILE"
                ;;
            8) update_images ;;
            9) test_api ;;
            10) open_firewall ;;
            q|Q) exit 0 ;;
            *) echo -e "${RED}Invalid option${NC}" ;;
        esac

        echo ""
        echo "Press Enter to continue..."
        read -r
    done
fi
