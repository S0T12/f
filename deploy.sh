#!/bin/bash
# XAU/USD Trading System - Deployment Script

set -e

echo "=== XAU/USD Trading System Deployment ==="
echo ""

# Variables
PROJECT_DIR="/root/projects/xau-trading"
FRONTEND_DIR="${PROJECT_DIR}/frontend"
DOCKER_DIR="${PROJECT_DIR}/docker"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Step 1: Updating repository...${NC}"
cd ${PROJECT_DIR}
git pull origin main

echo -e "${YELLOW}Step 2: Installing Flutter...${NC}"
if ! command -v flutter &> /dev/null; then
    echo "Flutter not found. Installing..."
    cd /root
    git clone https://github.com/flutter/flutter.git -b stable --depth 1
    export PATH="$PATH:/root/flutter/bin"
    echo 'export PATH="$PATH:/root/flutter/bin"' >> ~/.bashrc
    flutter precache --web
    flutter config --enable-web
else
    echo "Flutter already installed"
fi

echo -e "${YELLOW}Step 3: Building Flutter PWA...${NC}"
cd ${FRONTEND_DIR}
flutter pub get
flutter build web --release --dart-define=API_URL=http://156.236.31.98:8080/api/v1

echo -e "${YELLOW}Step 4: Creating build directory for nginx...${NC}"
mkdir -p ${FRONTEND_DIR}/build/web

echo -e "${YELLOW}Step 5: Building and starting Docker containers...${NC}"
cd ${DOCKER_DIR}
docker compose down --remove-orphans 2>/dev/null || true
docker compose build --no-cache
docker compose up -d

echo -e "${YELLOW}Step 6: Waiting for services to start...${NC}"
sleep 30

echo -e "${YELLOW}Step 7: Checking service health...${NC}"
echo "PostgreSQL:"
docker exec xau_postgres pg_isready -U xau_trading && echo "  ✓ PostgreSQL is ready" || echo "  ✗ PostgreSQL failed"

echo "Redis:"
docker exec xau_redis redis-cli -a xau_redis_password_2024 ping && echo "  ✓ Redis is ready" || echo "  ✗ Redis failed"

echo "RabbitMQ:"
docker exec xau_rabbitmq rabbitmq-diagnostics -q ping && echo "  ✓ RabbitMQ is ready" || echo "  ✗ RabbitMQ failed"

echo "Backend API:"
curl -s http://localhost:8001/health && echo "  ✓ Backend is ready" || echo "  ✗ Backend failed"

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Access Points:"
echo "  - PWA Frontend: http://156.236.31.98:8080"
echo "  - API Documentation: http://156.236.31.98:8080/docs"
echo "  - API Health: http://156.236.31.98:8001/health"
echo "  - Grafana: http://156.236.31.98:3002 (admin/xau_admin_2024)"
echo "  - Prometheus: http://156.236.31.98:9091"
echo "  - RabbitMQ Management: http://156.236.31.98:15673 (xau_trading/xau_rabbitmq_password_2024)"
echo ""
echo "Docker commands:"
echo "  - View logs: docker compose logs -f"
echo "  - Stop all: docker compose down"
echo "  - Restart: docker compose restart"
