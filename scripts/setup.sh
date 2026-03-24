#!/bin/bash

# ============================================================================
# Noble RAG Infrastructure Setup Script
# ============================================================================
# This script initializes all required infrastructure components:
# - PostgreSQL database with schema
# - Qdrant vector database
# - Redis cache/session store
#
# Usage:
#   ./scripts/setup.sh [options]
#   
# Options:
#   --build             Build all containers
#   --start             Start all services
#   --stop              Stop all services
#   --restart           Restart all services
#   --logs              View container logs
#   --health            Check health of all services
#   --clean             Remove all containers and volumes (WARNING!)
#   --full-setup        [Default] Build, start, and health check
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

# Service names
POSTGRES_SERVICE="postgres"
QDRANT_SERVICE="qdrant"
REDIS_SERVICE="redis"
PGADMIN_SERVICE="pgadmin"
WHISPER_SERVICE="whisper-service"

# Health check timeouts
HEALTH_TIMEOUT=60
HEALTH_INTERVAL=5

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_success "Docker is available"
}

check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed. Please install it first."
        exit 1
    fi
    log_success "docker-compose is available"
}

setup_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$ENV_EXAMPLE" ]; then
            log_info "Creating .env from .env.example..."
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            log_success ".env file created"
        else
            log_error ".env and .env.example not found"
            exit 1
        fi
    else
        log_success ".env file found"
    fi
}

build_containers() {
    log_info "Building Docker containers..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build
    log_success "Containers built successfully"
}

start_services() {
    log_info "Starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    log_success "Services started"
}

stop_services() {
    log_info "Stopping services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    log_success "Services stopped"
}

restart_services() {
    log_info "Restarting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    log_success "Services restarted"
}

view_logs() {
    log_info "Showing logs (Ctrl+C to exit)..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
}

check_postgres() {
    log_info "Checking PostgreSQL health..."
    local elapsed=0
    
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if docker exec noble_rag_postgres pg_isready -U "${POSTGRES_USER:-rag_user}" -d "${POSTGRES_DB:-noble_rag}" &> /dev/null; then
            log_success "PostgreSQL is healthy"
            
            # Test database connection
            docker exec noble_rag_postgres psql -U "${POSTGRES_USER:-rag_user}" -d "${POSTGRES_DB:-noble_rag}" -c "SELECT version();" &> /dev/null
            log_success "PostgreSQL database connection verified"
            return 0
        fi
        
        log_info "Waiting for PostgreSQL... ($elapsed/$HEALTH_TIMEOUT seconds)"
        sleep $HEALTH_INTERVAL
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done
    
    log_error "PostgreSQL health check failed"
    return 1
}

check_qdrant() {
    log_info "Checking Qdrant health..."
    local elapsed=0
    
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if curl -s http://localhost:6333/health &> /dev/null; then
            log_success "Qdrant is healthy"
            
            # Initialize Qdrant collections if available
            if command -v python3 &> /dev/null; then
                log_info "Initializing Qdrant collections..."
                cd "$PROJECT_ROOT"
                python3 scripts/init_qdrant.py
                cd - > /dev/null
            fi
            return 0
        fi
        
        log_info "Waiting for Qdrant... ($elapsed/$HEALTH_TIMEOUT seconds)"
        sleep $HEALTH_INTERVAL
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done
    
    log_error "Qdrant health check failed"
    return 1
}

check_redis() {
    log_info "Checking Redis health..."
    local elapsed=0
    
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if docker exec noble_rag_redis redis-cli ping &> /dev/null; then
            log_success "Redis is healthy"
            return 0
        fi
        
        log_info "Waiting for Redis... ($elapsed/$HEALTH_TIMEOUT seconds)"
        sleep $HEALTH_INTERVAL
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done
    
    log_error "Redis health check failed"
    return 1
}

health_check_all() {
    log_info "Running health checks on all services..."
    
    local postgres_ok=0
    local qdrant_ok=0
    local redis_ok=0
    
    check_postgres && postgres_ok=1 || postgres_ok=0
    check_qdrant && qdrant_ok=1 || qdrant_ok=0
    check_redis && redis_ok=1 || redis_ok=0
    
    echo ""
    echo -e "${BLUE}=== Health Check Summary ===${NC}"
    
    if [ $postgres_ok -eq 1 ]; then
        log_success "PostgreSQL"
    else
        log_error "PostgreSQL"
    fi
    
    if [ $qdrant_ok -eq 1 ]; then
        log_success "Qdrant"
    else
        log_error "Qdrant"
    fi
    
    if [ $redis_ok -eq 1 ]; then
        log_success "Redis"
    else
        log_error "Redis"
    fi
    
    echo ""
    
    if [ $postgres_ok -eq 1 ] && [ $qdrant_ok -eq 1 ] && [ $redis_ok -eq 1 ]; then
        log_success "All services are healthy!"
        return 0
    else
        log_error "Some services are not healthy"
        return 1
    fi
}

clean_all() {
    log_warning "This will remove all containers and volumes. Do you want to continue? (yes/no)"
    read -r response
    
    if [ "$response" = "yes" ]; then
        log_info "Removing all containers and volumes..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

print_usage() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  --build              Build Docker containers"
    echo "  --start              Start all services"
    echo "  --stop               Stop all services"
    echo "  --restart            Restart all services"
    echo "  --logs               View container logs"
    echo "  --health             Check health of all services"
    echo "  --clean              Remove all containers and volumes"
    echo "  --full-setup         Build, start, and health check (default)"
    echo "  --help               Show this help message"
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║       Noble RAG Infrastructure Setup & Management          ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Check prerequisites
    check_docker
    check_docker_compose
    
    log_info "Project root: $PROJECT_ROOT"
    log_info "Docker Compose file: $DOCKER_COMPOSE_FILE"
    
    # Handle command line arguments
    local command="${1:---full-setup}"
    
    case "$command" in
        --build)
            check_docker_compose
            build_containers
            ;;
        --start)
            setup_env_file
            start_services
            ;;
        --stop)
            stop_services
            ;;
        --restart)
            setup_env_file
            restart_services
            health_check_all
            ;;
        --logs)
            view_logs
            ;;
        --health)
            health_check_all
            ;;
        --clean)
            clean_all
            ;;
        --full-setup)
            setup_env_file
            build_containers
            start_services
            echo ""
            health_check_all
            ;;
        --help|--usage)
            print_usage
            ;;
        *)
            log_error "Unknown option: $command"
            print_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
