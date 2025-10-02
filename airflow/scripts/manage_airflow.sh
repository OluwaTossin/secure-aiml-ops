#!/bin/bash

# Airflow Deployment and Management Script
# This script provides commands for managing the Airflow deployment

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Function to display usage
usage() {
    echo -e "${BLUE}Airflow Management Script for Secure AI/ML Operations${NC}"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start      Start all Airflow services"
    echo "  stop       Stop all Airflow services"
    echo "  restart    Restart all Airflow services"
    echo "  status     Show status of all services"
    echo "  logs       Show logs from all services"
    echo "  webserver  Show only webserver logs"
    echo "  scheduler  Show only scheduler logs"
    echo "  worker     Show only worker logs"
    echo "  shell      Open Airflow shell"
    echo "  test-dag   Test a specific DAG"
    echo "  list-dags  List all available DAGs"
    echo "  trigger    Trigger a DAG run"
    echo "  cleanup    Clean up old logs and temp files"
    echo "  backup     Backup Airflow database"
    echo "  restore    Restore Airflow database from backup"
    echo "  health     Check system health"
    echo "  update     Update Airflow to latest version"
    echo "  help       Show this help message"
    echo ""
}

# Function to check if Docker is running
check_docker() {
    if ! docker info &>/dev/null; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if services are running
check_services() {
    if ! docker-compose ps | grep -q "Up"; then
        print_warning "Airflow services are not running. Use '$0 start' to start them."
        return 1
    fi
    return 0
}

# Start services
start_services() {
    print_header "🚀 Starting Airflow services..."
    check_docker
    
    docker-compose up -d
    
    print_status "Waiting for services to be ready..."
    sleep 15
    
    print_status "Services started successfully!"
    print_status "Web UI: http://localhost:8080"
    print_status "Username: admin, Password: admin"
}

# Stop services
stop_services() {
    print_header "🛑 Stopping Airflow services..."
    
    docker-compose down
    
    print_status "All services stopped successfully!"
}

# Restart services
restart_services() {
    print_header "🔄 Restarting Airflow services..."
    
    stop_services
    sleep 5
    start_services
}

# Show service status
show_status() {
    print_header "📊 Service Status"
    check_docker
    
    echo ""
    docker-compose ps
    echo ""
    
    # Check health endpoints
    if check_services; then
        print_status "Checking health endpoints..."
        
        # Check webserver health
        if curl -s http://localhost:8080/health &>/dev/null; then
            print_status "✅ Webserver: Healthy"
        else
            print_warning "⚠️  Webserver: Not responding"
        fi
        
        # Check scheduler health (via API if available)
        if curl -s http://localhost:8080/api/v1/health &>/dev/null; then
            print_status "✅ API: Healthy"
        else
            print_warning "⚠️  API: Not responding"
        fi
    fi
}

# Show logs
show_logs() {
    local service=${1:-}
    
    check_docker
    check_services || return 1
    
    if [ -z "$service" ]; then
        print_header "📋 Showing logs from all services (Ctrl+C to exit)"
        docker-compose logs -f
    else
        print_header "📋 Showing logs from $service (Ctrl+C to exit)"
        docker-compose logs -f "$service"
    fi
}

# Open Airflow shell
open_shell() {
    print_header "🐚 Opening Airflow shell..."
    check_docker
    check_services || return 1
    
    docker-compose exec airflow-webserver airflow shell
}

# Test DAG
test_dag() {
    local dag_id=${1:-}
    
    if [ -z "$dag_id" ]; then
        print_error "Please specify a DAG ID. Use '$0 list-dags' to see available DAGs."
        return 1
    fi
    
    print_header "🧪 Testing DAG: $dag_id"
    check_docker
    check_services || return 1
    
    docker-compose exec airflow-webserver airflow dags test "$dag_id" $(date '+%Y-%m-%d')
}

# List DAGs
list_dags() {
    print_header "📋 Available DAGs"
    check_docker
    check_services || return 1
    
    docker-compose exec airflow-webserver airflow dags list
}

# Trigger DAG
trigger_dag() {
    local dag_id=${1:-}
    
    if [ -z "$dag_id" ]; then
        print_error "Please specify a DAG ID. Use '$0 list-dags' to see available DAGs."
        return 1
    fi
    
    print_header "▶️  Triggering DAG: $dag_id"
    check_docker
    check_services || return 1
    
    docker-compose exec airflow-webserver airflow dags trigger "$dag_id"
    print_status "DAG triggered successfully!"
}

# Cleanup old files
cleanup() {
    print_header "🧹 Cleaning up old logs and temp files..."
    
    # Clean up old logs (older than 7 days)
    find ./logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Clean up Docker
    docker system prune -f
    
    # Clean up old DAG runs if services are running
    if check_services; then
        print_status "Cleaning up old DAG runs..."
        docker-compose exec airflow-webserver airflow db clean --clean-before-timestamp $(date -d '30 days ago' '+%Y-%m-%d')
    fi
    
    print_status "Cleanup completed!"
}

# Backup database
backup_db() {
    print_header "💾 Backing up Airflow database..."
    check_docker
    check_services || return 1
    
    local backup_file="airflow_backup_$(date '+%Y%m%d_%H%M%S').sql"
    
    docker-compose exec postgres pg_dump -U airflow airflow > "./backups/$backup_file"
    
    print_status "Database backup saved as: ./backups/$backup_file"
}

# Restore database
restore_db() {
    local backup_file=${1:-}
    
    if [ -z "$backup_file" ]; then
        print_error "Please specify a backup file to restore."
        return 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        return 1
    fi
    
    print_header "🔄 Restoring Airflow database from: $backup_file"
    print_warning "This will overwrite the current database. Continue? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        check_docker
        check_services || return 1
        
        docker-compose exec -T postgres psql -U airflow -d airflow < "$backup_file"
        print_status "Database restored successfully!"
        print_status "Please restart Airflow services."
    else
        print_status "Restore cancelled."
    fi
}

# Health check
health_check() {
    print_header "🏥 System Health Check"
    
    # Check Docker
    if docker info &>/dev/null; then
        print_status "✅ Docker: Running"
    else
        print_error "❌ Docker: Not running"
        return 1
    fi
    
    # Check disk space
    local disk_usage=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 90 ]; then
        print_status "✅ Disk space: ${disk_usage}% used"
    else
        print_warning "⚠️  Disk space: ${disk_usage}% used (high)"
    fi
    
    # Check memory usage
    local mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$mem_usage" -lt 90 ]; then
        print_status "✅ Memory: ${mem_usage}% used"
    else
        print_warning "⚠️  Memory: ${mem_usage}% used (high)"
    fi
    
    # Check services if running
    if check_services; then
        show_status
        
        # Check DAG integrity
        print_status "Checking DAG integrity..."
        if docker-compose exec airflow-webserver airflow dags list-import-errors | grep -q "No import errors"; then
            print_status "✅ DAGs: No import errors"
        else
            print_warning "⚠️  DAGs: Import errors detected"
            docker-compose exec airflow-webserver airflow dags list-import-errors
        fi
    fi
    
    print_status "Health check completed!"
}

# Update Airflow
update_airflow() {
    print_header "🔄 Updating Airflow to latest version..."
    print_warning "This will stop services and pull latest images. Continue? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        stop_services
        docker-compose pull
        start_services
        print_status "Airflow updated successfully!"
    else
        print_status "Update cancelled."
    fi
}

# Create backups directory if it doesn't exist
mkdir -p ./backups

# Main script logic
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    webserver)
        show_logs "airflow-webserver"
        ;;
    scheduler)
        show_logs "airflow-scheduler"
        ;;
    worker)
        show_logs "airflow-worker"
        ;;
    shell)
        open_shell
        ;;
    test-dag)
        test_dag "${2:-}"
        ;;
    list-dags)
        list_dags
        ;;
    trigger)
        trigger_dag "${2:-}"
        ;;
    cleanup)
        cleanup
        ;;
    backup)
        backup_db
        ;;
    restore)
        restore_db "${2:-}"
        ;;
    health)
        health_check
        ;;
    update)
        update_airflow
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        print_error "Invalid command: ${1:-}"
        echo ""
        usage
        exit 1
        ;;
esac