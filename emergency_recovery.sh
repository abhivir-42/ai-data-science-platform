#!/bin/bash

# 🚨 EMERGENCY RECOVERY SCRIPT - AI DATA SCIENCE PLATFORM
# Created: September 23, 2025
# Purpose: Automated recovery for presentation system

set -e

echo "🔥 EMERGENCY RECOVERY SCRIPT ACTIVATED"
echo "====================================="

# Configuration
SSH_KEY="~/.ssh/fetch_ai_server_rsa"
SERVER="abhivir@35.197.223.41"
PROJECT_DIR="ai-data-science-platform"
BACKUP_FILE="working_state_backup_20250923_215026.tar.gz"

# Function to run command on server
run_remote() {
    ssh -i "$SSH_KEY" "$SERVER" "cd $PROJECT_DIR && $1"
}

# Function to test system health
test_health() {
    echo "🔍 Testing system health..."
    
    # Test frontend
    if curl -s -I "http://35.197.223.41:8001" | grep -q "200 OK"; then
        echo "✅ Frontend: OK"
        FRONTEND_OK=true
    else
        echo "❌ Frontend: FAILED"
        FRONTEND_OK=false
    fi
    
    # Test backend
    if curl -s "http://35.197.223.41:8000/api/health" | grep -q "healthy"; then
        echo "✅ Backend: OK"
        BACKEND_OK=true
    else
        echo "❌ Backend: FAILED"
        BACKEND_OK=false
    fi
    
    # Test ML Training
    if curl -s "http://35.197.223.41:8008/health" | grep -q "healthy"; then
        echo "✅ ML Training Agent: OK"
        ML_OK=true
    else
        echo "❌ ML Training Agent: FAILED"
        ML_OK=false
    fi
    
    if [ "$FRONTEND_OK" = true ] && [ "$BACKEND_OK" = true ] && [ "$ML_OK" = true ]; then
        echo "🎉 SYSTEM IS HEALTHY!"
        return 0
    else
        echo "⚠️  SYSTEM NEEDS RECOVERY"
        return 1
    fi
}

# Recovery Option 1: Quick Restart
recovery_option_1() {
    echo "🔄 RECOVERY OPTION 1: Quick Container Restart"
    echo "============================================="
    
    run_remote "sudo docker-compose restart"
    echo "⏳ Waiting 60 seconds for services to start..."
    sleep 60
    
    if test_health; then
        echo "✅ RECOVERY OPTION 1 SUCCESSFUL!"
        exit 0
    else
        echo "❌ RECOVERY OPTION 1 FAILED, trying Option 2..."
    fi
}

# Recovery Option 2: Full System Recovery
recovery_option_2() {
    echo "🔧 RECOVERY OPTION 2: Full System Recovery"
    echo "=========================================="
    
    # Stop everything
    echo "🛑 Stopping all services..."
    run_remote "sudo docker-compose down"
    
    # Restore backup
    echo "📦 Restoring backup..."
    run_remote "tar -xzf $BACKUP_FILE"
    
    # Full rebuild
    echo "🏗️  Rebuilding all services..."
    run_remote "sudo docker-compose up --build -d"
    
    # Wait for services
    echo "⏳ Waiting 120 seconds for services to start..."
    sleep 120
    
    if test_health; then
        echo "✅ RECOVERY OPTION 2 SUCCESSFUL!"
        exit 0
    else
        echo "❌ RECOVERY OPTION 2 FAILED, trying Nuclear Option..."
    fi
}

# Recovery Option 3: Nuclear Option
recovery_option_3() {
    echo "☢️  NUCLEAR OPTION: Complete System Reset"
    echo "========================================"
    
    echo "🧹 Cleaning everything..."
    run_remote "sudo docker-compose down --volumes --rmi all"
    run_remote "sudo docker system prune -f"
    
    echo "📦 Restoring backup..."
    run_remote "tar -xzf $BACKUP_FILE"
    
    echo "🏗️  Complete rebuild..."
    run_remote "sudo docker-compose up --build -d"
    
    echo "⏳ Waiting 180 seconds for complete rebuild..."
    sleep 180
    
    if test_health; then
        echo "✅ NUCLEAR OPTION SUCCESSFUL!"
        exit 0
    else
        echo "💥 NUCLEAR OPTION FAILED - MANUAL INTERVENTION REQUIRED"
        exit 1
    fi
}

# Main recovery logic
main() {
    echo "📊 Current time: $(date)"
    echo "🖥️  Server: $SERVER"
    echo "📁 Project: $PROJECT_DIR"
    echo ""
    
    # Initial health check
    if test_health; then
        echo "🎯 SYSTEM IS ALREADY HEALTHY - NO RECOVERY NEEDED!"
        exit 0
    fi
    
    echo ""
    echo "🚨 SYSTEM UNHEALTHY - STARTING RECOVERY SEQUENCE"
    echo ""
    
    # Try recovery options in order
    recovery_option_1
    recovery_option_2  
    recovery_option_3
    
    # If we get here, everything failed
    echo "💥 ALL RECOVERY OPTIONS FAILED"
    echo "📞 MANUAL INTERVENTION REQUIRED"
    echo "📋 Check PRESENTATION_READY_SYSTEM_CONFIG.md for manual steps"
    exit 1
}

# Handle command line arguments
case "${1:-auto}" in
    "test")
        test_health
        ;;
    "option1")
        recovery_option_1
        ;;
    "option2") 
        recovery_option_2
        ;;
    "nuclear")
        recovery_option_3
        ;;
    "auto"|*)
        main
        ;;
esac
