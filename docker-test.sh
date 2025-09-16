#!/bin/bash

# Docker Setup Test Script for AI Data Science Platform
# This script helps validate your Docker configuration

echo "🐳 AI Data Science Platform - Docker Setup Test"
echo "================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✅ Docker is installed"

# Check if Docker Compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo "✅ docker-compose is available"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    echo "✅ docker compose (plugin) is available"
else
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Validate docker-compose.yml
echo "🔍 Validating docker-compose.yml..."
if $COMPOSE_CMD config > /dev/null 2>&1; then
    echo "✅ docker-compose.yml is valid"
else
    echo "❌ docker-compose.yml has errors:"
    $COMPOSE_CMD config
    exit 1
fi

# Check if Dockerfiles exist
echo "🔍 Checking Dockerfiles..."
if [ -f "backend/Dockerfile" ]; then
    echo "✅ Backend Dockerfile exists"
else
    echo "❌ Backend Dockerfile is missing"
    exit 1
fi

if [ -f "frontend/Dockerfile" ]; then
    echo "✅ Frontend Dockerfile exists"
else
    echo "❌ Frontend Dockerfile is missing"
    exit 1
fi

# Check if .dockerignore files exist
if [ -f "backend/.dockerignore" ]; then
    echo "✅ Backend .dockerignore exists"
else
    echo "⚠️  Backend .dockerignore is missing (recommended)"
fi

if [ -f "frontend/.dockerignore" ]; then
    echo "✅ Frontend .dockerignore exists"
else
    echo "⚠️  Frontend .dockerignore is missing (recommended)"
fi

echo ""
echo "🎉 Docker setup validation completed!"
echo ""
echo "Next steps:"
echo "1. Run: $COMPOSE_CMD build --no-cache"
echo "2. Run: $COMPOSE_CMD up -d"
echo "3. Check services: $COMPOSE_CMD ps"
echo "4. Test endpoints:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/api/health"
echo "   - MLflow: http://localhost:5000"
echo "   - Flower (Celery): http://localhost:5555"
echo ""
echo "To stop: $COMPOSE_CMD down"
echo "To clean up: $COMPOSE_CMD down -v --rmi all"
