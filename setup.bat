@echo off
setlocal enabledelayedexpansion

echo.
echo 🐱 ThunderKittens Docker Setup (Windows) 🐱
echo ===========================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    echo    Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker compose version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    docker-compose --version >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo ❌ Docker Compose is not available. Please update Docker Desktop.
        pause
        exit /b 1
    )
    set COMPOSE_CMD=docker-compose
) else (
    set COMPOSE_CMD=docker compose
)

REM Check if NVIDIA Docker runtime is available
echo 🔍 Checking GPU access...
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ⚠️  NVIDIA Docker runtime not found or GPU not accessible.
    echo    This is required for ThunderKittens to work with GPU acceleration.
    echo    Please ensure:
    echo    1. Docker Desktop is running
    echo    2. WSL2 backend is enabled
    echo    3. NVIDIA drivers are installed on Windows
    echo    4. NVIDIA Container Toolkit is installed in WSL2
    echo.
    set /p continue="   Continue anyway? (y/N): "
    if /i "!continue!" neq "y" exit /b 1
)

REM Create necessary directories
echo 📁 Creating workspace directories...
if not exist "workspace" mkdir workspace
if not exist "examples" mkdir examples

REM Create .env file
echo 📝 Creating environment configuration...
(
echo # ThunderKittens Environment Configuration
echo CUDA_VISIBLE_DEVICES=0
echo JUPYTER_PORT=8888
echo TENSORBOARD_PORT=6006
) > .env

REM Handle different commands
set command=%1
if "%command%"=="" set command=build

if /i "%command%"=="build" goto :build_and_start
if /i "%command%"=="start" goto :start
if /i "%command%"=="stop" goto :stop
if /i "%command%"=="restart" goto :restart
if /i "%command%"=="shell" goto :shell
if /i "%command%"=="logs" goto :logs
if /i "%command%"=="clean" goto :clean
if /i "%command%"=="help" goto :help

echo ❌ Unknown command: %command%
echo.
goto :help

:build_and_start
echo 🔨 Building ThunderKittens Docker image...
%COMPOSE_CMD% build
if %ERRORLEVEL% neq 0 (
    echo ❌ Build failed!
    pause
    exit /b 1
)

echo 🚀 Starting ThunderKittens container...
%COMPOSE_CMD% up -d
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to start container!
    pause
    exit /b 1
)

echo ✅ ThunderKittens container is running!
echo.
echo 📊 Jupyter Notebook: http://localhost:8888
echo 📈 TensorBoard: http://localhost:6006
echo.
echo To access the container shell:
echo   docker exec -it thunderkittens-dev bash
echo.
echo To view logs:
echo   docker logs thunderkittens-dev
echo.
echo To stop the container:
echo   %COMPOSE_CMD% down
goto :end

:start
echo 🚀 Starting ThunderKittens container...
%COMPOSE_CMD% start
goto :end

:stop
echo 🛑 Stopping ThunderKittens container...
%COMPOSE_CMD% stop
goto :end

:restart
echo 🔄 Restarting ThunderKittens container...
%COMPOSE_CMD% restart
goto :end

:shell
echo 🐚 Opening shell in ThunderKittens container...
docker exec -it thunderkittens-dev bash
goto :end

:logs
echo 📋 Showing ThunderKittens container logs...
docker logs thunderkittens-dev
goto :end

:clean
echo 🧹 Cleaning up ThunderKittens Docker resources...
%COMPOSE_CMD% down -v
docker rmi thunderkittens:latest >nul 2>&1
echo ✅ Cleanup complete!
goto :end

:help
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   build     Build and start the container (default)
echo   start     Start existing container
echo   stop      Stop the container
echo   restart   Restart the container
echo   shell     Open shell in running container
echo   logs      Show container logs
echo   clean     Remove container and images
echo   help      Show this help message
goto :end

:end
if not "%command%"=="shell" pause