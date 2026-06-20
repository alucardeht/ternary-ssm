@echo off
REM Quick activation script for ML environment

echo Activating Python ML Environment...
call .\.venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo Python version:
python --version
echo.
echo PyTorch status:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
echo.
echo Ready for ML experiments!
