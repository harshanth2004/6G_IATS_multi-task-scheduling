#!/bin/bash
echo "Creating Virtual Environment..."
python3 -m venv btp_env

echo "Activating Environment..."
source btp_env/bin/activate

echo "Installing Requirements..."
pip install numpy scipy matplotlib

echo "Setup Complete!"