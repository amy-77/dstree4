#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Build the demo
echo "Building the demo..."
make -f Makefile.demo clean
make -f Makefile.demo

# Run the demo
echo -e "\nRunning the demo...\n"
./demo_regional_spline 