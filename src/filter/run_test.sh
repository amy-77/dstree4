#!/bin/bash

# Navigate to the correct directory
cd "$(dirname "$0")"

# Build the test using the makefile
make -f Makefile.test clean
make -f Makefile.test

# Run the test
./test_regional_spline 