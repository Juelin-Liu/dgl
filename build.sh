#!/bin/bash

rm -rf build

cmake -B build -GNinja

cmake --build build -j