#!/bin/bash

cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j