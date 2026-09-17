#!/bin/bash


GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN} Starting LCM type generation...${NC}"

# Clean
rm */*.jar
rm */*.java
rm */*.hpp
rm */*.class
rm */*.py
rm */*.pyc

rm -rf python
rm -rf cpp
rm -rf java
rm -rf file_list.txt
rm -rf lcmtypes
ls
