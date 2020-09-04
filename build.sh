#!/bin/bash

set -eu

docker build -t registry.gitlab.com/lexunit/generali-car_clustering/lpr_service:0.2 .
