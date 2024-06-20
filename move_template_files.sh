#!/bin/sh

rsync -avh nomad-bayesian-optimization/ .
rm -rfv nomad-bayesian-optimization
