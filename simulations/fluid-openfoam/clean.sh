#!/bin/sh
set -e -u

rm ../precice-run/ -r
. ../tools/cleaning-tools.sh

clean_openfoam .
