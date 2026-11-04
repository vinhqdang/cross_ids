#!/bin/bash
# usage: runs_sf.sh target steps
for v in "" "--notext"; do
  python -m crossids.schemaflow.train --target $1 --steps $2 $v --out runs/sf > runs/sf_$1$v.log 2>&1
  tag=${v:+_notext}
  python -m crossids.schemaflow.evaluate --target $1 --ckpt runs/sf/$1${tag}_s0.pt $v --out results/schemaflow/$1${tag}.csv >> runs/sf_$1$v.log 2>&1
done
