#!/bin/bash
# usage: runs_sf.sh target steps   (train + evaluate, with and without feature text)
for v in "" "--notext"; do
  tag=${v:+_notext}
  [ -f runs/sf/$1${tag}_s0.pt ] || python -m crossids.schemaflow.train --target $1 --steps $2 $v --out runs/sf > runs/sf_$1$tag.log 2>&1
  python -m crossids.schemaflow.evaluate --target $1 --ckpt runs/sf/$1${tag}_s0.pt $v --out results/schemaflow/$1${tag}.csv >> runs/sf_$1$tag.log 2>&1
done
