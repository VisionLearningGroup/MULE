#!/bin/bash

set -x
set -e

mkdir fasttext
cd fasttext

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.vec.gz

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz
gunzip cc.de.300.vec.gz

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz
gunzip cc.fr.300.vec.gz

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz
gunzip cc.cs.300.vec.gz

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz
gunzip cc.zh.300.vec.gz

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
gunzip cc.ja.300.vec.gz

cd ..

