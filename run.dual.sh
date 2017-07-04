#!/bin/bash

HOMEDIR=`pwd`

# corpus
SRC_LANG="en"
TGT_LANG="de"
SRC_TRAIN="$HOMEDIR/data/src-train.txt"
TGT_TRAIN="$HOMEDIR/data/tgt-train.txt"
SRC_VALID="$HOMEDIR/data/src-val.txt"
TGT_VALID="$HOMEDIR/data/tgt-val.txt"
SRC_VOCAB=30000
SRC_LENGTH=50
TGT_VOCAB=30000
TGT_LENGTH=50


WORKINGDIR=$HOMEDIR/workingdir_dual
SRC_TGT_FOLDER=$WORKINGDIR/${SRC_LANG}2${TGT_LANG}
TGT_SRC_FOLDER=$WORKINGDIR/${TGT_LANG}2${SRC_LANG}

mkdir -p $WORKINGDIR
mkdir -p $SRC_TGT_FOLDER
mkdir -p $TGT_SRC_FOLDER

export CUDA_VISIBLE_DEVICES=0,1
. /DEV/torch/install/bin/torch-activate

# preprocess
th preprocess.lua -src_vocab_size $SRC_VOCAB -src_seq_length $SRC_LENGTH -tgt_vocab_size $TGT_VOCAB -tgt_seq_length $TGT_LENGTH -train_src $SRC_TRAIN -train_tgt $TGT_TRAIN -valid_src $SRC_VALID -valid_tgt $TGT_VALID -save_data $SRC_TGT_FOLDER/bin

th preprocess.lua -src_vocab_size $TGT_VOCAB -src_seq_length $TGT_LENGTH -tgt_vocab_size $SRC_VOCAB -tgt_seq_length $SRC_LENGTH -train_src $TGT_TRAIN -train_tgt $SRC_TRAIN -valid_src $TGT_VALID -valid_tgt $SRC_VALID -save_data $TGT_SRC_FOLDER/bin

# train-baseline
th train.lua -data $SRC_TGT_FOLDER/bin-train.t7 -save_model $SRC_TGT_FOLDER/model -end_epoch 1 -gpuid 1

th train.lua -data $TGT_SRC_FOLDER/bin-train.t7 -save_model $TGT_SRC_FOLDER/model -end_epoch 1 -gpuid 1

# pre-processing-dual
DUAL_DATA_FOLDER=$WORKINGDIR/dual_data
mkdir -p $DUAL_DATA_FOLDER

$HOMEDIR/tools/scripts/filter_by_voc.pl $SRC_TRAIN $SRC_TGT_FOLDER/bin.src.dict > $DUAL_DATA_FOLDER/train.${SRC_LANG}.cand
$HOMEDIR/tools/scripts/filter_by_voc.pl $TGT_TRAIN $TGT_SRC_FOLDER/bin.src.dict > $DUAL_DATA_FOLDER/train.${TGT_LANG}.cand

cat $DUAL_DATA_FOLDER/train.${SRC_LANG}.cand | awk '{print NF,$0}' | sort -n | cut -d' ' -f 2- > $DUAL_DATA_FOLDER/train.${SRC_LANG}.cand.sort
cat $DUAL_DATA_FOLDER/train.${TGT_LANG}.cand | awk '{print NF,$0}' | sort -n | cut -d' ' -f 2- > $DUAL_DATA_FOLDER/train.${TGT_LANG}.cand.sort

BATCH=32
$HOMEDIR/tools/scripts/rand_batch.pl $DUAL_DATA_FOLDER/train.${SRC_LANG}.cand.sort $BATCH > $DUAL_DATA_FOLDER/train.${SRC_LANG}.cand.sort.$BATCH
$HOMEDIR/tools/scripts/rand_batch.pl $DUAL_DATA_FOLDER/train.${TGT_LANG}.cand.sort $BATCH > $DUAL_DATA_FOLDER/train.${TGT_LANG}.cand.sort.$BATCH

$HOMEDIR/tools/scripts/gen_batch_file.pl $DUAL_DATA_FOLDER/train.${SRC_LANG}.cand.sort.$BATCH $DUAL_DATA_FOLDER/train.${TGT_LANG}.cand.sort.$BATCH $BATCH > $DUAL_DATA_FOLDER/train.dual.batch$BATCH.txt

# train-dual
EPOCH=1
TRAIN_BATCH_FILE=$DUAL_DATA_FOLDER/train.dual.batch$BATCH.txt
AB_MODEL=`ls ${SRC_TGT_FOLDER}/model_epoch${EPOCH}_*`
BA_MODEL=`ls ${TGT_SRC_FOLDER}/model_epoch${EPOCH}_*`
AB_DATA=${SRC_TGT_FOLDER}/bin-train.t7
BA_DATA=${TGT_SRC_FOLDER}/bin-train.t7
N_BEST=2
START_EPOCH=2

export THC_CACHING_ALLOCATOR=0
th dual.lua -dual \
  -train_src $TRAIN_BATCH_FILE \
  -model $AB_MODEL \
  -model_ba $BA_MODEL \
  -train_from $AB_MODEL \
  -data $AB_DATA \
  -data_ba $BA_DATA \
  -save_model ./model  \
  -max_batch_size $BATCH  \
  -n_best $N_BEST  \
  -start_epoch $START_EPOCH \
  -max_sent_length 50 \
  -report_ppl_every_N_batches 5 \
  -learning_rate 0.01 \
  -gpuid 1

