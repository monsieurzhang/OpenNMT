# Dual Learning for Machine Translation

This is an unofficial implementation based on [Dual Learning for Machine Translation](https://papers.nips.cc/paper/6469-dual-learning-for-machine-translation.pdf) upon [OpenNMT](http://opennmt.net/).

The whole workflow works like:

* Read raw sentences (from language A)
* Translate with model A-->B to get K-best translations (in language B)
* Translate with model B-->A to get 1-best translation (in language A)
* Build batches to feed into trainer (all together K batches)
* Train model A-->B with averaged gradient calculated based on K batches
* Train model B-->A with averaged gradient calculated based on K batches
* Read raw sentences (from language B)
* iterate as above mentioned

## Quickstart

```
./run.dual.sh
```
to see the training on demo data.

## Notes

1) Preprocess the data.

We provide several scripts (./tools/scripts) to create the preprocessed data. We need to filter raw sentences (e.g. remove sentences with more than one <unk>), sort and randomize batch sentences, and add mono-lingual data into preprocessed file. In this demo, we do not add mono-lingual data here.

2) Train the model.

We use two GPUs to help decoding and training. Specially, GPU1 is used for decoder_ab & trainer_ab, GPU2 is used for decoder_ba & trainer_ba. We set the batch_size to 32, which can be hold in 8G memory. We use SGD as default optimizer and set a small learning rate (0.01). K-best translation is set to 2. You may refer to ./run.dual.sh for details.

3) Performance.

We use both log P and BLEU score as the reward function. In the case of BLEU, the performance (evaluated by ppl. based on validation set) increases a little and then begins to decrease. The training speed is slow since there needs (1+K)*2*decode_whole_training_set+(1+1)*2*training. There is no parallelization at this time.
