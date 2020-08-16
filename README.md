# SIFRank
Original paper [SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model](https://ieeexplore.ieee.org/document/8954611)

## requirements
```
allennlp==0.8.4
nltk==3.4.3
torch==1.2.0
stanza==1.0.0
```
## Download
* ELMo ``elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`` from [here](https://allennlp.org/elmo) , and save it to the ``auxiliary_data/`` directory

## Sample usage
```
import sys
sys.path.append('/content/drive/My Drive/SIFRank')
sys.path.append('/content/drive/My Drive/SIFRank/embeddings')
import stanza
import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus

#download from https://allennlp.org/elmo
options_file = "https://exawizardsallenlp.blob.core.windows.net/data/options.json"
weight_file = "/content/drive/My Drive/SIFRank/auxiliary_data/weights.hdf5"

ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file, cuda_device=0)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
ja_model = stanza.Pipeline(
    lang="ja", processors={}, use_gpu=True
)
elmo_layers_weight = [0.0, 1.0, 0.0]

text = "ここにテキストを入力してください。"
keyphrases = SIFRank(text, SIF, ja_model, N=15,elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, ja_model, N=15, elmo_layers_weight=elmo_layers_weight)

print(keyphrases)
print(keyphrases_)
```

## Cite
If you use this code, please cite this paper
```
@article{DBLP:journals/access/SunQZWZ20,
  author    = {Yi Sun and
               Hangping Qiu and
               Yu Zheng and
               Zhongwei Wang and
               Chaoran Zhang},
  title     = {SIFRank: {A} New Baseline for Unsupervised Keyphrase Extraction Based
               on Pre-Trained Language Model},
  journal   = {{IEEE} Access},
  volume    = {8},
  pages     = {10896--10906},
  year      = {2020},
  url       = {https://doi.org/10.1109/ACCESS.2020.2965087},
  doi       = {10.1109/ACCESS.2020.2965087},
  timestamp = {Fri, 07 Feb 2020 12:04:22 +0100},
  biburl    = {https://dblp.org/rec/journals/access/SunQZWZ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
