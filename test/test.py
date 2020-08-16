import sys
sys.path.append('/content/drive/My Drive/SIFRank')
sys.path.append('/content/drive/My Drive/SIFRank/embeddings')
import stanza
import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import stanza

#download from https://allennlp.org/elmo
options_file = "https://exawizardsallenlp.blob.core.windows.net/data/options.json"
weight_file = "/content/drive/My Drive/SIFRank/auxiliary_data/weights.hdf5"

ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file, cuda_device=0)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
en_model = stanza.Pipeline(
    lang="ja", processors={}, use_gpu=True
)
elmo_layers_weight = [0.0, 1.0, 0.0]

text = "私の名前はジョンです。"
keyphrases = SIFRank(text, SIF, en_model, N=15,elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, en_model, N=15, elmo_layers_weight=elmo_layers_weight)

print(keyphrases)
print(keyphrases_)