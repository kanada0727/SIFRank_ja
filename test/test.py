import sys
sys.path.append('/content/drive/My Drive/SIFRank_ja')
sys.path.append('/content/drive/My Drive/SIFRank_ja/embeddings')
import stanza
import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus

#download from https://allennlp.org/elmo
options_file = "https://exawizardsallenlp.blob.core.windows.net/data/options.json"
weight_file = "/content/drive/My Drive/SIFRank_ja/auxiliary_data/weights.hdf5"

ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file, cuda_device=0)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
ja_model = stanza.Pipeline(
    lang="ja", processors={}, use_gpu=True
)
elmo_layers_weight = [0.0, 1.0, 0.0]

text = "ここにテキストを入力してください。"
keyphrases = SIFRank(text, SIF, ja_model, N=5,elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, ja_model, N=5, elmo_layers_weight=elmo_layers_weight)

print(keyphrases)
print(keyphrases_)
