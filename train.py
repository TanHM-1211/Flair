from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from utils import vn_char, punct
import torch
import pickle
import os

# are you training a forward or backward LM?
is_forward_lm = True
suffix = 'forward' if is_forward_lm else 'backward'

# load the character dictionary
dictionary: Dictionary = Dictionary()
for i in vn_char:
    dictionary.add_item(i)

# get your corpus, process forward and at the character level
if os.path.isfile('/mnt/disk1/tan_hm/saved_corpus.pkl'):
    with open('/mnt/disk1/tan_hm/saved_corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
else:
    corpus = TextCorpus('/mnt/disk1/tan_hm/corpus',
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    with open('/mnt/disk1/tan_hm/saved_corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=2048,
                               nlayers=1)

trainer = LanguageModelTrainer(language_model, corpus)


trainer.train('/mnt/disk1/tan_hm/Flair_language_model_' + suffix,
              sequence_length=256,
              mini_batch_size=200,
              max_epochs=100,
              learning_rate=5,
              clip=0.5,
              patience=10,
              checkpoint=True,
              num_workers=4)


