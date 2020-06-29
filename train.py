from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from utils import vn_char, punct
# are you training a forward or backward LM?
is_forward_lm = False

# load the character dictionary
dictionary: Dictionary = Dictionary()
for i in vn_char:
    dictionary.add_item(i)

# get your corpus, process forward and at the character level
corpus = TextCorpus('/content/drive/My Drive/aimesoft_training/Flair/corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)


# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=2048,
                               nlayers=1)

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('language_model',
              sequence_length=256,
              mini_batch_size=100,
              max_epochs=1,
              learning_rate=20,
              checkpoint=True,
              num_workers=1)

