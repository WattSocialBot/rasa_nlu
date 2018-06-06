from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import typing
from typing import Any

from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

from allennlp.commands.elmo import ElmoEmbedder

# TODO(asuglia): define a better way to specify the elmo embeddings
# options_file = os.path.join(os.path.dirname(__file__), "elmo", "elmo_2x4096_512_2048cnn_2xhighway_options.json")
# weight_file = os.path.join(os.path.dirname(__file__), "elmo", "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")

elmo = ElmoEmbedder(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
)


# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector

def ndim(spacy_nlp):
    """Number of features used to represent a document / sentence."""
    # type: Language -> int
    return 1024


def features_for_doc(doc):
    """Feature vector for a single document / sentence."""
    # type: Doc -> np.ndarray
    elmo_embeddings = elmo.embed_sentence([token.text for token in doc])

    return np.sum(np.sum(elmo_embeddings, 1), 0)


class ElmoFeaturizer(Featurizer):
    name = "intent_featurizer_elmo"

    provides = ["text_features"]

    requires = ["spacy_doc"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        for example in training_data.intent_examples:
            self._set_spacy_features(example)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        self._set_spacy_features(message)

    def _set_spacy_features(self, message):
        """Adds the spacy word vectors to the messages text features."""

        fs = features_for_doc(message.get("spacy_doc"))
        features = self._combine_with_existing_text_features(message, fs)
        message.set("text_features", features)
