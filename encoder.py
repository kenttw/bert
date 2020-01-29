import tensorflow as tf
import tensorflow_hub as hub
import run_classifier
import tokenization
import test_data

class Encoder:
    def __init__(self,BERT_MODEL_HUB= "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", MAX_SEQ_LENGTH = 128):
        # This is a path to an uncased (all lowercase) version of BERT
        self.BERT_MODEL_HUB = BERT_MODEL_HUB
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.BATCH_SIZE = 32
        self.tokenizer = self.create_tokenizer_from_hub_module()

        model_fn = self.model_fn_builder()
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params={"batch_size": self.BATCH_SIZE})

    def create_tokenizer_from_hub_module(self):
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = hub.Module(self.BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])

        return tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)




    def create_model(self, input_ids, input_mask, segment_ids):
      """Creates a classification model."""

      bert_module = hub.Module(
          self.BERT_MODEL_HUB,
          trainable=True)
      bert_inputs = dict(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids)
      bert_outputs = bert_module(
          inputs=bert_inputs,
          signature="tokens",
          as_dict=True)

      # Use "pooled_output" for classification tasks on an entire sentence.
      # Use "sequence_outputs" for token-level output.
      output_layer = bert_outputs["pooled_output"]
      return output_layer

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def model_fn_builder(self):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        encode_vec = self.create_model(input_ids, input_mask, segment_ids,)

        predictions = {
            'encode_vec': encode_vec,
        }
        return tf.estimator.EstimatorSpec(mode,predictions=predictions)

      # Return the actual model function in the closure
      return model_fn



    def getPrediction(self, in_sentences):
      label_list = [0, 1]
      input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
      input_features = run_classifier.convert_examples_to_features(input_examples, label_list, self.MAX_SEQ_LENGTH, self.tokenizer)
      predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=self.MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
      predictions = self.estimator.predict(predict_input_fn)
      return [(sentence, prediction['encode_vec'] ) for sentence, prediction in zip(in_sentences, predictions)]


    def doPrediction(self,data):

        predictions = self.getPrediction(test_data)

        print(predictions)

if __name__ == "__main__":
    test_data = test_data.test_data
    enc = encoder()
    enc.doPrediction(test_data)


