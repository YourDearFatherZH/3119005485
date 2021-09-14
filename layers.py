#encoding=utf8
from keras.engine import Layer, InputSpec
import tensorflow as tf
import keras.backend as K
import keras

class SoftAttention(object):
    """
    Layer to compute local inference between two encoded sentences a and b.
    """

    def __call__(self, inputs):
        a = inputs[0]
        b = inputs[1]

        attention = keras.layers.Lambda(self._attention,
                                        output_shape = self._attention_output_shape,
                                        arguments = None)(inputs)

        align_a = keras.layers.Lambda(self._soft_alignment,
                                     output_shape = self._soft_alignment_output_shape,
                                     arguments = None)([attention, a])
        align_b = keras.layers.Lambda(self._soft_alignment,
                                     output_shape = self._soft_alignment_output_shape,
                                     arguments = None)([attention, b])

        return align_a, align_b

    def _attention(self, inputs):
        """
        Compute the attention between elements of two sentences with the dot
        product.
        Args:
            inputs: A list containing two elements, one for the first sentence
                    and one for the second, both encoded by a BiLSTM.
        Returns:
            A tensor containing the dot product (attention weights between the
            elements of the two sentences).
        """
        attn_weights = K.batch_dot(x=inputs[0],
                                   y=K.permute_dimensions(inputs[1],
                                                          pattern=(0, 2, 1)))
        return K.permute_dimensions(attn_weights, (0, 2, 1))

    def _attention_output_shape(self, inputs):
        input_shape = inputs[0]
        embedding_size = input_shape[1]
        return (input_shape[0], embedding_size, embedding_size)

    def _soft_alignment(self, inputs):
        """
        Compute the soft alignment between the elements of two sentences.
        Args:
            inputs: A list of two elements, the first is a tensor of attention
                    weights, the second is the encoded sentence on which to
                    compute the alignments.
        Returns:
            A tensor containing the alignments.
        """
        attention = inputs[0]
        sentence = inputs[1]

        # Subtract the max. from the attention weights to avoid overflows.
        exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
        exp_sum = K.sum(exp, axis=-1, keepdims=True)
        softmax = exp / exp_sum

        return K.batch_dot(softmax, sentence)

    def _soft_alignment_output_shape(self, inputs):
        attention_shape = inputs[0]
        sentence_shape = inputs[1]
        return (attention_shape[0], attention_shape[1], sentence_shape[2])


