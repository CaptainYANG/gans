import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.flags.FLAGS
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use [28]")
flags.DEFINE_integer("input_width", None, "The size of image to use. If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [28]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS
def main(argv=None):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.session() as sess:
        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            samples=FLAGS.batch_size,
            c_dim=FLAGS.c_dim,
            y_dim=10,
            dataset=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            is_crop=FLAGS.is_crop,
            sample_dir=FLAGS.sample_dir)

    def show_all_variables():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    show_all_variables()
    if FLAGS.is_train:
        dcgan.train(FLAGS)
    else:
        if not dcgan.load(FLAGS.checkpoint_dir):
            raise Exception("[!] Train a model first, then run test mode")
if __name__ == "__main__":
    tf.app.run()