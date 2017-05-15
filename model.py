

class DCGAN(object):
	def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
		batch_size=64, sample_num = 64, output_height=64, output_width=64,
    y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
    gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
    input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
	self.sess = sess
  self.is_crop = is_crop
  self.is_grayscale = (c_dim == 1)

  self.batch_size = batch_size
  self.sample_num = sample_num

  self.input_height = input_height
  self.input_width = input_width
  self.output_height = output_height
  self.output_width = output_width

  self.y_dim = y_dim
  self.z_dim = z_dim

  self.gf_dim = gf_dim
  self.df_dim = df_dim

  self.gfc_dim = gfc_dim
  self.dfc_dim = dfc_dim

  self.c_dim = c_dim

  self.d_bn1 = batch_norm(name='d_bn1')
  self.d_bn2 = batch_norm(name='d_bn2')

	if not self.y_dim:
	  self.d_bn3 = batch_norm(name='d_bn3')
	  self.g_bn0 = batch_norm(name='g_bn0')
	  self.g_bn1 = batch_norm(name='g_bn1')
	  self.g_bn2 = batch_norm(name='g_bn2')

	if not self.y_dim:
	  self.g_bn3 = batch_norm(name='g_bn3')

	  self.dataset_name = dataset_name
	  self.input_fname_pattern = input_fname_pattern
	  self.checkpoint_dir = checkpoint_dir
	  self.build_model()

	 def build_model(self):
	 	self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
	 	image_dims = [self.input_height, self.input_width, self.c_dim]
	 	self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')
    inputs = self.inputs
    sample_inputs = self.sample_inputs
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G = self.generator(self.z, self.y)
    self.D, self.D_logits = \
    self.discriminator(inputs, self.y, reuse=False)
    self.sampler = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = \
    self.discriminator(self.G, self.y, reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

