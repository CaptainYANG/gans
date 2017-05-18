

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

    def train(self, config):
    	data_X, data_y = self.load_mnist()
    	d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
    	.minimize(self.d_loss, var_list=self.d_vars)
      g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
      .minimize(self.g_loss, var_list=self.g_vars)

      try:
      	tf.global_variables_initializer().run()
      except:
      	tf.initialize_all_variables().run()

      self.g_sum = merge_summary([self.z_sum, self.d__sum,
      	self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter("./logs", self.sess.graph)

      sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
      sample_inputs = data_X[0:self.sample_num]
      sample_labels = data_y[0:self.sample_num]

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
      	counter = checkpoint_counter
      	print(" [*] Load SUCCESS")
      else:
      	print(" [!] Load failed...")

      for epoch in xrange(config.epoch):
      	batch_idxs = min(len(data_X), config.train_size) // config.batch_size
      	for idx in xrange(0, batch_idxs):
      		batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
      		batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
      		batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
      		.astype(np.float32)
      		#update d 
      		_, summary_str = self.sess.run([d_optim, self.d_sum],
      			feed_dict={
      			self.inputs: batch_images,
      			self.z: batch_z,
      			self.y: batch_labels,
      			})
      		self.writer.add_summary(summary_str, counter)

      		#update g twice  y?
      		_, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
          counter += 1
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          	% (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 100) == 1:
          if config.dataset == 'mnist':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )
            # save samples
            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            save_images(samples, [manifold_h, manifold_w],
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      x = conv_cond_concat(image, yb)

      h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
      h0 = conv_cond_concat(h0, yb)

      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
      h1 = tf.reshape(h1, [self.batch_size, -1])      
      h1 = concat([h1, y], 1)
      
      h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
      h2 = concat([h2, y], 1)

      h3 = linear(h2, 1, 'd_h3_lin')
      
      return tf.nn.sigmoid(h3), h3

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
    	s_h, s_w = self.output_height, self.output_width
    	s_h2, s_h4 = int(s_h/2), int(s_h/4)
    	s_w2, s_w4 = int(s_w/2), int(s_w/4)

    	yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
    	z = concat([z, y], 1)
    	#100 -> 1024
    	h0 = tf.nn.relu(
    		self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
    	h0 = concat([h0, y], 1)

    	#1024 -> 64*2*7*7
    	h1 = tf.nn.relu(self.g_bn1(
    		linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
    	h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
    	h1 = conv_cond_concat(h1, yb)
    	
    	h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
    		[self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
    	h2 = conv_cond_concat(h2, yb)

    	return tf.nn.sigmoid(
    		deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

