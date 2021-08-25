import tensorflow as tf
        
def get_pixel_value(img, idx):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (H, W, C)
    - idx: flattened tensor of shape (N)

    Returns
    -------
    - output: tensor of shape (H, W, C)
    """
    img_reshape = tf.reshape(img, shape=(-1, 2))
    return tf.gather(img_reshape, idx)

def grid_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    W = tf.shape(img)[0]
    H = tf.shape(img)[1]
    max_x = tf.cast(W, 'int32')
    max_y = tf.cast(H, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    with tf.Session() as sess:
        print(x.eval())
        print(y.eval())

    x = x * tf.cast(max_x, 'float32')
    y = y * tf.cast(max_y, 'float32')
    with tf.Session() as sess:
        print(x.eval())
        print(y.eval())

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x - 1)
    x1 = tf.clip_by_value(x1, zero, max_x - 1)
    y0 = tf.clip_by_value(y0, zero, max_y - 1)
    y1 = tf.clip_by_value(y1, zero, max_y - 1)

    bl_idx = tf.math.add(tf.math.scalar_mul(W, y0), x0)
    br_idx = tf.math.add(tf.math.scalar_mul(W, y0), x1)
    tl_idx = tf.math.add(tf.math.scalar_mul(W, y1), x0)
    tr_idx = tf.math.add(tf.math.scalar_mul(W, y1), x1)

    Ibl = get_pixel_value(img, bl_idx)
    Ibr = get_pixel_value(img, br_idx)
    Itl = get_pixel_value(img, tl_idx)
    Itr = get_pixel_value(img, tr_idx)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=1)
    wb = tf.expand_dims(wb, axis=1)
    wc = tf.expand_dims(wc, axis=1)
    wd = tf.expand_dims(wd, axis=1)

    # compute output
    out = tf.add_n([wa*Ibl, wb*Ibr, wc*Itl, wd*Itr])

    return out



grid = tf.constant(1., shape = [64, 64, 2])
x = tf.linspace(0.4, 0.6, 21, name="linspace")
y = tf.linspace(0.4, 0.6, 21, name="linspace")


vals = grid_sampler(grid, x, y)

with tf.Session() as sess:
    print(vals.eval())