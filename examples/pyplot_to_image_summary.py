"""
example of writing matplotlib plot to image summary
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tfu

plot_placeholder, summary_op = tfu.png_bytes_to_image_summary("plot")

writer = tf.summary.FileWriter('./logs')
with tf.Session() as sess:
    plt.plot([1, 2])
    plt.title("test")
    plot_bytes = tfu.pyplot_to_png_bytes()
    summary = sess.run(summary_op, feed_dict={plot_placeholder: plot_bytes})
    writer.add_summary(summary)
