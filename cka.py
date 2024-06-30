# CKA functions
def gram_matrix(X):
    return tf.linalg.matmul(X, X, transpose_b=True)

def center_gram_matrix(K):
    n = tf.shape(K)[0]
    one_n = tf.ones((n, n), dtype=tf.float32) / tf.cast(n, tf.float32)
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n

def hsic(K, L):
    return tf.linalg.trace(tf.linalg.matmul(K, L))

def cka_loss(X, Y):
    K = gram_matrix(X)
    L = gram_matrix(Y)
    Kc = center_gram_matrix(K)
    Lc = center_gram_matrix(L)
    hsic_xy = hsic(Kc, Lc)
    hsic_xx = hsic(Kc, Kc)
    hsic_yy = hsic(Lc, Lc)
    return -hsic_xy / tf.sqrt(hsic_xx * hsic_yy)
