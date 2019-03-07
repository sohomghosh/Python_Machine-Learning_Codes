model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 2.0, 3.0], dtype=float)
ys = np.array([2.0, 4.0, 5.0], dtype=float)

model.fit(xs, ys, epochs = 500)
