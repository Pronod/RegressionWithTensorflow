import tensorflow as tf

class Model:
    def __init__(self, hidden_layers_units, learning_rate=0.1):
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Normalization(axis=-1))

        for units in hidden_layers_units:
            self.model.add(tf.keras.layers.Dense(units=units, activation='relu'))

        self.model.add(tf.keras.layers.Dense(units=1))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=tf.keras.metrics.RootMeanSquaredError())
        
    def fit(self, X, Y, epochs, validation_split=0.2):
        return self.model.fit(X, Y, epochs=epochs, validation_split=validation_split)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)