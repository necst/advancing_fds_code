from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier


def __make_neural_network(
    fl_neurons=32,
    sl_neurons=16,
    activation_funct="relu",
    dropout_rate=0.1,
    input_shape=(102,),
    n_jobs=-1
):
    """Returns a neural network with two hidden layers with dropout on the first layer."""
    print(
        f"DEBUG: {fl_neurons}, {sl_neurons}, {activation_funct}, "
        f"{dropout_rate}, {input_shape}"
    )
    model = Sequential()

    # first layer
    model.add(
        Dense(
            units=fl_neurons,
            activation=activation_funct,
            input_shape=input_shape,
        )
    )

    # normalization layer
    model.add(BatchNormalization())

    # dropout layer
    model.add(Dropout(dropout_rate))

    # second layer
    model.add(Dense(units=sl_neurons, activation=activation_funct))

    # normalization layer
    model.add(BatchNormalization())

    # final classification layer
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(
        # loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", c_acc]
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def make_network_keras_wrap(
    input_shape, build_fn=__make_neural_network, **params
):
    """Returns a neural network classifier with scikit-learn API wrapper."""
    return KerasClassifier(build_fn=build_fn, input_shape=input_shape, **params)
