from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from mlpipe.data_reader.mongodb import load_ids, MongoDBGenerator
from mlpipe.utils import MLPipeLogger, Config
from mlpipe.callbacks import SaveToMongoDB, UpdateManager
from examples.cifar10.processor import PreProcessData

EPOCH_NUMBER = 3

if __name__ == "__main__":
    MLPipeLogger.init()

    try:
        Config.add_config('./examples/cifar10/config.ini')
        collection_details = ("localhost_mongo_db", "cifar10", "train")

        # Create Data Generators
        train_data, val_data = load_ids(
            collection_details,
            data_split=(70, 30),
            limit=1000,
        )

        processors = [PreProcessData()]
        train_gen = MongoDBGenerator(
            collection_details,
            train_data,
            batch_size=128,
            processors=processors
        )
        val_gen = MongoDBGenerator(
            collection_details,
            val_data,
            batch_size=128,
            processors=processors
        )

        # Configure Model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

        # Save to MongoDB callback
        save_to_mongodb_cb = SaveToMongoDB(("localhost_mongo_db", "models"), "test", model)

        # To update to MLPipe-Manager use this callback
        update_manager_cb = UpdateManager("test", model, EPOCH_NUMBER, len(train_gen))

        model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            epochs=EPOCH_NUMBER,
            verbose=1,
            callbacks=[save_to_mongodb_cb],
            initial_epoch=0,
        )

    except Exception as e:
        MLPipeLogger.logger.exception(e)
