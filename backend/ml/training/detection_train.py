#created by Arn Christian S. Rosales
#skin lens detection model
#05-01-2026
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

#constant value
img_size = (224, 224)
batch_size = 32
seed = 42

# Get absolute paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Go up to ml/ directory

#dataset directory
train_dir = os.path.join(parent_dir, "dataset", "detection", "train")
val_dir = os.path.join(parent_dir, "dataset", "detection", "valid")
test_dir = os.path.join(parent_dir, "dataset", "detection", "test")

#model path
model_path = os.path.join(parent_dir, "models", "detection", "skinlens_detection.keras")

#load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=seed
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode="categorical",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

#class labels (each folder from dataset)
class_labels = train_ds.class_names
print("Class labels:", class_labels)

#image processing
def normalize_batch(images, labels):
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

train_ds = train_ds.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)

#performance optimization
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

#handling class imbalance
y_train = np.concatenate([y.numpy() for _, y in train_ds.unbatch().batch(10000)], axis=0)
y_train = np.argmax(y_train, axis=1).astype(int).ravel()
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weights)

#data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
], name="augmentation")

#mobilenetv2 base model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

#layers from base model
inputs = tf.keras.layers.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
model = tf.keras.models.Model(inputs, outputs)

#focal loss
focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=0.25
)

#model compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss,
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)
model.summary()

#callbacks overlifting prevention
cb = [
    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

#training phase 1
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=cb
)

#fine tuning
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

#compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss,
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

#training phase 2
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=cb
)

#evaluation
results = model.evaluate(test_ds, verbose=1)
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)
    y_pred.extend(preds)
    y_true.extend(np.argmax(labels.numpy(), axis=1))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    digits=4
))

model.save(model_path)
print(f"\nSaved model to {model_path}")