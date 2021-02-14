import numpy as np
import os
import tensorflow as tf
import time

from cora_data import CoraData
from graph_sampling import generate_batch_nodes
from models import NodeClassifier


def train_one_epoch(model, optimizer, loss_func, train_nodes, full_labels, full_adjList, features_map, batch_size,
                    num_sampled_neighbors, K):

    loss_metric = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    num_train_nodes = len(train_nodes)
    for batch_idx in range(num_train_nodes // batch_size):
        # sample nodes for forward pass
        sampled_nodes = np.random.choice(train_nodes, replace=False, size=(batch_size,))
        sampled_nodes_targets = full_labels[sampled_nodes]

        batch_nodes, batch_adjList = generate_batch_nodes(sampled_nodes, full_adjList, num_sampled_neighbors, K)
        inputs = (batch_nodes, batch_adjList, features_map)

        # forward pass and then backward pass for gradients and updates
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss_func(y_true=sampled_nodes_targets, y_pred=logits)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # record metrics
        loss_metric.update_state(loss_value)
        accuracy_metric.update_state(sampled_nodes_targets, logits)

    epoch_loss = round(loss_metric.result().numpy(), 5)
    epoch_accuracy = round(accuracy_metric.result().numpy(), 5)
    return epoch_loss, epoch_accuracy


def evaluate(model, nodes, y, full_adjList, features_map, loss_func, K):

    batch_nodes, batch_adjList = generate_batch_nodes(nodes, full_adjList, None, K)

    inputs = (batch_nodes, batch_adjList, features_map)
    logits = model(inputs)

    loss_value = loss_func(y_true=y, y_pred=logits)
    accuracy_value = tf.reduce_mean(
        tf.cast(
            tf.argmax(logits, axis=1) == y,
            tf.float32
        )
    )

    loss_value = round(loss_value.numpy(), 5)
    accuracy_value = round(accuracy_value.numpy(), 5)
    return loss_value, accuracy_value


def train_one_model(data, features_map, num_classes, K, input_dim, hidden_dims, num_sampled_neighbors, agg_method,
                    learning_rate, num_epochs=10, batch_size=128):

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = NodeClassifier(
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        K=K,
        agg_method=agg_method
    )

    epoch_train_loss = [0] * num_epochs
    epoch_train_accuracy = [0] * num_epochs
    epoch_val_loss = [0] * num_epochs
    epoch_val_accuracy = [0] * num_epochs

    for epoch_idx in range(num_epochs):
        start_time = time.time()

        # train one epoch
        train_loss_i, train_accuracy_i = train_one_epoch(
            model,
            optimizer,
            loss_func,
            data.train_nodes,
            data.labels,
            data.adjList,
            features_map,
            batch_size,
            num_sampled_neighbors,
            K
        )

        time_elapsed = round(time.time() - start_time, 2)
        epoch_train_loss[epoch_idx] = train_loss_i
        epoch_train_accuracy[epoch_idx] = train_accuracy_i

        # evaluate on validation
        val_loss_i, val_accuracy_i = evaluate(
            model,
            data.val_nodes,
            data.labels[data.val_nodes],
            data.adjList,
            features_map,
            loss_func,
            K
        )
        epoch_val_loss[epoch_idx] = val_loss_i
        epoch_val_accuracy[epoch_idx] = val_accuracy_i

        if epoch_idx % 5 == 1:
            print(f'epoch {epoch_idx} took {time_elapsed} seconds')
            print(f'train loss: {train_loss_i}, train accuracy: {train_accuracy_i}')
            print(f'val loss: {val_loss_i}, val accuracy: {val_accuracy_i}')
            print(' ')


    return model, epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy
