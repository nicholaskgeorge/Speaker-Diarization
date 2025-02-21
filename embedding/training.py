import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


def objective_function(model, x_same, x_diff, K=1.0):
    """
    Implements equation 3 from the paper:
    E = -∑(x,y∈Psame) ln(Pr(x,y)) - K∑(x,y∈Pdiff) ln(1-Pr(x,y))
    
    Args:
        x_same: Pairs of embeddings from same speaker
        x_diff: Pairs of embeddings from different speakers
        K: Weight for different-speaker pairs (balances the loss)
    """
    # Get scores for same-speaker pairs
    same_scores = model.scoring.score_pairs(x_same[:, 0], x_same[:, 1])
    same_probs = tf.sigmoid(same_scores)
    
    # Get scores for different-speaker pairs
    diff_scores = model.scoring.score_pairs(x_diff[:, 0], x_diff[:, 1])
    diff_probs = tf.sigmoid(diff_scores)
    
    # Calculate negative log likelihood for same-speaker pairs
    same_loss = -tf.reduce_mean(tf.math.log(same_probs + 1e-10))
    
    # Calculate negative log likelihood for different-speaker pairs
    diff_loss = -K * tf.reduce_mean(tf.math.log(1 - diff_probs + 1e-10))
    
    return same_loss + diff_loss

@tf.function
def train_step(model, optimizer, x_same, x_diff):
    with tf.GradientTape() as tape:
        # Get embeddings for all segments
        embeddings_same = model(x_same, training=True)
        embeddings_diff = model(x_diff, training=True)
        
        # Calculate loss using objective function
        loss = objective_function(model, embeddings_same, embeddings_diff)
    
    # Update model parameters
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
def train(model, dataset, epochs=10):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for same_pairs, diff_pairs in dataset:
            loss = train_step(model, optimizer, same_pairs, diff_pairs)
            total_loss += loss
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Example usage:
def create_training_batches(features, speaker_labels, batch_size=16):
    """
    Creates batches with same-speaker and different-speaker pairs
    """
    same_pairs = []
    diff_pairs = []
    
    # Create same-speaker pairs
    for speaker in np.unique(speaker_labels):
        speaker_features = features[speaker_labels == speaker]
        indices = np.random.permutation(len(speaker_features))
        for i in range(0, len(indices)-1, 2):
            same_pairs.append([speaker_features[indices[i]], 
                             speaker_features[indices[i+1]]])
    
    # Create different-speaker pairs
    speakers = np.unique(speaker_labels)
    for i in range(len(speakers)):
        for j in range(i+1, len(speakers)):
            spk1_features = features[speaker_labels == speakers[i]]
            spk2_features = features[speaker_labels == speakers[j]]
            
            for k in range(min(len(spk1_features), len(spk2_features))):
                diff_pairs.append([spk1_features[k], spk2_features[k]])
    
    same_pairs = np.array(same_pairs)
    diff_pairs = np.array(diff_pairs)
    
    # Create tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((same_pairs, diff_pairs))
    return dataset.shuffle(1000).batch(batch_size)

# Initialize and train
model = SpeakerDiarizationModel()
dataset = create_training_batches(features, speaker_labels)
train(model, dataset)