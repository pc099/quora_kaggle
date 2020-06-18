import pandas as pd
import torch
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader,RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

train_data_path = r'input_data/train.csv'
test_data_path = r'input_data/test.csv'
save_path = r'input_data'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

sentences = train_df['question_text'][:10000].values
labels =train_df['target'][:10000].values


print('loading Bert Tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print('original: ',sentences[0])
print('Tokenizer: ', tokenizer.tokenize((sentences[0])))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
'''
We are required to:

1) Add special tokens to the start and end of each sentence.
2) Pad & truncate all sentences to a single constant length.
3) Explicitly differentiate real tokens from padding tokens with the "attention mask".'''


input_ids = []
# for every sentence.....
for sent in sentences:
    encoded_sentence = tokenizer.encode(sent,add_special_tokens=True, )
    input_ids.append(encoded_sentence)
"""`encode` will:
  (1) Tokenize the sentence.
  (2) Prepend the `[CLS]` token to the start.
  (3) Append the `[SEP]` token to the end.
  (4) Map tokens to their IDs.
  add_special_token adds [CLS], [SEP]
  # This function also supports truncation and conversion
                        # to pytorch tensors, but we need to do padding, so we
                        # can't use these features :( .
                        #max_length = 128,          # Truncate all sentences.
                        #return_tensors = 'pt',     # Return pytorch tensors
  """

print('original :', sentences[0])
print('token IDs :', input_ids[0])

print('Max sentence length: ', max([len(sen) for sen in input_ids]))

# for padding the sentences

MAX_LEN = 64
print('\nPassing/truncating all sentences to %d values...' % MAX_LEN)
print('\nPadding token"{:}", ID: {:} '.format(tokenizer.pad_token, tokenizer.pad_token_id))

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype= 'long', value=0, truncating='post', padding='post')
print('\n Done.')

# attention_masks
'''The attention mask simply makes it explicit which tokens are actual words versus which are padding.
The BERT vocabulary does not use the ID 0, so if a token ID is 0, then it's padding, 
and otherwise it's a real token.'''
attention_mask = []
for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_mask.append(att_mask)

# test train split

train_inputs,validation_inputs,train_labels,validation_labels  = train_test_split(input_ids, labels, random_state=42, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_mask, labels, random_state=42, test_size=0.1)

#Our model expects PyTorch tensors rather than numpy.ndarrays, so convert all of our dataset variables.

train_masks = torch.tensor(train_masks).long()
validation_masks = torch.tensor(validation_masks).long()

train_inputs = torch.tensor(train_inputs).long()
validation_inputs = torch.tensor(validation_inputs).long()

train_labels = torch.tensor(train_labels).long()
validation_labels = torch.tensor(validation_labels).long()

# We'll also create an iterator for our dataset using the torch DataLoader class.
# This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory.


# The DataLoader needs to know our batch size for training, so we specify it
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

batch_size =16
# batch_size =32
# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs,train_masks ,train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks,validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False,
                                                      output_hidden_states=False)

# In the below cell, I've printed out the names and dimensions of the weights for:
    # The embedding layer.
    # The first of the twelve transformers.
    # The output layer.
'''The BERT model has 201 different named parameters.

==== Embedding Layer ====

bert.embeddings.word_embeddings.weight                  (30522, 768)
bert.embeddings.position_embeddings.weight                (512, 768)
bert.embeddings.token_type_embeddings.weight                (2, 768)
bert.embeddings.LayerNorm.weight                              (768,)
bert.embeddings.LayerNorm.bias                                (768,)

==== First Transformer ====

bert.encoder.layer.0.attention.self.query.weight          (768, 768)
bert.encoder.layer.0.attention.self.query.bias                (768,)
bert.encoder.layer.0.attention.self.key.weight            (768, 768)
bert.encoder.layer.0.attention.self.key.bias                  (768,)
bert.encoder.layer.0.attention.self.value.weight          (768, 768)
bert.encoder.layer.0.attention.self.value.bias                (768,)
bert.encoder.layer.0.attention.output.dense.weight        (768, 768)
bert.encoder.layer.0.attention.output.dense.bias              (768,)
bert.encoder.layer.0.attention.output.LayerNorm.weight        (768,)
bert.encoder.layer.0.attention.output.LayerNorm.bias          (768,)
bert.encoder.layer.0.intermediate.dense.weight           (3072, 768)
bert.encoder.layer.0.intermediate.dense.bias                 (3072,)
bert.encoder.layer.0.output.dense.weight                 (768, 3072)
bert.encoder.layer.0.output.dense.bias                        (768,)
bert.encoder.layer.0.output.LayerNorm.weight                  (768,)
bert.encoder.layer.0.output.LayerNorm.bias                    (768,)

==== Output Layer ====

bert.pooler.dense.weight                                  (768, 768)
bert.pooler.dense.bias                                        (768,)
classifier.weight                                           (2, 768)
classifier.bias                                                 (2,)
'''
# Now that we have our model loaded we need to grab the training hyperparameters from within the stored model.
# For the purposes of fine-tuning, the authors recommend choosing from the following values:
# Batch size: 16, 32 (We chose 32 when creating our DataLoaders).
# Learning rate (Adam): 5e-5, 3e-5, 2e-5 (We'll use 2e-5).
# Number of epochs: 2, 3, 4 (We'll use 4).
# The epsilon parameter eps = 1e-8 is "a very small number to prevent any division by zero in the implementation"

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-5)


epochs = 4
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=0,t_total=total_steps)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

# store the avg loss after each epoch so we can plot them.

loss_value = []
# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_value.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

torch.save(model.state_dict(),save_path)
'''model.load_state_dict(torch.load(save_path))
model.eval()'''

# for epoch_i in range(0,epochs):
#     t0 = time.time()
#     total_loss = 0
#     # Put the model into training mode. Don't be mislead--the call to
#     # `train` just changes the *mode*, it doesn't *perform* the training.
#     # `dropout` and `batchnorm` layers behave differently during training
#     # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
#     model.train()
# #     for each batch of training data ..
#     for step, batch in enumerate(train_dataloader):
#         # Progress update every 40 batches.
#         if step% 40 == 0 and not step== 0:
#             elapsed_time = format_time(time.time() - t0)
#             # report progess
#             print(' Batch {:>5,} of {:>5,}. Elapsed {:}.' .format(step, len(train_dataloader), elapsed_time))
#             # Unpack this training batch from our dataloader.
#             #
#             # As we unpack the batch, we'll also copy each tensor to the GPU using the
#             # `to` method.
#             #
#             # `batch` contains three pytorch tensors:
#             #   [0]: input ids
#             #   [1]: attention masks
#             #   [2]: labels
#             b_input_ids = batch[0].to(device)
#             b_input_mask = batch[1].to(device)
#             b_labels_ids = batch[2].to(device)
#             # Always clear any previously calculated gradients before performing a
#             # backward pass. PyTorch doesn't do this automatically because
#             # accumulating the gradients is "convenient while training RNNs".
#             # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
#             model.zero_grad()
#             # Perform a forward pass (evaluate the model on this training batch).
#             # This will return the loss (rather than the model output) because we
#             # have provided the `labels`.
#             # The documentation for this `model` function is here:
#             # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
#             outputs = model(b_input_ids, token_type_ids=True, attention_mask = b_input_mask, labels= b_labels_ids)
#             # The call to `model` always returns a tuple, so we need to pull the
#             # loss value out of the tuple.
#             loss = outputs[0]
#             # Accumulate the training loss over all of the batches so that we can
#             # calculate the average loss at the end. `loss` is a Tensor containing a
#             # single value; the `.item()` function just returns the Python value
#             # from the tensor.
#             total_loss += loss.item()
#             # Perform a backward pass to calculate the gradients.
#             loss.backward()
#             # Clip the norm of the gradients to 1.0.
#             # This is to help prevent the "exploding gradients" problem.
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             # Update parameters and take a step using the computed gradient.
#             # The optimizer dictates the "update rule"--how the parameters are
#             # modified based on their gradients, the learning rate, etc.
#             scheduler.step()
# #         calculate the avg loss over the training data.
#         avg_train_loss = total_loss / len(train_dataloader)
# #         store the loss value for plotting the learing curve
#         loss_value.append(avg_train_loss)
#         print('')
#         print('Average training loss: {0:.2f}'.format(avg_train_loss))
#         print('training epoch took: {:}'.format(format_time(time.time() - t0)))
#         # ========================================
#         #               Validation
#         # ========================================
#         # After the completion of each training epoch, measure our performance on
#         # our validation set.
#         print('')
#         print('Running Validation....')
#
#         t0 = time.time()
#         # Put the model in evaluation mode--the dropout layers behave differently
#         # during evaluation.
#         model.eval()
#         # Tracking variables
#         eval_loss, eval_accuracy = 0, 0
#         nb_eval_steps, nb_eval_examples = 0, 0
#         # Evaluate data for one epoch
#         for batch in validation_dataloader:
#             # Add batch to GPU
#             batch = tuple(t.to(device) for t in batch)
#
#             # Unpack the inputs from our dataloader
#             b_input_ids, b_input_mask, b_labels = batch
#
#             # Telling the model not to compute or store gradients, saving memory and
#             # speeding up validation
#             with torch.no_grad():
#                 # Forward pass, calculate logit predictions.
#                 # This will return the logits rather than the loss because we have
#                 # not provided labels.
#                 # token_type_ids is the same as the "segment ids", which
#                 # differentiates sentence 1 and 2 in 2-sentence tasks.
#                 # The documentation for this `model` function is here:
#                 # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
#                 outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)
#
#             # Get the "logits" output by the model. The "logits" are the output
#             # values prior to applying an activation function like the softmax.
#             logits = outputs[0]
#
#             # Move logits and labels to CPU
#             logits = logits.detach().cpu().numpy()
#             label_ids = b_labels.to('cpu').numpy()
#
#             # Calculate the accuracy for this batch of test sentences.
#             tmp_eval_accuracy = flat_accuracy(logits, label_ids)
#
#             # Accumulate the total accuracy.
#             eval_accuracy += tmp_eval_accuracy
#
#             # Track the number of batches
#             nb_eval_steps += 1
#
#         # Report the final accuracy for this validation run.
#         print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
#         print("  Validation took: {:}".format(format_time(time.time() - t0)))
#
# print("")
# print("Training complete!")





