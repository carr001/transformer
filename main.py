import threading
import random
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from datasets import load_dataset
from model.transformer import Transformer
from custom_dataset import CustomDataset


# Ru is source, En is target
dec_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-cased")  # english
enc_tokenizer = BertTokenizerFast.from_pretrained("DeepPavlov/rubert-base-cased")  # russian


en_tokens = dec_tokenizer("Hello, my name is Kirill").input_ids
print(en_tokens)
print(dec_tokenizer.decode(en_tokens))

ru_tokens = enc_tokenizer("Привет, меня зовут Кирилл. Ты понял?").input_ids
print(ru_tokens)
print(enc_tokenizer.decode(ru_tokens))


dataset = load_dataset("wmt/wmt16", 'ru-en')

print(dataset['train'][0])
print(dataset['validation'][0])
print(dataset['test'][0])

print(enc_tokenizer.eos_token)
print(enc_tokenizer.sep_token)
print(enc_tokenizer.cls_token_id)
print(enc_tokenizer.pad_token_id)
print(len(enc_tokenizer.get_vocab()))
print(enc_tokenizer.decode([77527]))



def tokenize_sequence(sequence, tokenizer: BertTokenizerFast, max_len, pad):
    if pad:
        return tokenizer(
            sequence, truncation=True, max_length=max_len, return_tensors='pt', padding="max_length"
        ).input_ids.squeeze()
    else:
        return tokenizer(
            sequence, truncation=True, max_length=max_len, return_tensors='pt'
        ).input_ids.squeeze()


def make_padding_mask(tokens_with_pad, tokens_without_pad, dtype):
    with_pad_len = tokens_with_pad.shape[0]
    no_pad_len = tokens_without_pad.shape[0]
    padding_mask = torch.zeros((with_pad_len, with_pad_len), dtype=dtype)  # mask is square matrix (len x len)

    # no_pad_len - 1 - the first position of pad (<eos> token if no padding is provided)
    # with_pad_len - 1 - the position of <eos> (<eos> is not masked)
    padding_mask[:, no_pad_len - 1 : with_pad_len - 1] = -torch.inf
    return padding_mask


def reform_dataset(
    dataset, source_tokenizer, target_tokenizer, max_len, mask_dtype, source_label='ru', target_label='en'
):
    new_dataset = {}
    for key in dataset.keys():
        new_dataset[key] = {"x": [], "y": [], "x_padding_mask": [], "y_padding_mask": []}

        i = 0
        for data in dataset[key]["translation"]:
            x_pad = tokenize_sequence(data[source_label].lower(), source_tokenizer, max_len, pad=True)
            y_pad = tokenize_sequence(data[target_label].lower(), target_tokenizer, max_len, pad=True)
            new_dataset[key]["x"].append(x_pad)
            new_dataset[key]["y"].append(y_pad)

            x_no_pad = tokenize_sequence(data[source_label].lower(), source_tokenizer, max_len, pad=False)
            y_no_pad = tokenize_sequence(data[target_label].lower(), target_tokenizer, max_len, pad=False)
            new_dataset[key]["x_padding_mask"].append(make_padding_mask(x_pad, x_no_pad, mask_dtype))
            new_dataset[key]["y_padding_mask"].append(make_padding_mask(y_pad, y_no_pad, mask_dtype))
            i += 1
            print(i)

    return new_dataset


def define_device():
    if torch.cuda.is_available():
        print("Running on GPU")
        return torch.device("cpu")
    else:
        print("Running on  CPU")
        return torch.device("cpu")


def save_model(state_dict, filename):
    print("Best val loss. Saving model...")
    torch.save(state_dict, filename)
    print("Model saved")


#print(tokenize_sequence("Привет, меня зовут Кирилл. Ты понял?", ru_tokenizer, 40, remove_cls=False))
# Define hyperparameters
epochs_amount = 9
batch_size = 16
init_lr = 1e-5
weight_decay = 5e-4
adam_eps = 5e-9
optim_scheduler_warmup = 100

device = define_device()
dtype = torch.float32

model_max_len = 256
d_model = 512
attention_d_head = 64
ffn_d_hidden = 1024
dropout_prob = 0.1


# Prepare data
print(dataset["train"]["translation"][0]["ru"])
with torch.no_grad():
    dataset = reform_dataset(
        dataset, enc_tokenizer, dec_tokenizer, model_max_len, dtype, source_label='ru', target_label='en'
    )
print(dataset["train"]["x"][0])
print(dec_tokenizer.decode(dataset["train"]["y"][0]))



# Define dataloaders
train_dataset = CustomDataset(
    dataset["train"]["x"], dataset["train"]["y"],
    dataset["train"]["x_padding_mask"], dataset["train"]["y_padding_mask"],
    device
)
val_dataset = CustomDataset(
    dataset["validation"]["x"], dataset["validation"]["y"],
    dataset["validation"]["x_padding_mask"], dataset["validation"]["y_padding_mask"],
    device
)
train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)


print(f"Test dataset size: {len(train_dataset)}")


# Define model
loss_function = nn.CrossEntropyLoss(ignore_index=dec_tokenizer.pad_token_id)
transformer = Transformer(
    3, 3,
    len(enc_tokenizer.get_vocab()), len(dec_tokenizer.get_vocab()),
    d_model, attention_d_head, ffn_d_hidden, dropout_prob, model_max_len, device, dtype,
    encoder_padding_index=enc_tokenizer.pad_token_id, decoder_padding_index=dec_tokenizer.pad_token_id
)
transformer = transformer.to(device)


optimizer = torch.optim.Adam(transformer.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=0.9, patience=10)


# Train and validation
train_losses, val_losses = [], []
best_val_loss = torch.inf
for epoch in range(epochs_amount):
    print(f"Epoch {epoch + 1}/{epochs_amount}")
    epoch_start = datetime.now()

    # train
    transformer.train()
    train_loss = 0
    train_step_start = datetime.now()
    for i, (train_x, train_y, train_x_pad_mask, train_y_pad_mask) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        logits = transformer(
            train_x, train_y[:, :-1],
            encoder_padding_mask=train_x_pad_mask, decoder_padding_mask=train_y_pad_mask[:, :-1, :-1]
        )  # all (including <BOS>) except <EOS> in target

        batch, tokens, channels = logits.shape

        train_y = train_y[:, 1:].reshape(batch * tokens)  # all (including <EOS>) except <BOS> in target
        logits = logits.view(batch * tokens, channels)

        loss = loss_function(logits, target=train_y)
        loss.backward()
        #nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.detach().cpu() / train_y.shape[0]
        if i % 50 == 0:
            print(f"\t\tTrain batch {i}")
    print(f"\tTrain step ended in: {datetime.now() - train_step_start}. Running loss is {train_loss}")
    train_losses.append(train_loss)

    # validation
    transformer.eval()
    with torch.no_grad():
        val_loss = 0
        val_step_start = datetime.now()

        for val_x, val_y, val_x_pad_mask, val_y_pad_mask in val_loader:
            logits = transformer(
                val_x, val_y[:, :-1],
                encoder_padding_mask=val_x_pad_mask, decoder_padding_mask=val_y_pad_mask[:, :-1, :-1]
            )  # all (including <BOS>) except <EOS> in target

            batch, tokens, channels = logits.shape
            val_y = val_y[:, 1:].reshape(batch * tokens)  # all (including <EOS>) except <BOS> in target
            logits = logits.view(batch * tokens, channels)

            loss = loss_function(logits, target=val_y)

            val_loss += loss.detach().cpu() / val_y.shape[0]
        print(f"\tValidation step ended in: {datetime.now() - val_step_start}. Running loss is {val_loss}")
        val_losses.append(val_loss)

        if epoch > optim_scheduler_warmup:  # Update lr
            scheduler.step(val_loss)

        if val_loss < best_val_loss:  # Save model
            print(f"Epoch: {epoch + 1} has the best val_loss")
            save_wieghts_daemon = threading.Thread(
                target=save_model, args=(transformer.state_dict(), './transformer_best_val_loss.pt'), daemon=True
            )
            save_wieghts_daemon.start()
            best_val_loss = val_loss


    print(f"\tEpoch {epoch + 1} ended in {datetime.now() - epoch_start}")

save_last_wieghts_daemon = threading.Thread(
    target=save_model, args=(transformer.state_dict(), './transformer_last_weights.pt')
)
save_last_wieghts_daemon.start()

# Check translation

used = []
with torch.no_grad():
    for i in range(10):
        while True:
            choice = random.randint(0, len(dataset["test"]["x"]) - 1)
            if choice not in used:
                used.append(choice)
                break

        x = dataset["test"]["x"][choice]
        y = dataset["test"]["y"][choice]
        x_pad_mask = dataset["test"]["x_padding_mask"][choice]
        pred_y = transformer.generate(
            x.to(device), dec_tokenizer.cls_token_id, dec_tokenizer.sep_token_id, encoder_padding_mask=x_pad_mask.to(device)
        )
        print(f"Number: {i}, element with index {choice}")
        print(f"\tRu: {enc_tokenizer.decode(x)}.\n\tEn: {dec_tokenizer.decode(y)}\n\tModel_prediction:{dec_tokenizer.decode(pred_y)}")

    save_wieghts_daemon.join()