
# Finetuning LLama 2

Questo progetto riguarda l'addestramento di un modello linguistico HuggingFace utilizzando il framework TRL (Transfer and Representation Learning). Lo scopo del progetto è ottimizzare un modello pre-addestrato su un dataset personalizzato per compiti di generazione di testo. Lo script principale di questo progetto è uno script Python che illustra il processo di addestramento. Viene fornito anche un semplice script per l'inferenza.

## Tabella dei Contenuti

- [Finetuning LLama 2](#finetuning-llama-2)
  - [Tabella dei Contenuti](#tabella-dei-contenuti)
  - [Introduzione](#introduzione)
  - [Primi Passi](#primi-passi)
    - [Prerequisiti](#prerequisiti)
    - [Installazione](#installazione)
  - [Utilizzo](#utilizzo)
    - [Prima di iniziare](#prima-di-iniziare)
    - [Argomenti da Riga di Comando](#argomenti-da-riga-di-comando)
    - [Esempio di utilizzo](#esempio-di-utilizzo)
    - [Esecuzione](#esecuzione)
  - [Inferenza](#inferenza)
  - [Contributori](#contributori)
  - [Licenza](#licenza)

## Introduzione

Il progetto si concentra sull'ottimizzazione di un modello LLama 2. Il modello viene addestrato su un dataset personalizzato per eseguire compiti di generazione di testo in lingua italiana. Il processo di addestramento include il caricamento del modello, del dataset e degli argomenti di addestramento, seguito dall'esecuzione dell'addestramento. Opzionalmente, è possibile utilizzare adattatori LoRA (Learned Randomized Adapters) per l'addestramento.

## Primi Passi

### Prerequisiti

Prima di iniziare, assicurati di avere le seguenti dipendenze installate:

- Python 3.x
- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- Libreria PEFT (Parallel Efficient Training)
- tqdm
- Libreria `huggingface_hub`

### Installazione

1. Clona la repository:
   ```shell
   git clone https://github.com/itsrocchi/finetuning-llama2-ita
   cd finetuning-llama2-ita
   ```

2. Installa le dipendenze Python necessarie:
   ```shell
   pip install -r requirements.txt
   ```

## Utilizzo

### Prima di iniziare

Prima di lanciare lo script è necessario modificarlo aggiungendo parametri personalizzati, ovvero gli hf_token e il nome del nuovo modello

### Argomenti da Riga di Comando

Lo script principale supporta diversi argomenti da riga di comando per personalizzare il processo di addestramento. Gli argomenti seguenti sono disponibili:

- `model_name`: Nome del modello pre-addestrato (default: "meta-llama/Llama-2-7b-chat-hf").
- `dataset_name`: Nome del dataset personalizzato (default: "seeweb/Seeweb-it-292-forLLM").
- `dataset_text_field`: Nome del campo di testo nel dataset (default: "text").
- `log_with`: Specifica come registrare i dati di addestramento ("wandb" per utilizzare Weights & Biases, altrimenti niente registrazione).
- `learning_rate`: Tasso di apprendimento (default: 1.41e-5).
- `batch_size`: Dimensione del batch (default: 64).
- `seq_length`: Lunghezza massima della sequenza in input (default: 512).
- `gradient_accumulation_steps`: Numero di passaggi di accumulo del gradiente (default: 16).
- `load_in_8bit`: Carica il modello con precisione a 8 bit (default: False).
- `load_in_4bit`: Carica il modello con precisione a 4 bit (default: False).
- `use_peft`: Abilita l'utilizzo di PEFT (Parallel Efficient Training) per addestrare gli adattatori (default: False).
- `trust_remote_code`: Abilita `trust_remote_code` (default: True).
- `output_dir`: Directory di output per i risultati dell'addestramento (default: "output").
- `peft_lora_r`: Parametro "r" degli adattatori LoRA (default: 64).
- `peft_lora_alpha`: Parametro "alpha" degli adattatori LoRA (default: 16).
- `logging_steps`: Intervallo di passaggi di addestramento per la registrazione (default: 1).
- `use_auth_token`: Utilizza il token di autenticazione di HuggingFace per accedere al modello (default: True).
- `num_train_epochs`: Numero di epoche di addestramento (default: 3).
- `max_steps`: Numero massimo di passaggi di addestramento (default: -1, senza limite).

Ricorda che puoi utilizzare questi argomenti da riga di comando per personalizzare il processo di addestramento in base alle tue esigenze.

### Esempio di utilizzo

```shell
python finetuner.py \
--model_name meta-llama/Llama-2-7b-chat-hf \
--dataset_name seeweb/Seeweb-it-292-forLLM \
--load_in_4bit \
--use_peft \
--batch_size 4 \
--gradient_accumulation_steps 1 \
--num_train_epochs 10
```

### Esecuzione

Lo script caricherà il modello pre-addestrato in base al `model_name` specificato, verrà poi caricato il Dataset specificato e quest'ultimo verrà utilizzato per il fine-tuning del modello precedentemente caricato. Dopo l'addestramento il modello verrà salvato nella cartella specificata nel codice e verrà caricato su HuggingFace sull'account corrispondente ai token inseriti.

## Inferenza

Nella repository è presente uno script per eseguire la generazione di testo utilizzando il modello ottimizzato. Per utilizzarlo basta semplicemente modificare i parametri inserendo la directory del modello e modificando il prompt e una volta lanciato lo script caricherà il modello e genererà testo basandosi sul prompt.


## Contributori

- Lorenzo Rocchi (Seeweb s.r.l)

## Licenza

This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/itsrocchi/finetuning-llama2-ita/blob/03359c5e96673611a9632d3ae5d42598151a25f9/LICENSE) for more details.
