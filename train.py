from utils import *
from seq2seq_with_attention import Seq2Seq
from seq2seq_without_attention import Seq2Seq as Seq2SeqWithoutAttention
import argparse
import pytorch_lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
import torch


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Train a Seq2Seq model for transliteration')
    parser.add_argument('--data_dir', type=str, default="dakshina_dataset_v1.0_hi/lexicons", help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--embedding_dimension', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--hidden_layer_size', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the encoder and decoder')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--cell_type', type=str, default="LSTM", help='Type of RNN cell to use (RNN, LSTM, GRU)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='Teacher forcing ratio')
    parser.add_argument('--use_attention', action='store_true', help='Use attention mechanism')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train')

    args = parser.parse_args()

    # Load data module
    data_module = TrasnliterationDataModule(args.data_dir, args.batch_size)

    # Initialize the appropriate model
    ModelClass = Seq2Seq if args.use_attention else Seq2SeqWithoutAttention
    model = ModelClass(
        input_embedding_size=len(data_module.source_vocab),
        output_embedding_size=len(data_module.target_vocab),
        embedding_dimension=args.embedding_dimension,
        hidden_layer_size=args.hidden_layer_size,
        number_of_layers_encoder=args.num_layers,
        number_of_layers_decoder=args.num_layers,
        dropout=args.dropout,
        cell_type=args.cell_type,
        learning_rate=args.learning_rate,
        teacher_forcing_ratio=args.teacher_forcing_ratio
    )

    # Set up WandB logger
    wandb_logger = WandbLogger(project="da6401_assignment3_v1", log_model=True)

    # Trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        precision="16-mixed",
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the model
    model_filename = "seq2seq_with_attention.pkl" if args.use_attention else "seq2seq_without_attention.pkl"
    torch.save(model.state_dict(), model_filename)

    # Finish WandB run
    wandb.finish()
