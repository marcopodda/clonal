"""Train Child-Sum Tree-LSTM model on the Stanford Sentiment Treebank."""
import globvar
from utils import parameters, training, history, tree, dataset
from modules import model


def main():
    params, arg_config = parameters.parse_arguments()

    # Get or create History
    hist = history.get(
        globvar.PICKLE_DIR, params.name, params.override, arg_config)

    # Report config to be used
    config = hist.config
    print(config)

    print('Loading data...')
    train_data, dev_data = dataset.get_data()
    train_loader = dataset.get_data_loader(train_data, config.batch_size)
    dev_loader = dataset.get_data_loader(dev_data, config.batch_size)

    print('Loading model...')
    clonal_model = model.ClonalModelRegressor(params.name, config)

    print('Loading trainer...')
    trainer = training.PyTorchTrainerRegression(
        clonal_model, hist, train_loader, dev_loader, globvar.CKPT_DIR)

    print('Training...')
    trainer.train()


if __name__ == '__main__':
    main()
