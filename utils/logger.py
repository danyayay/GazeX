import time 
import logging


class TrainingLogger:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time.time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_iteration(self, epoch, iter, loss):
        logging.info(f"Epoch {epoch + 1}, iter {iter}, loss={loss:.4f}")
        
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs['epoch_time'] = elapsed_time  # Add epoch time to logs

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0:
            logging.info(f"Batch {batch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")
    
    def message(self, logs=None):
        logging.info(logs)