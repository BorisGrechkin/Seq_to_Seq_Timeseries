import matplotlib.pyplot as plt

class Visualization:


    @staticmethod
    def save_learning_loss(train_losses:list, validation_losses:list,
                           img_save_path:str):

        plt.plot(train_losses, label='Train Loss')
        plt.plot(validation_losses, label='Val Loss')
        plt.legend()
        plt.savefig(img_save_path)


    @staticmethod
    def save_predicted_and_actual_seq(predicted_seq:list, real_seq:list,
                                      img_save_path:str):

        assert len(predicted_seq) == len(real_seq), \
            'length of predicted and real seq do not match'

        plt.figure(figsize=(12, 4))
        plt.plot(predicted_seq, label='Real')
        plt.plot(real_seq, label='Predicted')
        plt.legend()
        plt.savefig(img_save_path)