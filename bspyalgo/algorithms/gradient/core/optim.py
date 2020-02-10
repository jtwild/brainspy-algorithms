import torch 
import torch_optimizer as optim

def get_optimizer(parameters, configs):
    if configs['optimizer'] == 'yogi':
        return get_yogi(parameters,configs)
    elif configs['optimizer'] == 'adam':
        return get_adam(parameters,configs)


def get_yogi(parameters, configs):
    print('Prediction using YOGI optimizer')
    if "betas" in configs.keys():
        print("Set betas to values from the config file: ")
        print(*configs["betas"], sep=", ")
        return optim.Yogi(parameters,
                                            lr=configs['learning_rate'],
                                            betas=configs["betas"]
                                            )

    else:
        return optim.Yogi(parameters,
                                            lr=configs['learning_rate'])

def get_adam(parameters, configs):
    print('Prediction using ADAM optimizer')
    if "betas" in configs.keys():
        print("Set betas to values from the config file: ")
        print(*configs["betas"], sep=", ")
        return torch.optim.Adam(self.processor.parameters(),
                                            lr=configs['learning_rate'],
                                            betas=configs["betas"]
                                            )

    else:
        return torch.optim.Adam(self.processor.parameters(),
                                            lr=configs['learning_rate'])