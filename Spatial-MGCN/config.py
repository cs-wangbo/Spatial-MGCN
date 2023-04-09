import configparser


class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # Parameter
        self.fdim = conf.getint("Data_Setting", "fdim")
        self.k = conf.getint("Model_Setup", "k")
        self.radius = conf.getint("Model_Setup", "radius")
        self.seed = conf.getint("Model_Setup", "seed")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.nhid1 = conf.getint("Model_Setup", "nhid1")
        self.nhid2 = conf.getint("Model_Setup", "nhid2")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.alpha = conf.getfloat("Model_Setup", "alpha")
        self.beta = conf.getfloat("Model_Setup", "beta")
        self.gamma = conf.getfloat("Model_Setup", "gamma")
        self.no_cuda = conf.getboolean("Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("Model_Setup", "no_seed")