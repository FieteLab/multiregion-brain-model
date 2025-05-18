class BaseWrapper:
    def __init__(self, env):
        self.env = env
        self.is_eval = False
        self.current_pos = -1
    
    @property
    def best_action(self):
        return self.env.best_action

    @property
    def step_size(self):
        return self.env.step_size

    @property
    def num_images(self):
        return self.env.num_images

    @property
    def current(self):
        return self.env.current

    @current.setter
    def current(self, value):
        self.env.current = value
    
    @property
    def interval(self):
        return self.env.interval

    def eval(self):
        self.is_eval = True

    def train(self):
        self.is_eval = False

    def get_seq_len(self):
        raise NotImplementedError

    def __len__(self):
        return self.env.__len__()

    def set_mental(self, val=False):
        return self.env.set_mental(val)

    def set_mode(self, mode):
        return self.env.set_mode(mode)

    def get_mode(self):
        return self.env.get_mode()